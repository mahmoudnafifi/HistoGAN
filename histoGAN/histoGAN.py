"""
 If you find this code useful, please cite our paper:

 Mahmoud Afifi, Marcus A. Brubaker, and Michael S. Brown. "HistoGAN:
 Controlling Colors of GAN-Generated and Real Images via Color Histograms."
 In CVPR, 2021.

 @inproceedings{afifi2021histogan,
  title={Histo{GAN}: Controlling Colors of {GAN}-Generated and Real Images via
  Color Histograms},
  author={Afifi, Mahmoud and Brubaker, Marcus A. and Brown, Michael S.},
  booktitle={CVPR},
  year={2021}
}
"""

import json
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
from torch_optimizer import DiffGrad
from torch.autograd import grad as torch_grad
import torchvision
from torchvision import transforms
from vector_quantize_pytorch import VectorQuantize
from linear_attention_transformer import ImageLinearAttention
from PIL import Image
from pathlib import Path
from utils.diff_augment import DiffAugment

try:
  from apex import amp

  APEX_AVAILABLE = True
except:
  APEX_AVAILABLE = False

assert torch.cuda.is_available(), ('You need to have an Nvidia GPU with CUDA '
                                   'installed.')

num_cores = multiprocessing.cpu_count()

# constants
EXTS = ['jpg', 'png']
EPS = 1e-8
SCALE = 1 / np.sqrt(2.0)


# helper classes
class NanException(Exception):
  pass


class EMA():
  def __init__(self, beta):
    super().__init__()
    self.beta = beta

  def update_average(self, old, new):
    if old is None:
      return new
    return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
  def forward(self, x):
    return x.reshape(x.shape[0], -1)


class RandomApply(nn.Module):
  def __init__(self, prob, fn, fn_else=lambda x: x):
    super().__init__()
    self.fn = fn
    self.fn_else = fn_else
    self.prob = prob

  def forward(self, x):
    fn = self.fn if random() < self.prob else self.fn_else
    return fn(x)


class Residual(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x):
    return self.fn(x) + x


class Rezero(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
    self.g = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    return self.fn(x) * self.g


class PermuteToFrom(nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    out, loss = self.fn(x)
    out = out.permute(0, 3, 1, 2)
    return out, loss


# helpers

def default(value, d):
  return d if value is None else value


def cycle(iterable):
  while True:
    for i in iterable:
      yield i


def cast_list(el):
  return el if isinstance(el, list) else [el]


def is_empty(t):
  if isinstance(t, torch.Tensor):
    return t.nelement() == 0
  return t is None


def raise_if_nan(t):
  if torch.isnan(t):
    raise NanException


def loss_backwards(fp16, loss, optimizer, **kwargs):
  if fp16:
    with amp.scale_loss(loss, optimizer) as scaled_loss:
      scaled_loss.backward(**kwargs)
  else:
    loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10):
  batch_size = images.shape[0]
  gradients = torch_grad(outputs=output, inputs=images,
                         grad_outputs=torch.ones(output.size()).cuda(),
                         create_graph=True, retain_graph=True,
                         only_inputs=True)[0]
  gradients = gradients.reshape(batch_size, -1)
  return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def noise(n, latent_dim):
  return torch.randn(n, latent_dim).cuda()


def noise_list(n, layers, latent_dim):
  return [(noise(n, latent_dim), layers)]


def mixed_list(n, layers, latent_dim):
  tt = int(torch.rand(()).numpy() * layers)
  return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)


def hist_interpolation(hist1, hist2):
  ratio = torch.rand(1)
  return hist1 * ratio + hist2 * (1 - ratio)


def latent_to_w(style_vectorizer, latent_descr):
  return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size):
  return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()


def leaky_relu(p=0.2):
  return nn.LeakyReLU(p, inplace=True)


def slerp(val, low, high):
  low_norm = low / torch.norm(low, dim=1, keepdim=True)
  high_norm = high / torch.norm(high, dim=1, keepdim=True)
  omega = torch.acos((low_norm * high_norm).sum(1))
  so = torch.sin(omega)
  res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
      torch.sin(val * omega) / so).unsqueeze(1) * high
  return res


def evaluate_in_chunks(max_batch_size, model, *args):
  split_args = list(
    zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
  chunked_outputs = [model(*i) for i in split_args]
  if len(chunked_outputs) == 1:
    return chunked_outputs[0]
  return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
  return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def],
                   dim=1)


def set_requires_grad(model, bool):
  for p in model.parameters():
    p.requires_grad = bool


# dataset

def convert_rgb_to_transparent(image):
  if image.mode == 'RGB':
    return image.convert('RGBA')
  return image


def convert_transparent_to_rgb(image):
  if image.mode == 'RGBA':
    return image.convert('RGB')
  return image


class expand_greyscale(object):
  def __init__(self, num_channels):
    self.num_channels = num_channels

  def __call__(self, tensor):
    return tensor.expand(self.num_channels, -1, -1)


def resize_to_minimum_size(min_size, image):
  if max(*image.size) < min_size:
    return torchvision.transforms.functional.resize(image, min_size)
  return image


class Dataset(data.Dataset):
  def __init__(self, folder, image_size=256, transparent=False, hist_insz=150,
               hist_bin=64, hist_method='inverse-quadratic',
               hist_resizing='sampling', test=False, aug_prob=0.0):
    super().__init__()
    self.folder = folder
    self.image_size = image_size
    self.test = test
    self.paths = [p for ext in EXTS for p in
                  Path(f'{folder}').glob(f'**/*.{ext}')]
    self.histblock = RGBuvHistBlock(insz=hist_insz, h=hist_bin,
                                    method=hist_method, resizing=hist_resizing,
                                    device='cpu')
    set_requires_grad(self.histblock, False)
    convert_image_fn = (convert_transparent_to_rgb if not transparent else
                        convert_rgb_to_transparent)
    num_channels = 3 if not transparent else 4

    if self.test is False:
      self.transform = transforms.Compose([
        transforms.Lambda(convert_image_fn),
        transforms.Lambda(partial(resize_to_minimum_size, self.image_size)),
        transforms.Resize(self.image_size),
        RandomApply(aug_prob, transforms.RandomResizedCrop(
          self.image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)),
                    transforms.CenterCrop(self.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(expand_greyscale(num_channels))
      ])

    self.transform_hist = transforms.Compose([
      transforms.Lambda(convert_image_fn),
      transforms.ToTensor(),
      transforms.Lambda(expand_greyscale(num_channels))
    ])

  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):
    if self.test is False:
      path = self.paths[index]
      img = Image.open(path)
      inds = np.random.randint(0, high=len(self.paths), size=2)
      img1 = Image.open(self.paths[inds[0]])
      img2 = Image.open(self.paths[inds[1]])
      hist1 = self.histblock(torch.unsqueeze(self.transform_hist(img1), dim=0))
      hist2 = self.histblock(torch.unsqueeze(self.transform_hist(img2), dim=0))
      return {'images': self.transform(img),
              'histograms': torch.squeeze(hist_interpolation(hist1, hist2))}
    else:
      path = self.paths[index]
      img = Image.open(path)
      hist = self.histblock(torch.unsqueeze(self.transform_hist(img), dim=0))
      return {'histograms': torch.squeeze(hist)}


# augmentations

def random_hflip(tensor, prob):
  if prob > random():
    return tensor
  return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
  def __init__(self, D):
    super().__init__()
    self.D = D

  def forward(self, images, prob=0., types=[], detach=False):
    if random() < prob:
      images = random_hflip(images, prob=0.5)
      images = DiffAugment(images, types=types)

    if detach:
      images = images.detach()

    return self.D(images)


# Hist module
class RGBuvHistBlock(nn.Module):
  def __init__(self, h=64, insz=150, resizing='interpolation',
               method='thresholding', sigma=0.02, device='cuda'):
    """ Computes the RGB-uv histogram feature of a given image.
    Args:
      h: histogram dimension size (scalar). The default value is 64.
      insz: maximum size of the input image; if it is larger than this size, the
        image will be resized (scalar). Default value is 150 (i.e., 150 x 150
        pixels).
      resizing: resizing method if applicable. Options are: 'interpolation' or
        'sampling'. Default is 'interpolation'.
      method: the method used to count the number of pixels for each bin in the
        histogram feature. Options are: 'thresholding', 'RBF' (radial basis
        function), or 'inverse-quadratic'. Default value is 'inverse-quadratic'.
      sigma: if the method value is 'RBF' or 'inverse-quadratic', then this is
        the sigma parameter of the kernel function. The default value is 0.02.
      intensity_scale: boolean variable to use the intensity scale (I_y in
        Equation 2). Default value is True.

    Methods:
      forward: accepts input image and returns its histogram feature. Note that
        unless the method is 'thresholding', this is a differentiable function
        and can be easily integrated with the loss function. As mentioned in the
         paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
         training.
    """
    super(RGBuvHistBlock, self).__init__()
    self.h = h
    self.insz = insz
    self.device = device
    self.resizing = resizing
    self.method = method
    if self.method == 'thresholding':
      self.eps = 6.0 / h
    else:
      self.sigma = sigma

  def forward(self, x):
    x = torch.clamp(x, 0, 1)
    if x.shape[2] > self.insz or x.shape[3] > self.insz:
      if self.resizing == 'interpolation':
        x_sampled = F.interpolate(x, size=(self.insz, self.insz),
                                  mode='bilinear', align_corners=False)
      elif self.resizing == 'sampling':
        inds_1 = torch.LongTensor(
          np.linspace(0, x.shape[2], self.h, endpoint=False)).to(
          device=self.device)
        inds_2 = torch.LongTensor(
          np.linspace(0, x.shape[3], self.h, endpoint=False)).to(
          device=self.device)
        x_sampled = x.index_select(2, inds_1)
        x_sampled = x_sampled.index_select(3, inds_2)
      else:
        raise Exception(
          f'Wrong resizing method. It should be: interpolation or sampling. '
          f'But the given value is {self.resizing}.')
    else:
      x_sampled = x

    L = x_sampled.shape[0]  # size of mini-batch
    # print("size is %d" % L)
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]
    X = torch.unbind(x_sampled, dim=0)
    hists = torch.zeros((x_sampled.shape[0], 3, self.h, self.h)).to(
      device=self.device)
    for l in range(L):
      I = torch.t(torch.reshape(X[l], (3, -1)))
      II = torch.pow(I, 2)
      Iy = torch.unsqueeze(torch.sqrt(II[:, 0] + II[:, 1] + II[:, 2] + EPS),
                           dim=1)

      Iu0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 1] + EPS),
                            dim=1)
      Iv0 = torch.unsqueeze(torch.log(I[:, 0] + EPS) - torch.log(I[:, 2] + EPS),
                            dim=1)
      diff_u0 = abs(
        Iu0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      diff_v0 = abs(
        Iv0 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      if self.method == 'thresholding':
        diff_u0 = torch.reshape(diff_u0, (-1, self.h)) <= self.eps / 2
        diff_v0 = torch.reshape(diff_v0, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u0 = torch.exp(-diff_u0)  # Gaussian
        diff_v0 = torch.exp(-diff_v0)
      elif self.method == 'inverse-quadratic':
        diff_u0 = torch.pow(torch.reshape(diff_u0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v0 = torch.pow(torch.reshape(diff_v0, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u0 = 1 / (1 + diff_u0)  # Inverse quadratic
        diff_v0 = 1 / (1 + diff_v0)
      else:
        raise Exception(
          f'Wrong kernel method. It should be either thresholding, RBF, '
          f'inverse-quadratic. But the given value is {self.method}.')
      diff_u0 = diff_u0.type(torch.float32)
      diff_v0 = diff_v0.type(torch.float32)
      a = torch.t(Iy * diff_u0)
      hists[l, 0, :, :] = torch.mm(a, diff_v0)

      Iu1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 0] + EPS),
                            dim=1)
      Iv1 = torch.unsqueeze(torch.log(I[:, 1] + EPS) - torch.log(I[:, 2] + EPS),
                            dim=1)
      diff_u1 = abs(
        Iu1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      diff_v1 = abs(
        Iv1 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))

      if self.method == 'thresholding':
        diff_u1 = torch.reshape(diff_u1, (-1, self.h)) <= self.eps / 2
        diff_v1 = torch.reshape(diff_v1, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = torch.exp(-diff_u1)  # Gaussian
        diff_v1 = torch.exp(-diff_v1)
      elif self.method == 'inverse-quadratic':
        diff_u1 = torch.pow(torch.reshape(diff_u1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v1 = torch.pow(torch.reshape(diff_v1, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u1 = 1 / (1 + diff_u1)  # Inverse quadratic
        diff_v1 = 1 / (1 + diff_v1)

      diff_u1 = diff_u1.type(torch.float32)
      diff_v1 = diff_v1.type(torch.float32)
      a = torch.t(Iy * diff_u1)
      hists[l, 1, :, :] = torch.mm(a, diff_v1)

      Iu2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 0] + EPS),
                            dim=1)
      Iv2 = torch.unsqueeze(torch.log(I[:, 2] + EPS) - torch.log(I[:, 1] + EPS),
                            dim=1)
      diff_u2 = abs(
        Iu2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      diff_v2 = abs(
        Iv2 - torch.unsqueeze(torch.tensor(np.linspace(-3, 3, num=self.h)),
                              dim=0).to(self.device))
      if self.method == 'thresholding':
        diff_u2 = torch.reshape(diff_u2, (-1, self.h)) <= self.eps / 2
        diff_v2 = torch.reshape(diff_v2, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u2 = torch.exp(-diff_u2)  # Gaussian
        diff_v2 = torch.exp(-diff_v2)
      elif self.method == 'inverse-quadratic':
        diff_u2 = torch.pow(torch.reshape(diff_u2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_v2 = torch.pow(torch.reshape(diff_v2, (-1, self.h)),
                            2) / self.sigma ** 2
        diff_u2 = 1 / (1 + diff_u2)  # Inverse quadratic
        diff_v2 = 1 / (1 + diff_v2)
      diff_u2 = diff_u2.type(torch.float32)
      diff_v2 = diff_v2.type(torch.float32)
      a = torch.t(Iy * diff_u2)
      hists[l, 2, :, :] = torch.mm(a, diff_v2)

    # normalization
    hists_normalized = hists / (
        ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

    return hists_normalized


class HistVectorizer(nn.Module):
  def __init__(self, insize, emb, depth):
    super().__init__()
    self.flatten = Flatten()
    fc_layers = []
    for i in range(depth):
      if i == 0:
        fc_layers.extend(
          [nn.Linear(insize * insize * 3, emb * 2), leaky_relu()])
      elif i == 1:
        fc_layers.extend([nn.Linear(emb * 2, emb), leaky_relu()])
      else:
        fc_layers.extend([nn.Linear(emb, emb), leaky_relu()])
    self.fcs = nn.Sequential(*fc_layers)

  def forward(self, x):
    return self.fcs(self.flatten(x))


class StyleVectorizer(nn.Module):
  def __init__(self, emb, depth):
    super().__init__()

    layers = []
    for i in range(depth):
      layers.extend([nn.Linear(emb, emb), leaky_relu()])

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class RGBBlock(nn.Module):
  def __init__(self, latent_dim, input_channel, upsample, rgba=False):
    super().__init__()
    self.input_channel = input_channel
    self.to_style = nn.Linear(latent_dim, input_channel)

    out_filters = 3 if not rgba else 4
    self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False) if upsample else None

  def forward(self, x, prev_rgb, istyle):
    style = self.to_style(istyle)
    x = self.conv(x, style)

    if prev_rgb is not None:
      x = x + prev_rgb

    if self.upsample is not None:
      x = self.upsample(x)

    return x


class Conv2DMod(nn.Module):
  def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1,
               dilation=1, **kwargs):
    super().__init__()
    self.filters = out_chan
    self.demod = demod
    self.kernel = kernel
    self.stride = stride
    self.dilation = dilation
    self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
    nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in',
                            nonlinearity='leaky_relu')

  def _get_same_padding(self, size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

  def forward(self, x, y):
    b, c, h, w = x.shape

    w1 = y[:, None, :, None, None]
    w2 = self.weight[None, :, :, :, :]
    weights = w2 * (w1 + 1)

    if self.demod:
      d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
      weights = weights * d

    x = x.reshape(1, -1, h, w)

    _, _, *ws = weights.shape
    weights = weights.reshape(b * self.filters, *ws)

    padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
    x = F.conv2d(x, weights, padding=padding, groups=b)

    x = x.reshape(-1, self.filters, h, w)
    return x


class GeneratorBlock(nn.Module):
  def __init__(self, latent_dim, input_channels, filters, upsample=True,
               upsample_rgb=True, rgba=False):
    super().__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False) if upsample else None

    self.to_style1 = nn.Linear(latent_dim, input_channels)
    self.to_noise1 = nn.Linear(1, filters)
    self.conv1 = Conv2DMod(input_channels, filters, 3)

    self.to_style2 = nn.Linear(latent_dim, filters)
    self.to_noise2 = nn.Linear(1, filters)
    self.conv2 = Conv2DMod(filters, filters, 3)

    self.activation = leaky_relu()
    self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

  def forward(self, x, prev_rgb, istyle, inoise, latent=None):
    if self.upsample is not None:
      x = self.upsample(x)

    inoise = inoise[:, :x.shape[2], :x.shape[3], :]
    noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
    noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

    style1 = self.to_style1(istyle)
    x = self.conv1(x, style1)
    x = self.activation(x + noise1)
    if latent is not None:
      x = x + latent
    style2 = self.to_style2(istyle)
    x = self.conv2(x, style2)
    x = self.activation(x + noise2)

    rgb = self.to_rgb(x, prev_rgb, istyle)
    return x, rgb


class DiscriminatorBlock(nn.Module):
  def __init__(self, input_channels, filters, downsample=True):
    super().__init__()
    self.conv_res = nn.Conv2d(input_channels, filters, 1)

    self.net = nn.Sequential(
      nn.Conv2d(input_channels, filters, 3, padding=1),
      leaky_relu(),
      nn.Conv2d(filters, filters, 3, padding=1),
      leaky_relu()
    )

    self.downsample = nn.Conv2d(filters, filters, 3, padding=1,
                                stride=2) if downsample else None

  def forward(self, x):
    res = self.conv_res(x)
    x = self.net(x)
    x = x + res
    if self.downsample is not None:
      x = self.downsample(x)
    return x


class Generator(nn.Module):
  def __init__(self, image_size, latent_dim, network_capacity=16,
               transparent=False):
    super().__init__()
    self.image_size = image_size
    self.latent_dim = latent_dim
    self.num_layers = int(log2(image_size) - 1)

    init_channels = 4 * network_capacity
    self.initial_block = nn.Parameter(torch.randn((init_channels, 4, 4)))
    filters = [init_channels] + [network_capacity * (2 ** (i + 1)) for i in
                                 range(self.num_layers)][::-1]
    in_out_pairs = zip(filters[0:-1], filters[1:])

    self.blocks = nn.ModuleList([])
    for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
      not_first = ind != 0
      not_last = ind != (self.num_layers - 1)

      block = GeneratorBlock(
        latent_dim,
        in_chan,
        out_chan,
        upsample=not_first,
        upsample_rgb=not_last,
        rgba=transparent
      )
      self.blocks.append(block)

  def forward(self, styles, hists, input_noise):
    batch_size = styles.shape[0]
    x = self.initial_block.expand(batch_size, -1, -1, -1)
    styles = styles.transpose(0, 1)
    hists = hists.transpose(0, 1)
    styles = torch.cat((styles, hists), dim=0)

    rgb = None
    for style, block in zip(styles, self.blocks):
      x, rgb = block(x, rgb, style, input_noise)
    return rgb


class Discriminator(nn.Module):
  def __init__(self, image_size, network_capacity=16, fq_layers=[],
               fq_dict_size=256, attn_layers=[],
               transparent=False):
    super().__init__()
    num_layers = int(log2(image_size) - 1)
    num_init_filters = 3 if not transparent else 4

    blocks = []
    filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in
                                    range(num_layers + 1)]
    chan_in_out = list(zip(filters[0:-1], filters[1:]))

    blocks = []
    quantize_blocks = []
    attn_blocks = []

    for ind, (in_chan, out_chan) in enumerate(chan_in_out):
      num_layer = ind + 1
      is_not_last = ind != (len(chan_in_out) - 1)

      block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
      blocks.append(block)

      attn_fn = nn.Sequential(*[
        Residual(Rezero(ImageLinearAttention(out_chan))) for _ in range(2)
      ]) if num_layer in attn_layers else None

      attn_blocks.append(attn_fn)

      quantize_fn = PermuteToFrom(VectorQuantize(
        out_chan, fq_dict_size)) if num_layer in fq_layers else None
      quantize_blocks.append(quantize_fn)

    self.blocks = nn.ModuleList(blocks)
    self.attn_blocks = nn.ModuleList(attn_blocks)
    self.quantize_blocks = nn.ModuleList(quantize_blocks)

    latent_dim = 2 * 2 * filters[-1]

    self.flatten = Flatten()
    self.to_logit = nn.Linear(latent_dim, 1)

  def forward(self, x):
    b, *_ = x.shape

    quantize_loss = torch.zeros(1).to(x)

    for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks,
                                            self.quantize_blocks):
      x = block(x)

      if attn_block is not None:
        x = attn_block(x)

      if q_block is not None:
        x, loss = q_block(x)
        quantize_loss += loss

    x = self.flatten(x)
    x = self.to_logit(x)
    return x.squeeze(), quantize_loss


class HistoGAN(nn.Module):
  def __init__(self, image_size, latent_dim=512, style_depth=8,
               network_capacity=16, transparent=False, fp16=False,
               steps=1, lr=1e-4, fq_layers=[], fq_dict_size=256, attn_layers=[],
               aug=False, hist=64):
    super().__init__()

    self.lr = lr
    self.aug = aug
    self.steps = steps
    self.ema_updater = EMA(0.995)
    self.S = StyleVectorizer(latent_dim, style_depth)
    self.H = HistVectorizer(hist, latent_dim, int(style_depth))
    self.G = Generator(image_size, latent_dim, network_capacity,
                       transparent=transparent)
    self.D = Discriminator(image_size, network_capacity, fq_layers=fq_layers,
                           fq_dict_size=fq_dict_size,
                           attn_layers=attn_layers, transparent=transparent)

    self.SE = StyleVectorizer(latent_dim, style_depth)
    self.HE = HistVectorizer(hist, latent_dim, int(style_depth))
    self.GE = Generator(image_size, latent_dim, network_capacity,
                        transparent=transparent)

    # wrapper for augmenting all images going into the discriminator
    if self.aug:
      self.D_aug = AugWrapper(self.D)
    else:
      self.D_aug = None

    set_requires_grad(self.SE, False)
    set_requires_grad(self.HE, False)
    set_requires_grad(self.GE, False)

    generator_params = list(self.G.parameters()) + list(
      self.S.parameters()) + list(self.H.parameters())
    self.G_opt = DiffGrad(generator_params, lr=self.lr, betas=(0.5, 0.9))
    self.D_opt = DiffGrad(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

    self._init_weights()
    self.reset_parameter_averaging()

    self.cuda()

    if fp16:
      (self.S, self.G, self.D, self.H, self.SE, self.HE, self.GE), (
        self.G_opt, self.D_opt) = \
        amp.initialize(
          [self.S, self.G, self.D, self.H, self.SE, self.HE, self.GE],
          [self.G_opt, self.D_opt],
          opt_level='O2')

  def _init_weights(self):
    for m in self.modules():
      if type(m) in {nn.Conv2d, nn.Linear}:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in',
                                nonlinearity='leaky_relu')

    for block in self.G.blocks:
      nn.init.zeros_(block.to_noise1.weight)
      nn.init.zeros_(block.to_noise2.weight)
      nn.init.zeros_(block.to_noise1.bias)
      nn.init.zeros_(block.to_noise2.bias)

  def EMA(self):
    def update_moving_average(ma_model, current_model):
      for current_params, ma_params in zip(current_model.parameters(),
                                           ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

    update_moving_average(self.SE, self.S)
    update_moving_average(self.HE, self.H)
    update_moving_average(self.GE, self.G)

  def reset_parameter_averaging(self):
    self.SE.load_state_dict(self.S.state_dict())
    self.HE.load_state_dict(self.H.state_dict())
    self.GE.load_state_dict(self.G.state_dict())

  def forward(self, x):
    return x


class Trainer():
  def __init__(self, name, results_dir, models_dir, image_size,
               network_capacity, transparent=False, batch_size=4,
               mixed_prob=0.9, gradient_accumulate_every=1, lr=2e-4,
               num_workers=None, save_every=1000, trunc_psi=0.6,
               fp16=False, fq_layers=[], fq_dict_size=256, attn_layers=[],
               hist_method='inverse-quadratic',
               hist_resizing='sampling', hist_sigma=0.02, hist_bin=64,
               hist_insz=150, aug_prob=0.0, dataset_aug_prob=0.0,
               aug_types=None, *args, **kwargs):

    if aug_types is None:
      aug_types = ['translation', 'cutout']
    self.GAN_params = [args, kwargs]
    self.GAN = None
    self.hist_method = hist_method
    self.hist_resizing = hist_resizing
    self.hist_sigma = hist_sigma
    self.hist_bin = hist_bin
    self.hist_insz = hist_insz
    self.histBlock = RGBuvHistBlock(insz=self.hist_insz, h=self.hist_bin,
                                    method=self.hist_method,
                                    resizing=self.hist_resizing,
                                    sigma=self.hist_sigma)
    set_requires_grad(self.histBlock, True)
    self.name = name
    self.results_dir = Path(results_dir)
    self.models_dir = Path(models_dir)
    self.config_path = self.models_dir / name / '.config.json'

    assert log2(
      image_size).is_integer(), ('image size must be a power of 2 (64, 128, '
                                 '256, 512, 1024)')
    self.image_size = image_size
    self.network_capacity = network_capacity
    self.transparent = transparent
    self.fq_layers = cast_list(fq_layers)
    self.fq_dict_size = fq_dict_size

    self.attn_layers = cast_list(attn_layers)

    self.aug_prob = aug_prob
    self.aug_types = aug_types
    self.dataset_aug_prob = dataset_aug_prob

    self.lr = lr
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.mixed_prob = mixed_prob

    self.save_every = save_every
    self.steps = 0

    self.av = None
    self.trunc_psi = trunc_psi

    self.pl_mean = 0

    self.gradient_accumulate_every = gradient_accumulate_every

    assert not fp16 or fp16 and APEX_AVAILABLE, ('Apex is not available for '
                                                 'you to use mixed precision '
                                                 'training')
    self.fp16 = fp16

    self.d_loss = 0
    self.g_loss = 0
    self.last_gp_loss = 0
    self.last_cr_loss = 0
    self.q_loss = 0

    self.pl_length_ma = EMA(0.99)
    self.init_folders()

    self.loader = None

    self.loader_evaluate = None

  def init_GAN(self):
    args, kwargs = self.GAN_params
    self.GAN = HistoGAN(lr=self.lr, image_size=self.image_size,
                        network_capacity=self.network_capacity,
                        transparent=self.transparent,
                        fq_layers=self.fq_layers,
                        fq_dict_size=self.fq_dict_size,
                        attn_layers=self.attn_layers, fp16=self.fp16,
                        hist=self.hist_bin, aug=self.aug_prob > 0,
                        *args, **kwargs)

  def write_config(self):
    self.config_path.write_text(json.dumps(self.config()))

  def load_config(self):
    config = self.config() if not self.config_path.exists() else json.loads(
      self.config_path.read_text())
    self.image_size = config['image_size']
    self.network_capacity = config['network_capacity']
    self.transparent = config['transparent']
    self.fq_layers = config['fq_layers']
    self.fq_dict_size = config['fq_dict_size']
    self.attn_layers = config.pop('attn_layers', [])
    del self.GAN
    self.init_GAN()

  def config(self):
    return {'image_size': self.image_size,
            'network_capacity': self.network_capacity,
            'transparent': self.transparent, 'fq_layers': self.fq_layers,
            'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers}

  def set_data_src(self, folder):
    self.dataset = Dataset(folder, image_size=self.image_size,
                           transparent=self.transparent,
                           hist_insz=self.hist_insz, hist_bin=self.hist_bin,
                           hist_method=self.hist_method,
                           hist_resizing=self.hist_resizing,
                           aug_prob=self.dataset_aug_prob)
    self.loader = cycle(data.DataLoader(self.dataset,
                                        num_workers=default(self.num_workers,
                                                            num_cores),
                                        batch_size=self.batch_size,
                                        drop_last=True, shuffle=True,
                                        pin_memory=True))
    self.dataset_evaluate = Dataset(folder, image_size=150,
                                    transparent=self.transparent,
                                    hist_insz=self.hist_insz,
                                    hist_bin=self.hist_bin,
                                    hist_method=self.hist_method,
                                    hist_resizing=self.hist_resizing, test=True)
    self.loader_evaluate = cycle(data.DataLoader(self.dataset_evaluate,
                                                 num_workers=default(
                                                   self.num_workers, num_cores),
                                                 batch_size=4,
                                                 drop_last=True, shuffle=True,
                                                 pin_memory=True))

  def train(self, alpha=2):
    assert self.loader is not None, ('You must first initialize the data '
                                     'source with `. set_data_src(<folder of '
                                     'images>)`')

    torch.autograd.set_detect_anomaly(False)

    if self.GAN is None:
      self.init_GAN()
    self.GAN.train()
    total_disc_loss = torch.tensor(0.).cuda()
    total_gen_loss = torch.tensor(0.).cuda()
    total_hist_loss = torch.tensor(0.).cuda()

    batch_size = self.batch_size

    image_size = self.GAN.G.image_size
    latent_dim = self.GAN.G.latent_dim
    num_layers = self.GAN.G.num_layers

    aug_prob = self.aug_prob

    if aug_prob > 0.0:
      Disc = self.GAN.D_aug
      aug_types = self.aug_types
      aug_kwargs = {'prob': aug_prob, 'types': aug_types}
    else:
      Disc = self.GAN.D

    apply_gradient_penalty = self.steps % 4 == 0
    apply_path_penalty = self.steps % 32 == 0

    backwards = partial(loss_backwards, self.fp16)

    # train discriminator
    avg_pl_length = self.pl_mean
    self.GAN.D_opt.zero_grad()

    for i in range(self.gradient_accumulate_every):
      get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
      style = get_latents_fn(batch_size, num_layers - 2, latent_dim)
      noise = image_noise(batch_size, image_size)
      batch = next(self.loader)
      image_batch = batch['images'].cuda()
      image_batch.requires_grad_()
      hist_batch = batch['histograms'].cuda()
      w_space = latent_to_w(self.GAN.S, style)
      h_w_space = self.GAN.H(hist_batch)
      h_w_space = torch.unsqueeze(h_w_space, dim=1)
      h_w_space = torch.cat((h_w_space, h_w_space), dim=1)
      w_styles = styles_def_to_tensor(w_space)
      generated_images = self.GAN.G(w_styles, h_w_space, noise)
      if aug_prob > 0.0:
        fake_output, fake_q_loss = Disc(generated_images.clone().detach(),
                                         detach=True, **aug_kwargs)
        real_output, real_q_loss = Disc(image_batch, **aug_kwargs)
      else:
        fake_output, fake_q_loss = Disc(generated_images.clone().detach())
        real_output, real_q_loss = Disc(image_batch)

      divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
      disc_loss = divergence
      quantize_loss = (fake_q_loss + real_q_loss).mean()
      self.q_loss = float(quantize_loss.detach().item())
      disc_loss = disc_loss + quantize_loss

      if apply_gradient_penalty:
        gp = gradient_penalty(image_batch, real_output)
        self.last_gp_loss = gp.clone().detach().item()
        disc_loss = disc_loss + gp

      disc_loss = disc_loss / self.gradient_accumulate_every
      disc_loss.register_hook(raise_if_nan)
      backwards(disc_loss, self.GAN.D_opt)

      total_disc_loss += (divergence.detach().item() /
                          self.gradient_accumulate_every)

    self.d_loss = float(total_disc_loss)
    self.GAN.D_opt.step()

    # train generator
    self.GAN.G_opt.zero_grad()
    for i in range(self.gradient_accumulate_every):
      style = get_latents_fn(batch_size, num_layers - 2, latent_dim)
      noise = image_noise(batch_size, image_size)
      batch = next(self.loader)
      hist_batch = batch['histograms'].cuda()
      hist_batch.requires_grad_()

      h_w_space = self.GAN.H(hist_batch)
      h_w_space = torch.unsqueeze(h_w_space, dim=1)
      h_w_space = torch.cat((h_w_space, h_w_space), dim=1)
      w_space = latent_to_w(self.GAN.S, style)
      w_styles = styles_def_to_tensor(w_space)

      generated_images = self.GAN.G(w_styles, h_w_space, noise)
      if aug_prob > 0.0:
        fake_output, _ = Disc(generated_images, **aug_kwargs)
      else:
        fake_output, _ = Disc(generated_images)

      generated_histograms = self.histBlock(F.relu(generated_images))

      histogram_loss = alpha * SCALE * (torch.sqrt(
        torch.sum(
          torch.pow(torch.sqrt(hist_batch) - torch.sqrt(generated_histograms),
                    2)))) / hist_batch.shape[0]

      loss = fake_output.mean()
      gen_loss = loss + histogram_loss

      if apply_path_penalty:
        std = 0.1 / (w_styles.std(dim=0, keepdim=True) + EPS)
        w_styles_2 = w_styles + torch.randn(w_styles.shape).cuda() / (std + EPS)
        pl_images = self.GAN.G(w_styles_2, h_w_space, noise)
        pl_lengths = ((pl_images - generated_images) ** 2).mean(dim=(1, 2, 3))
        avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

        if not is_empty(self.pl_mean):
          pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
          if not torch.isnan(pl_loss):
            gen_loss = gen_loss + pl_loss

      gen_loss = gen_loss / self.gradient_accumulate_every
      gen_loss.register_hook(raise_if_nan)
      backwards(gen_loss, self.GAN.G_opt)

      total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

      total_hist_loss += (histogram_loss.detach().item() /
                          self.gradient_accumulate_every)

    self.g_loss = float(total_gen_loss)

    self.h_loss = float(total_hist_loss)
    self.GAN.G_opt.step()

    # calculate moving averages
    if apply_path_penalty and not np.isnan(avg_pl_length):
      self.pl_mean = self.pl_length_ma.update_average(self.pl_mean,
                                                      avg_pl_length)

    if self.steps % 10 == 0 and self.steps > 20000:
      self.GAN.EMA()

    if self.steps <= 25000 and self.steps % 1000 == 2:
      self.GAN.reset_parameter_averaging()

    # save from NaN errors
    checkpoint_num = floor(self.steps / self.save_every)

    if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
      print(
        f'NaN detected for generator or discriminator. Loading from '
        f'checkpoint #{checkpoint_num}')
      self.load(checkpoint_num)
      raise NanException

    # periodically save results
    if self.steps % self.save_every == 0:
      self.save(checkpoint_num)

    if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
      self.evaluate(floor(self.steps / 1000))

    self.steps += 1
    self.av = None

  @torch.no_grad()
  def evaluate(self, num=0, hist_batch=None, num_image_tiles=4,
               latents=None, n=None, save_noise_latent=False,
               load_noise_file=None, load_latent_file=None):
    self.GAN.eval()

    if hist_batch is None:
      batch = next(self.loader_evaluate)
      hist_batch = batch['histograms'].cuda()

    ext = 'jpg' if not self.transparent else 'png'
    num_rows = num_image_tiles

    if latents is None and n is None:
      latent_dim = self.GAN.G.latent_dim
      image_size = self.GAN.G.image_size
      num_layers = self.GAN.G.num_layers

      # latents and noise
      if load_noise_file is not None:
        n = torch.tensor(np.load(load_noise_file)).cuda()
      else:
        n = image_noise(num_rows ** 2, image_size)
      if load_latent_file is not None:
        latents = np.load(load_latent_file)
      else:
        latents = noise_list(num_rows ** 2, num_layers - 2, latent_dim)

    generated_images = self.generate_truncated(
      self.GAN.SE, self.GAN.HE, self.GAN.GE, hist_batch, latents, n,
      trunc_psi=self.trunc_psi)
    if num is not None:
      torchvision.utils.save_image(generated_images,
                                   str(self.results_dir / self.name /
                                       f'{str(num)}-ema.{ext}'),
                                   nrow=num_rows)
    if save_noise_latent:
      np.save(f'temp/{self.name}/{str(num)}-noise.npy', n.clone().cpu().numpy())
      np.save(f'temp/{self.name}/{str(num)}-latents.npy', latents)

    return generated_images

  @torch.no_grad()
  def generate_truncated(self, S, H, G, hist_batch, style, noi, trunc_psi=0.75):
    latent_dim = G.latent_dim

    if self.av is None:
      z = noise(2000, latent_dim)
      samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
      self.av = np.mean(samples, axis=0)
      self.av = np.expand_dims(self.av, axis=0)

    w_space = []
    for tensor, num_layers in style:
      tmp = S(tensor)
      av_torch = torch.from_numpy(self.av).cuda()
      tmp = trunc_psi * (tmp - av_torch) + av_torch
      w_space.append((tmp, num_layers))

    h_w_space = H(hist_batch)
    h_w_space = torch.unsqueeze(h_w_space, dim=1)
    h_w_space = torch.cat((h_w_space, h_w_space), dim=1)

    for i in range(int(np.log2(np.sqrt(w_space[0][0].shape[0])))):
      h_w_space = torch.cat((h_w_space, h_w_space), dim=0)

    w_styles = styles_def_to_tensor(w_space)
    generated_images = evaluate_in_chunks(self.batch_size, G, w_styles,
                                          h_w_space, noi)
    return generated_images.clamp_(0., 1.)

  def print_log(self):
    if hasattr(self, 'h_loss'):
      print(
        f'\nG: {self.g_loss:.2f} | H: {self.h_loss:.2f} | D: '
        f'{self.d_loss:.2f} | GP: {self.last_gp_loss:.2f}'
        f' | PL: {self.pl_mean:.2f} | CR: {self.last_cr_loss:.2f} | Q: '
        f'{self.q_loss:.2f}')
    else:
      print(
        f'\nG: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: '
        f'{self.last_gp_loss:.2f}'
        f' | PL: {self.pl_mean:.2f} | CR: {self.last_cr_loss:.2f} | Q: '
        f'{self.q_loss:.2f}')

  def model_name(self, num):
    return str(self.models_dir / self.name / f'model_{num}.pt')

  def init_folders(self):
    (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
    (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

  def clear(self):
    rmtree(f'./models/{self.name}', True)
    rmtree(f'./results/{self.name}', True)
    rmtree(str(self.config_path), True)
    self.init_folders()

  def save(self, num):
    torch.save(self.GAN.state_dict(), self.model_name(num))
    self.write_config()

  def load(self, num=-1):
    self.load_config()

    name = num
    if num == -1:
      file_paths = [p for p in
                    Path(self.models_dir / self.name).glob('model_*.pt')]
      saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
      if len(saved_nums) == 0:
        return
      name = saved_nums[-1]
      print(f'continuing from previous epoch - {name}')
    self.steps = name * self.save_every
    self.GAN.load_state_dict(
      torch.load(self.model_name(name),
                 map_location=f'cuda:{torch.cuda.current_device()}'))
