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
from math import floor, log2, sqrt, pi
from shutil import rmtree
from functools import partial
import multiprocessing
import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F
import os
from utils import color_transfer_MKL as ct
import utils.pyramid_upsampling as upsampling
from torch_optimizer import DiffGrad
from torch.autograd import grad as torch_grad
import torchvision
from torchvision import transforms
from histoGAN import GeneratorBlock, HistVectorizer, NanException, \
  Discriminator, Conv2DMod
from histogram_classes.RGBuvHistBlock import RGBuvHistBlock
from PIL import Image
from pathlib import Path

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


class Flatten(nn.Module):
  def forward(self, x):
    return x.reshape(x.shape[0], -1)


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

  gradients = gradients.view(batch_size, -1)
  return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def noise(n, latent_dim):
  return torch.randn(n, latent_dim).cuda()


def noise_list(n, layers, latent_dim):
  return [(noise(n, latent_dim), layers)]


def mixed_list(n, layers, latent_dim):
  tt = int(torch.rand(()).numpy() * layers)
  return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt,
                                                    latent_dim)


def hist_interpolation(hist1, hist2):
  ratio = torch.rand(1)
  return hist1 * ratio + hist2 * (1 - ratio)


def latent_to_w(style_vectorizer, latent_descr):
  return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size):
  return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0.0, 1.).cuda()


def leaky_relu(p=0.2):
  return nn.LeakyReLU(p, inplace=True)


def slerp(val, low, high):
  low_norm = low / torch.norm(low, dim=1, keepdim=True)
  high_norm = high / torch.norm(high, dim=1, keepdim=True)
  omega = torch.acos((low_norm * high_norm).sum(1))
  so = torch.sin(omega)
  res = ((torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
      torch.sin(val * omega) / so).unsqueeze(1) * high)
  return res


def evaluate_in_chunks(max_batch_size, model, *args):
  split_args = list(zip(*list(map(lambda x: x.split(max_batch_size,
                                                    dim=0), args))))
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


def get_gaussian_kernel(kernel_size=15, sigma=3, channels=3):
  x_coord = torch.arange(kernel_size)
  x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
  y_grid = x_grid.t()
  xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
  mean = (kernel_size - 1) / 2.
  variance = sigma ** 2.
  gaussian_kernel = (1. / (2. * pi * variance)) * torch.exp(
    -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
  gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
  gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
  gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
  gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                              kernel_size=kernel_size, groups=channels,
                              bias=False)
  gaussian_filter.weight.data = gaussian_kernel
  gaussian_filter.weight.requires_grad = False

  return gaussian_filter


def gaussian_op(x, kernel=None):
  if kernel is None:
    kernel = get_gaussian_kernel(kernel_size=15, sigma=15, channels=3).to(
      device=torch.cuda.current_device())
  return kernel(x)


def laplacian_op(x, kernel=None):
  if kernel is None:
    laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    channels = x.size()[1]
    kernel = torch.tensor(laplacian,
                          dtype=torch.float32).unsqueeze(0).expand(
      1, channels, 3, 3).to(device=torch.cuda.current_device())
  return F.conv2d(x, kernel, stride=1, padding=1)


def sobel_op(x, dir=0, kernel=None):
  if kernel is None:
    if dir == 0:
      sobel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]  # x
    elif dir == 1:
      sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # y
    channels = x.size()[1]
    kernel = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(
      1, channels, 3, 3).to(device=torch.cuda.current_device())
  return F.conv2d(x, kernel, stride=1, padding=1)


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


class reconstruction_loss(object):
  def __init__(self, loss):
    self.loss = loss
    if self.loss == '1st gradient':
      sobel_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]  # x
      sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]  # y
      sobel_x = torch.tensor(
        sobel_x, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      sobel_y = torch.tensor(
        sobel_y, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      self.kernel1 = sobel_x
      self.kernel2 = sobel_y
    elif self.loss == '2nd gradient':
      laplacian = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
      self.kernel1 = torch.tensor(
        laplacian, dtype=torch.float32).unsqueeze(0).expand(
        1, 3, 3, 3).to(device=torch.cuda.current_device())
      self.kernel2 = None
    else:
      self.kernel1 = None
      self.kernel2 = None

  def compute_loss(self, input, target):
    if self.loss == 'L1':
      reconstruction_loss = torch.mean(torch.abs(input - target))  # L1

    elif self.loss == '1st gradient':
      input_dfdx = sobel_op(input, kernel=self.kernel1)
      input_dfdy = sobel_op(input, kernel=self.kernel2)
      target_dfdx = sobel_op(target, kernel=self.kernel1)
      target_dfdy = sobel_op(target, kernel=self.kernel2)
      input_gradient = torch.sqrt(torch.pow(input_dfdx, 2) +
                                  torch.pow(input_dfdy, 2))
      target_gradient = torch.sqrt(torch.pow(
        target_dfdx, 2) + torch.pow(target_dfdy, 2))
      reconstruction_loss = torch.mean(torch.abs(
        input_gradient - target_gradient))  # L1

    elif self.loss == '2nd gradient':
      input_lap = laplacian_op(input, kernel=self.kernel1)
      target_lap = laplacian_op(target, kernel=self.kernel1)
      reconstruction_loss = torch.mean(torch.abs(input_lap - target_lap))  # L1
    else:
      reconstruction_loss = None

    return reconstruction_loss


def resize_to_minimum_size(min_size, image):
  if max(*image.size) < min_size:
    return torchvision.transforms.functional.resize(image, min_size)
  return image


class Dataset(data.Dataset):
  def __init__(self, folder, image_size=256, transparent=False,
               hist_insz=150, hist_bin=64, hist_sampling=True,
               hist_method='inverse-quadratic', hist_resizing='sampling',
               triple_hist=False,
               double_hist=False):
    super().__init__()
    self.folder = folder
    self.image_size = image_size
    self.paths = [p for ext in EXTS for p in Path(
      f'{folder}').glob(f'**/*.{ext}')]
    self.histblock = RGBuvHistBlock(
      insz=hist_insz, h=hist_bin, method=hist_method,
      resizing=hist_resizing, device='cpu')
    self.hist_sampling = hist_sampling

    self.triple_hist = triple_hist
    self.double_hist = double_hist
    set_requires_grad(self.histblock, False)
    convert_image_fn = (convert_transparent_to_rgb if not transparent else
                        convert_rgb_to_transparent)
    num_channels = 3 if not transparent else 4

    self.transform = transforms.Compose([
      transforms.Lambda(convert_image_fn),
      transforms.Lambda(partial(resize_to_minimum_size, image_size)),
      transforms.RandomHorizontalFlip(),
      transforms.Resize(image_size),
      transforms.CenterCrop(image_size),
      transforms.ToTensor(),
      transforms.Lambda(expand_greyscale(num_channels))
    ])

    self.transform_hist = transforms.Compose([
      transforms.Lambda(convert_image_fn), transforms.ToTensor(),
      transforms.Lambda(expand_greyscale(num_channels))
    ])


  def __len__(self):
    return len(self.paths)

  def __getitem__(self, index):

    path = self.paths[index]
    img = Image.open(path)
    if self.hist_sampling is True:
      if self.triple_hist is True:
        inds = np.random.randint(0, high=len(self.paths), size=6)
        img1 = Image.open(self.paths[inds[0]])
        img2 = Image.open(self.paths[inds[1]])
        img3 = Image.open(self.paths[inds[2]])
        img4 = Image.open(self.paths[inds[3]])
        img5 = Image.open(self.paths[inds[4]])
        img6 = Image.open(self.paths[inds[5]])

        hist1 = self.histblock(torch.unsqueeze(
          self.transform_hist(img1), dim=0))
        hist2 = self.histblock(torch.unsqueeze(
          self.transform_hist(img2), dim=0))
        hist3 = self.histblock(torch.unsqueeze(
          self.transform_hist(img3), dim=0))
        hist4 = self.histblock(torch.unsqueeze(
          self.transform_hist(img4), dim=0))
        hist5 = self.histblock(torch.unsqueeze(
          self.transform_hist(img5), dim=0))
        hist6 = self.histblock(torch.unsqueeze(
          self.transform_hist(img6), dim=0))
        return {'images': self.transform(img),
                'histograms': torch.squeeze(hist_interpolation(hist1, hist2)),
                'histograms2': torch.squeeze(hist_interpolation(hist3, hist4)),
                'histograms3': torch.squeeze(hist_interpolation(hist5, hist6))}


      elif self.double_hist is True:
        inds = np.random.randint(0, high=len(self.paths), size=4)
        img1 = Image.open(self.paths[inds[0]])
        img2 = Image.open(self.paths[inds[1]])
        img3 = Image.open(self.paths[inds[2]])
        img4 = Image.open(self.paths[inds[3]])

        hist1 = self.histblock(torch.unsqueeze(
          self.transform_hist(img1), dim=0))
        hist2 = self.histblock(torch.unsqueeze(
          self.transform_hist(img2), dim=0))
        hist3 = self.histblock(torch.unsqueeze(
          self.transform_hist(img3), dim=0))
        hist4 = self.histblock(torch.unsqueeze(
          self.transform_hist(img4), dim=0))

        return {'images': self.transform(img),
                'histograms': torch.squeeze(hist_interpolation(
                  hist1, hist2)),
                'histograms2': torch.squeeze(hist_interpolation(
                  hist3, hist4))}
      else:
        inds = np.random.randint(0, high=len(self.paths), size=2)
        img1 = Image.open(self.paths[inds[0]])
        img2 = Image.open(self.paths[inds[1]])
        hist1 = self.histblock(torch.unsqueeze(
          self.transform_hist(img1), dim=0))
        hist2 = self.histblock(torch.unsqueeze(
          self.transform_hist(img2), dim=0))
        return {'images': self.transform(img),
                'histograms': torch.squeeze(
                  hist_interpolation(hist1, hist2))}

    else:
      hist = self.histblock(torch.unsqueeze(
        self.transform_hist(img), dim=0))
      return {'images': self.transform(img),
              'histograms': torch.squeeze(hist)}


class RecoloringGAN(nn.Module):
  def __init__(self, image_size, latent_dim, network_capacity=16,
               transparent=False):
    super().__init__()
    self.image_size = image_size
    self.latent_dim = latent_dim
    self.num_layers = int(log2(image_size) - 1)
    init_channels = 4 * network_capacity
    filters = [init_channels] + [network_capacity * (2 ** (i + 1))
                                 for i in range(self.num_layers)][::-1]
    filters = filters[-3:]
    in_out_pairs = zip(filters[0:-1], filters[1:])
    self.num_layers = 2

    self.blocks = nn.ModuleList([])
    for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
      not_first = True
      not_last = ind != 1

      block = GeneratorBlock(
        latent_dim,
        in_chan,
        out_chan,
        upsample=not_first,
        upsample_rgb=not_last,
        rgba=transparent
      )
      self.blocks.append(block)

  def forward(self, x, rgb, hists, input_noise, latent1=None, latent2=None):
    rgb = None
    x, rgb = self.blocks[0](x, rgb, hists, input_noise, latent=latent1)
    x, rgb = self.blocks[1](x, rgb, hists, input_noise, latent=latent2)
    return rgb


class EncoderBlock(nn.Module):
  def __init__(self, input_channels, filters):
    super().__init__()
    self.conv_res = nn.Conv2d(input_channels, filters, 1)
    self.net = nn.Sequential(
      nn.Conv2d(input_channels, filters, 3, padding=1),
      nn.InstanceNorm2d(filters),
      leaky_relu(),
      nn.Conv2d(filters, filters, 3, padding=1),
      nn.InstanceNorm2d(filters),
      leaky_relu()
    )
    self.downsample = nn.Conv2d(filters, filters, 3, padding=1, stride=2)

  def forward(self, x):
    res = self.conv_res(x)
    x = self.net(x)
    x = x + res
    x_d = self.downsample(x)
    return x_d, x


class DecoderBlock(nn.Module):
  def __init__(self, input_channels, filters, internal_hist=False,
               latent_dim=None):
    super().__init__()
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                align_corners=False)
    self.conv_res = nn.Conv2d(input_channels, filters, 1)
    self.block1 = nn.Sequential(
      nn.Conv2d(input_channels, input_channels, 3, padding=1),
      leaky_relu()
    )
    self.block2 = nn.Sequential(
      nn.Conv2d(input_channels * 2, filters, 3, padding=1),
      leaky_relu()
    )
    self.conv_out_latent = nn.Sequential(
      nn.Conv2d(filters, filters, 3, padding=1),
      leaky_relu())
    self.conv_out_rgb = nn.Conv2d(filters, 3, 1)

    if internal_hist:
      self.to_latent = nn.Linear(latent_dim, input_channels)
      self.conv_latent = Conv2DMod(input_channels, input_channels, 3)
    else:
      self.to_latent = None
      self.conv_latent = None

  def forward(self, x, prev_rgb, prev_latent, h=None):
    curr_latent = self.block1(x)
    if self.to_latent is not None:
      prev_latent = self.conv_latent(prev_latent, self.to_latent(h))
    processed_x = self.block2(torch.cat((curr_latent, prev_latent), dim=1))
    x_res = self.conv_res(x)
    x = self.conv_out_latent(x_res + processed_x)
    rgb = self.conv_out_rgb(x)
    if prev_rgb is not None:
      rgb = rgb + prev_rgb
    rgb = self.upsample(rgb)
    x = self.upsample(x)
    return x, rgb


class RecoloringEncoderDecoder(nn.Module):
  def __init__(self, image_size, network_capacity=16, hist=64,
               latent_dim=512, style_depth=8,
               skip_conn_to_GAN=False, internal_hist=False):
    super().__init__()
    self.image_size = image_size
    self.encoder_num_layers = int(log2(image_size) - 2)
    self.decoder_num_layers = int(log2(image_size) - 4)
    self.skip_conn_to_GAN = skip_conn_to_GAN
    self.internal_hist = internal_hist
    encoder_filters = [network_capacity] + \
                      [network_capacity * (2 ** (i + 1))
                       for i in range(self.encoder_num_layers)]

    encoder_in_out_pairs = zip(encoder_filters[0:-1], encoder_filters[1:])

    decoder_filters = encoder_filters
    decoder_filters.reverse()
    decoder_filters = decoder_filters[:-(self.encoder_num_layers -
                                         self.decoder_num_layers)]
    decoder_in_out_pairs = zip(decoder_filters[0:-1], decoder_filters[1:])

    self.encoder_blocks = nn.ModuleList([])
    self.decoder_blocks = nn.ModuleList([])
    self.decoder_mapping = nn.Conv2d(decoder_filters[-1],
                                     8 * network_capacity, 1)
    self.mapping = nn.Conv2d(3, network_capacity, 3, padding=1)
    if self.skip_conn_to_GAN and not self.internal_hist:
      self.hist_projection = HistVectorizer(hist, latent_dim,
                                            int(style_depth))
      self.to_latent_1 = nn.Linear(latent_dim, encoder_filters[-3])
      self.to_latent_2 = nn.Linear(latent_dim, encoder_filters[-2])
      self.conv_latent_1 = Conv2DMod(encoder_filters[-3],
                                     2 ** 2 * network_capacity, 3)
      self.conv_latent_2 = Conv2DMod(encoder_filters[-2],
                                     2 ** (2 - 1) * network_capacity, 3)
    elif self.skip_conn_to_GAN and self.internal_hist:
      self.to_latent_1 = nn.Linear(latent_dim, encoder_filters[-3])
      self.to_latent_2 = nn.Linear(latent_dim, encoder_filters[-2])
      self.conv_latent_1 = Conv2DMod(encoder_filters[-3],
                                     2 ** 2 * network_capacity, 3)
      self.conv_latent_2 = Conv2DMod(encoder_filters[-2],
                                     2 ** (2 - 1) * network_capacity, 3)

    for ind, (in_chan, out_chan) in enumerate(encoder_in_out_pairs):
      block = EncoderBlock(in_chan, out_chan)
      self.encoder_blocks.append(block)

    for ind, (in_chan, out_chan) in enumerate(decoder_in_out_pairs):
      block = DecoderBlock(
        in_chan, out_chan, internal_hist=self.internal_hist,
        latent_dim=latent_dim)
      self.decoder_blocks.append(block)

  def forward(self, x, hists=None):
    if self.skip_conn_to_GAN and not self.internal_hist:
      h_w_space = self.hist_projection(hists)
      h1 = self.to_latent_1(h_w_space)
      h2 = self.to_latent_2(h_w_space)
    elif self.skip_conn_to_GAN and self.internal_hist:
      h1 = self.to_latent_1(hists)
      h2 = self.to_latent_2(hists)

    x = self.mapping(x)
    x_list = []
    if self.skip_conn_to_GAN:
      x_list_up = []
    for block in self.encoder_blocks:
      x, xup = block(x)
      x_list.append(x)
      if self.skip_conn_to_GAN:
        x_list_up.append(xup)

    x_list.reverse()
    x_list_e = x_list[:-2]
    if self.skip_conn_to_GAN:
      processed_latent_1 = self.conv_latent_1(x_list_up[1], h1)
      processed_latent_2 = self.conv_latent_2(x_list_up[0], h2)
    rgb = None
    for prev_latent, block in zip(x_list_e, self.decoder_blocks):
      x, rgb = block(x, rgb, prev_latent, h=hists)
    x = self.decoder_mapping(x)
    if self.skip_conn_to_GAN:
      return x, rgb, processed_latent_1, processed_latent_2
    else:
      return x, rgb


class recoloringGAN(nn.Module):
  def __init__(self, image_size, latent_dim=512, style_depth=8,
               network_capacity=16, transparent=False, fp16=False,
               steps=1, lr=1e-4, fq_layers=[], fq_dict_size=256,
               attn_layers=[], hist=64, skip_conn_to_GAN=False,
               fixed_gan_weights=False, initialize_gan=False,
               internal_hist=False):
    super().__init__()

    self.lr = lr
    self.steps = steps
    self.fixed_gan_weights = fixed_gan_weights
    self.internal_hist = internal_hist
    self.skip_conn_to_GAN = skip_conn_to_GAN
    self.ED = RecoloringEncoderDecoder(image_size,
                                       network_capacity=network_capacity,
                                       hist=hist,
                                       latent_dim=latent_dim,
                                       style_depth=style_depth,
                                       skip_conn_to_GAN=skip_conn_to_GAN,
                                       internal_hist=self.internal_hist)
    self.H = HistVectorizer(hist, latent_dim, int(style_depth))
    self.G = RecoloringGAN(image_size, latent_dim, network_capacity,
                           transparent=transparent)
    self.D = Discriminator(image_size, network_capacity,
                           fq_layers=fq_layers,
                           fq_dict_size=fq_dict_size,
                           attn_layers=attn_layers, transparent=transparent)

    set_requires_grad(self.ED, True)
    set_requires_grad(self.H, True)
    set_requires_grad(self.G, True)
    set_requires_grad(self.D, True)

    if self.fixed_gan_weights == False:
      learnable_params = list(
        self.ED.parameters()) + list(
        self.G.parameters()) + list(self.H.parameters())
    else:
      learnable_params = self.ED.parameters()
    self.G_opt = DiffGrad(learnable_params, lr=self.lr, betas=(0.5, 0.9))
    self.D_opt = DiffGrad(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
    if initialize_gan:
      self._init_weights(initializeGAN=True)
    else:
      self._init_weights(initializeGAN=False)

    self._init_weights(self.ED)
    self._init_weights(self.D)

    self.cuda()

    if fp16:
      (self.ED, self.G, self.H, self.D), (self.G_opt, self.D_opt) = (
        amp.initialize(
          [self.ED, self.G, self.H, self.D], [self.G_opt, self.D_opt],
          opt_level='O2'))

  def _init_weights(self, initializeGAN=False):
    if initializeGAN:
      for block in self.G.blocks:
        nn.init.zeros_(block.to_noise1.weight)
        nn.init.zeros_(block.to_noise2.weight)
        nn.init.zeros_(block.to_noise1.bias)
        nn.init.zeros_(block.to_noise2.bias)
      for m in self.H.modules():
        if type(m) in {nn.Conv2d, nn.Linear}:
          nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in',
                                  nonlinearity='leaky_relu')

    for m in self.ED.modules():
      if type(m) in {nn.Conv2d, nn.Linear}:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in',
                                nonlinearity='leaky_relu')

    for m in self.D.modules():
      if type(m) in {nn.Conv2d, nn.Linear}:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in',
                                nonlinearity='leaky_relu')

  def forward(self, x):
    return x


class recoloringTrainer():
  def __init__(self, name, results_dir, models_dir, image_size,
               network_capacity, transparent=False, batch_size=4,
               mixed_prob=0.9, gradient_accumulate_every=1, lr=2e-4,
               num_workers=None, save_every=1000, trunc_psi=0.6,
               fp16=False, fq_layers=[], fq_dict_size=256, attn_layers=[],
               hist_method='inverse-quadratic',
               hist_resizing='sampling', hist_sigma=0.02, hist_bin=64,
               hist_insz=150, fixed_gan_weights=False, skip_conn_to_GAN=False,
               rec_loss='laplacian', initialize_gan=False,
               variance_loss=True, internal_hist=False,
               change_hyperparameters=False,
               change_hyperparameters_after=100000, *args, **kwargs):

    self.GAN_params = [args, kwargs]
    self.GAN = None
    self.hist_method = hist_method
    self.hist_resizing = hist_resizing
    self.hist_sigma = hist_sigma
    self.hist_bin = hist_bin
    self.change_hyperparameters_after = change_hyperparameters_after
    self.hist_insz = hist_insz
    self.rec_loss = rec_loss
    self.internal_hist = internal_hist
    self.change_hyperparameters = change_hyperparameters
    self.variance_loss = variance_loss
    self.fixed_gan_weights = fixed_gan_weights
    self.skip_conn_to_GAN = skip_conn_to_GAN
    self.initialize_gan = initialize_gan
    self.histBlock = RGBuvHistBlock(insz=self.hist_insz, h=self.hist_bin,
                                    method=self.hist_method,
                                    resizing=self.hist_resizing,
                                    sigma=self.hist_sigma)
    set_requires_grad(self.histBlock, True)

    if variance_loss is True:
      self.histBlock_input = RGBuvHistBlock(insz=self.hist_insz,
                                            h=self.hist_bin,
                                            method=self.hist_method,
                                            resizing=self.hist_resizing,
                                            sigma=self.hist_sigma)

      self.gaussKernel = get_gaussian_kernel(kernel_size=15,
                                             sigma=5, channels=3).to(
        device=torch.cuda.current_device())

    if self.rec_loss is None:
      self.rec_loss_func = reconstruction_loss('L1')
    elif self.rec_loss == 'sobel':
      self.rec_loss_func = reconstruction_loss('1st gradient')
    elif self.rec_loss == 'laplacian':
      self.rec_loss_func = reconstruction_loss('2nd gradient')
    else:
      raise Exception('Unknown reconstruction losst!')

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

    self.lr = lr
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.mixed_prob = mixed_prob

    self.save_every = save_every
    self.steps = 0

    self.av = None
    self.trunc_psi = trunc_psi

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
    if self.variance_loss is True:
      self.var_loss = 0

    self.init_folders()

    self.loader = None

    self.loader_evaluate = None

  def init_GAN(self):
    args, kwargs = self.GAN_params
    self.GAN = recoloringGAN(lr=self.lr, image_size=self.image_size,
                             network_capacity=self.network_capacity,
                             transparent=self.transparent,
                             fq_layers=self.fq_layers,
                             fq_dict_size=self.fq_dict_size,
                             attn_layers=self.attn_layers,
                             fp16=self.fp16, hist=self.hist_bin,
                             fixed_gan_weights=self.fixed_gan_weights,
                             skip_conn_to_GAN=self.skip_conn_to_GAN,
                             initialize_gan=self.initialize_gan,
                             internal_hist=self.internal_hist,
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
            'transparent': self.transparent,
            'fq_layers': self.fq_layers,
            'fq_dict_size': self.fq_dict_size,
            'attn_layers': self.attn_layers}

  def set_data_src(self, folder, sampling):
    self.dataset = Dataset(folder, image_size=self.image_size,
                           transparent=self.transparent,
                           hist_insz=self.hist_insz,
                           hist_bin=self.hist_bin,
                           hist_method=self.hist_method,
                           hist_resizing=self.hist_resizing,
                           hist_sampling=sampling)
    self.loader = cycle(data.DataLoader(
      self.dataset, num_workers=default(self.num_workers, num_cores),
      batch_size=self.batch_size, drop_last=True, shuffle=True,
      pin_memory=True))
    if sampling is True:
      self.dataset_evaluate = Dataset(folder, image_size=self.image_size,
                                      transparent=self.transparent,
                                      hist_insz=self.hist_insz,
                                      hist_bin=self.hist_bin,
                                      hist_method=self.hist_method,
                                      hist_resizing=self.hist_resizing,
                                      hist_sampling=sampling,
                                      triple_hist=True)
    else:
      self.dataset_evaluate = Dataset(folder, image_size=self.image_size,
                                      transparent=self.transparent,
                                      hist_insz=self.hist_insz,
                                      hist_bin=self.hist_bin,
                                      hist_method=self.hist_method,
                                      hist_resizing=self.hist_resizing,
                                      hist_sampling=sampling)
    self.loader_evaluate = cycle(
      data.DataLoader(self.dataset_evaluate, num_workers=default(
        self.num_workers, num_cores), batch_size=4, drop_last=True,
                      shuffle=True, pin_memory=True))

  def train(self, alpha=32, beta=1.5, gamma=4):
    assert self.loader is not None, ('You must first initialize the data '
                                     'source with `. set_data_src(<folder of '
                                     'images>)`')

    if (self.steps >= self.change_hyperparameters_after and
                      self.change_hyperparameters):
      self.alpha = 8
      self.gamma = 2
      self.beta = 1

    torch.autograd.set_detect_anomaly(False)

    if self.GAN is None:
      self.init_GAN()
    self.GAN.train()
    total_disc_loss = torch.tensor(0.0).cuda()
    total_gen_loss = torch.tensor(0.0).cuda()
    total_rec_loss = torch.tensor(0.0).cuda()
    total_hist_loss = torch.tensor(0.0).cuda()

    if self.variance_loss is True:
      total_var_loss = torch.tensor(0.0).cuda()

    batch_size = self.batch_size

    image_size = self.GAN.G.image_size

    apply_gradient_penalty = self.steps % 4 == 0

    backwards = partial(loss_backwards, self.fp16)

    # train discriminator

    self.GAN.D_opt.zero_grad()

    for i in range(self.gradient_accumulate_every):
      batch = next(self.loader)
      image_batch = batch['images'].cuda()
      hist_batch = batch['histograms'].cuda()
      image_batch.requires_grad_()
      noise = image_noise(batch_size, image_size)

      h_w_space = self.GAN.H(hist_batch)
      if self.skip_conn_to_GAN and not self.internal_hist:
        image_latent, rgb, processed_latent_2, processed_latent_1 = (
          self.GAN.ED(image_batch, hist_batch))
        generated_images = self.GAN.G(
          image_latent, rgb, h_w_space, noise, processed_latent_2,
          processed_latent_1)
      elif self.skip_conn_to_GAN and self.internal_hist:
        image_latent, rgb, processed_latent_2, processed_latent_1 = (
          self.GAN.ED(image_batch, h_w_space))
        generated_images = self.GAN.G(image_latent, rgb, h_w_space,
                                      noise, processed_latent_2,
                                      processed_latent_1)
      elif self.internal_hist:
        image_latent, rgb = self.GAN.ED(image_batch, h_w_space)
        generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise)
      else:
        image_latent, rgb = self.GAN.ED(image_batch, hist_batch)
        generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise)
      fake_output, fake_q_loss = self.GAN.D(generated_images.clone().detach())
      real_output, real_q_loss = self.GAN.D(image_batch)
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
      batch = next(self.loader)
      image_batch = batch['images'].cuda()
      hist_batch = batch['histograms'].cuda()
      noise = image_noise(batch_size, image_size)
      h_w_space = self.GAN.H(hist_batch)
      if self.skip_conn_to_GAN and not self.internal_hist:
        image_latent, rgb, processed_latent_2, processed_latent_1 = (
          self.GAN.ED(image_batch, hist_batch))
        generated_images = self.GAN.G(image_latent, rgb, h_w_space,
                                      noise, processed_latent_2,
                                      processed_latent_1)
      elif self.skip_conn_to_GAN and self.internal_hist:
        image_latent, rgb, processed_latent_2, processed_latent_1 = (
          self.GAN.ED(image_batch, h_w_space))
        generated_images = self.GAN.G(image_latent, rgb, h_w_space,
                                      noise, processed_latent_2,
                                      processed_latent_1)
      elif self.internal_hist:
        image_latent, rgb = self.GAN.ED(image_batch, h_w_space)
        generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise)

      else:
        image_latent, rgb = self.GAN.ED(image_batch, hist_batch)
        generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise)

      fake_output, _ = self.GAN.D(generated_images)

      d_loss = gamma * fake_output.mean()

      generated_histograms = self.histBlock(F.relu(generated_images))

      histogram_loss = alpha * SCALE * (torch.sqrt(
        torch.sum(torch.pow(torch.sqrt(
          hist_batch) - torch.sqrt(
          generated_histograms), 2)))) / hist_batch.shape[0]

      reconstruction_loss = beta * self.rec_loss_func.compute_loss(
        image_batch, generated_images)

      if self.variance_loss is True:
        input_histograms = self.histBlock_input(F.relu(hist_batch))
        input_gauss = gaussian_op(image_batch, kernel=self.gaussKernel)
        generated_gauss = gaussian_op(
          generated_images, kernel=self.gaussKernel)
        var_loss = -1 * (beta / 10) * torch.sum(torch.abs(
          hist_batch - input_histograms)) * torch.mean(
          torch.abs(torch.std(torch.std(input_gauss, dim=2), dim=2) -
                    torch.std(torch.std(generated_gauss, dim=2), dim=2)))
        gen_loss = d_loss + histogram_loss + reconstruction_loss + var_loss
      else:
        gen_loss = d_loss + histogram_loss + reconstruction_loss

      gen_loss = gen_loss / self.gradient_accumulate_every
      gen_loss.register_hook(raise_if_nan)
      backwards(gen_loss, self.GAN.G_opt)

      total_rec_loss += (reconstruction_loss.detach().item() /
                        self.gradient_accumulate_every)
      total_gen_loss += (d_loss.detach().item() /
                         self.gradient_accumulate_every)
      total_hist_loss += (histogram_loss.detach().item() /
                          self.gradient_accumulate_every)
      if self.variance_loss is True:
        total_var_loss += (var_loss.detach().item() /
                           self.gradient_accumulate_every)

    self.g_loss = float(total_gen_loss)
    self.r_loss = float(total_rec_loss)
    self.h_loss = float(total_hist_loss)
    if self.variance_loss is True:
      self.var_loss = float(total_var_loss)

    self.GAN.G_opt.step()

    # save from NaN errors
    checkpoint_num = floor(self.steps / self.save_every)

    if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
      print(f'NaN detected for generator or discriminator. Loading from '
            f'checkpoint #{checkpoint_num}')
      self.load(checkpoint_num)
      raise NanException

    # periodically save results
    if self.steps % self.save_every == 0:
      self.save(checkpoint_num)

    if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
      if not self.fixed_gan_weights:
        self.evaluate(floor(self.steps / 1000), triple_hist=True)
      else:
        self.evaluate(floor(self.steps / 1000))
    self.steps += 1
    self.av = None

  @torch.no_grad()
  def evaluate(self, num=0, image_batch=None, hist_batch=None,
               triple_hist=False, double_hist=False, resizing=None,
               resizing_method=None, swapping_levels=1,
               pyramid_levels=5, level_blending=False, original_size=None,
               input_image_name=None, original_image=None,
               post_recoloring=False, save_input=True):
    self.GAN.eval()

    if hist_batch is None or image_batch is None:
      batch = next(self.loader_evaluate)
      image_batch = batch['images'].cuda()
      hist_batch = batch['histograms'].cuda()
      img_bt_sz = image_batch.shape[0]
      if triple_hist is True:
        image_batch = torch.cat((image_batch, image_batch,
                                 image_batch), dim=0)
        hist_batch_2 = batch['histograms2'].cuda()
        hist_batch_3 = batch['histograms3'].cuda()
        hist_batch = torch.cat((hist_batch, hist_batch_2,
                                hist_batch_3), dim=0)
      elif double_hist is True:
        image_batch = torch.cat((image_batch, image_batch), dim=0)
        hist_batch_2 = batch['histograms2'].cuda()
        hist_batch = torch.cat((hist_batch, hist_batch_2), dim=0)
    else:
      img_bt_sz = image_batch.shape[0]

    noise = image_noise(hist_batch.shape[0], image_batch.shape[-1])
    h_w_space = self.GAN.H(hist_batch)

    if self.skip_conn_to_GAN and not self.internal_hist:
      image_latent, rgb, processed_latent_2, processed_latent_1 = (
        self.GAN.ED(image_batch, hist_batch))
      generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise,
                                    processed_latent_2,
                                    processed_latent_1)
    elif self.skip_conn_to_GAN and self.internal_hist:
      image_latent, rgb, processed_latent_2, processed_latent_1 = (
        self.GAN.ED(image_batch, h_w_space))
      generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise,
                                    processed_latent_2,
                                    processed_latent_1)
    elif self.internal_hist:
      image_latent, rgb = self.GAN.ED(image_batch, h_w_space)
      generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise)
    else:
      image_latent, rgb = self.GAN.ED(image_batch, hist_batch)
      generated_images = self.GAN.G(image_latent, rgb, h_w_space, noise)

    ext = 'jpg' if not self.transparent else 'png'
    if double_hist is True or triple_hist is True:
      num_rows = img_bt_sz
    else:
      num_rows = int(np.ceil(sqrt(hist_batch.shape[0])))
    output_name = str(self.results_dir / self.name /
                      f'{str(num)}-generated.{ext}')
    torchvision.utils.save_image(
      generated_images, output_name, nrow=num_rows)

    if resizing is not None:
      if resizing == 'upscaling':
        print('Upsampling')
        if resizing_method == 'BGU':
          os.system('BGU.exe '
                    f'"{input_image_name}" '
                    f'"{output_name}" "{output_name}"')
        elif resizing_method == 'pyramid':
          reference = Image.open(input_image_name)
          transform = transforms.Compose([transforms.ToTensor()])
          reference = torch.unsqueeze(transform(reference), dim=0).to(
            device=torch.cuda.current_device())
          output = upsampling.pyramid_upsampling(
            generated_images, reference, levels=pyramid_levels,
            swapping_levels=swapping_levels, blending=level_blending)
          torchvision.utils.save_image(output, output_name, nrow=num_rows)

      elif resizing == 'downscaling':
        if original_size is not None:
          print('Resizing')
          img = Image.open(output_name)
          img = img.resize((original_size[0], original_size[1]))
          img.save(output_name)

    if post_recoloring is True:
      target = torch.squeeze(generated_images, dim=0).cpu().detach().numpy()
      target = target.transpose(1, 2, 0)
      print('Post-recoloring')
      result = ct.color_transfer_MKL(original_image, target)
      result = Image.fromarray(np.uint8(result * 255))
      result.save(output_name)

    if save_input is True:
      if double_hist is True or triple_hist is True:
        torchvision.utils.save_image(
          image_batch[:img_bt_sz, :, :, :],
          str(self.results_dir / self.name / f'{str(num)}-input.'
                                             f'{ext}'),
          nrow=img_bt_sz)
      else:
        torchvision.utils.save_image(image_batch, str(
          self.results_dir / self.name / f'{str(num)}-input.{ext}'),
                                     nrow=num_rows)

    return generated_images

  def print_log(self):
    if hasattr(self, 'var_loss'):
      print(f'\nG: {self.g_loss:.2f} | H: {self.h_loss:.2f} | '
            f'D: {self.d_loss:.2f} | R: {self.r_loss:.2f} '
            f'| V: {self.var_loss:.2f} | GP: {self.last_gp_loss:.2f}'
            f' | CR: {self.last_cr_loss:.2f} | Q: {self.q_loss:.2f}')
    else:
      print(f'\nG: {self.g_loss:.2f} | H: {self.h_loss:.2f} | '
            f'D: {self.d_loss:.2f} | R: {self.r_loss:.2f} |'
            f' GP: {self.last_gp_loss:.2f} '
            f'| CR: {self.last_cr_loss:.2f} | Q: {self.q_loss:.2f}')

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
      file_paths = [p for p in Path(self.models_dir /
                                    self.name).glob('model_*.pt')]
      saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
      if len(saved_nums) == 0:
        return -1
      name = saved_nums[-1]
      print(f'continuing from previous epoch - {name}')
    self.steps = name * self.save_every
    self.GAN.load_state_dict(
      torch.load(self.model_name(name),
                 map_location=f'cuda:{torch.cuda.current_device()}'))
    return 0
