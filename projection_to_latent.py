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

from histoGAN import Trainer
import torch
import argparse
import torchvision
import os
from datetime import datetime
from PIL import Image
from torchvision import transforms
from torch import optim
from utils.vggloss import VGGPerceptualLoss
import pickle
import numpy as np
import utils.pyramid_upsampling as upsampling
from utils import color_transfer_MKL as ct
from histogram_classes.RGBuvHistBlock import RGBuvHistBlock
mse = torch.nn.MSELoss()


# helpers
def noise(n, latent_dim):
  return torch.randn(n, latent_dim).cuda()


def noise_list(n, layers, latent_dim):
  return [(noise(n, latent_dim), layers)]


def latent_to_w(style_vectorizer, latent_descr):
  return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size):
  return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()


def set_requires_grad(model, bool):
  for p in model.parameters():
    p.requires_grad = bool


def styles_def_to_tensor(styles_def):
  return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def],
                   dim=1)


def noise(n, latent_dim):
  return torch.randn(n, latent_dim).cuda()


def process_image(model, histogram, style1_list, style2_list, torgb_style_list,
                  noise1_list=None, noise2_list=None, in_noise=None):
  rgb = None
  x = model.GAN.GE.initial_block.expand(1, -1, -1, -1)
  if noise1_list is not None and noise2_list is not None:
    for i, (s1, s2, rgb_s, n1, n2, block) in enumerate(zip(
        style1_list, style2_list, torgb_style_list,
        noise1_list, noise2_list, model.GAN.GE.blocks)):
      if i < (len(model.GAN.GE.blocks) - 2):
        x, rgb = block.forward_(x, rgb, s1, s2, rgb_s, noise1=n1, noise2=n2)
      else:
        s1 = block.to_style1(histogram)
        s2 = block.to_style2(histogram)
        rgb_s = block.to_rgb.to_style(histogram)
        x, rgb = block.forward_(x, rgb, s1, s2, rgb_s, noise1=n1, noise2=n2)
  else:
    for i, (s1, s2, rgb_s, block) in enumerate(zip(
        style1_list, style2_list, torgb_style_list, model.GAN.GE.blocks)):
      if i < (len(model.GAN.GE.blocks) - 2):
        x, rgb = block.forward_(x, rgb, s1, s2, rgb_s, inoise=in_noise)
      else:
        s1 = block.to_style1(histogram)
        s2 = block.to_style2(histogram)
        rgb_s = block.to_rgb.to_style(histogram)
        x, rgb = block.forward_(x, rgb, s1, s2, rgb_s, inoise=in_noise)

  return rgb


def recolor_image(model, model_name, target_hist_name, input_image_name,
                  target_hist, latent_noise, optimize_noise, add_noise=False,
                  random_styles=[], results_dir='results_projection_to_latent',
                  post_recoloring=False, upsampling_output=False,
                  upsampling_method='pyramid', swapping_levels=1,
                  pyramid_levels=5, level_blending=False):

  if random_styles:
    assert max(random_styles) <= (model.GAN.GE.num_layers - 2)
    random_styles = list(set(random_styles))
    new_style1_list = []
    new_style2_list = []
    new_torgb_style_list = []
    get_latents_fn = noise_list
    latent_dim = model.GAN.G.latent_dim
    style = get_latents_fn(1, len(random_styles), latent_dim)
    w_space = latent_to_w(model.GAN.SE, style)
    w_styles = styles_def_to_tensor(w_space)

    styles = w_styles.transpose(0, 1)

    for j, i in enumerate(random_styles):
      style = styles[j, :, :]
      style1 = model.GAN.GE.blocks[i-1].to_style1(style)
      style2 = model.GAN.GE.blocks[i-1].to_style2(style)
      torgb_style = model.GAN.GE.blocks[i-1].to_rgb.to_style(style)
      new_style1_list.append(style1.detach().requires_grad_())
      new_style2_list.append(style2.detach().requires_grad_())
      new_torgb_style_list.append(torgb_style.detach().requires_grad_())


  now = datetime.now()
  timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
  postfix = round(np.random.rand() * 1000)
  filename = os.path.basename(os.path.splitext(input_image_name)[0])

  with open(
      f'{results_dir}/{model_name}/{filename}/{filename}_final.pickle',
      'rb') as handle:
    data = pickle.load(handle)
    style1_list = data['style1_list']
    style2_list = data['style2_list']
    torgb_style_list = data['torgb_style_list']
    if random_styles:
      for j, i in enumerate(random_styles):
        style1_list[i - 1] = new_style1_list[j]
        style2_list[i - 1] = new_style2_list[j]
        torgb_style_list[i - 1] = new_torgb_style_list[j]

    if optimize_noise:
      if latent_noise:
        noise1_list = data['noise1_list']
        noise2_list = data['noise2_list']
        in_noise = None
      else:
        noise1_list = None
        noise2_list = None
        in_noise = data['in_noise']
        if add_noise:
          image_size = model.GAN.G.image_size
          shift = image_noise(1, image_size).to(device=in_noise.get_device())
          in_noise = (in_noise + shift) / 2
    else:
      noise1_list = None
      noise2_list = None
      image_size = model.GAN.G.image_size
      in_noise = image_noise(1, image_size)

  h_w_space = model.GAN.HE(target_hist)

  rgb = process_image(model, h_w_space, style1_list, style2_list,
                      torgb_style_list, noise1_list=noise1_list,
                      noise2_list=noise2_list, in_noise=in_noise)

  samples_name = ('generated-' + filename +
                  f'{os.path.basename(os.path.splitext(target_hist_name)[0])}'
                  f'-{timestamp}-{postfix}')

  out_name = f'{results_dir}/{model_name}/{filename}/{samples_name}.jpg'
  torchvision.utils.save_image(rgb, out_name, nrow=1)

  if post_recoloring:
    target = torch.squeeze(rgb, dim=0).cpu().detach().numpy()
    target = target.transpose(1, 2, 0)
    print('Post-recoloring')
    source = np.array(Image.open(input_image_name)) / 255
    result = ct.color_transfer_MKL(source, target)
    result = Image.fromarray(np.uint8(result * 255))
    result.save(out_name)

  if upsampling_output:
    print('Upsampling ...')
    if upsampling_method == 'BGU':
      os.system('BGU.exe '
                f'"{input_image_name}" '
                f'"{out_name}" "{out_name}"')
    elif upsampling_method == 'pyramid':
      reference = Image.open(input_image_name)
      transform = transforms.Compose([transforms.ToTensor()])
      reference = torch.unsqueeze(transform(reference), dim=0).to(
        device=torch.cuda.current_device())
      rgb = upsampling.pyramid_upsampling(rgb, reference,
                                          swapping_levels=swapping_levels,
                                          levels=pyramid_levels,
                                          blending=level_blending)
      torchvision.utils.save_image(rgb, out_name, nrow=1)
    else:
      raise Exception('Unknown upsampling method')

  print(f'sample images generated at {out_name}')


def project_to_latent(
    results_dir='./results_projection_to_latent',
    models_dir='./models',
    name='test',
    load_from=-1,
    image_size=128,
    target_hist=None,
    latent_noise=False,
    optimize_noise=True,
    add_noise=False,
    random_styles=[],
    generate=False,
    post_recoloring=False,
    upsampling_output=False,
    upsampling_method='pyramid',
    swapping_levels=1,
    pyramid_levels=5,
    level_blending=False,
    network_capacity=16,
    transparent=False,
    num_train_steps=10000,
    learning_rate=2e-4,
    pixel_loss='L1',
    vgg_loss_weight=0.005,
    pixel_loss_weight=1.0,
    style_reg_weight=0.0,
    noise_reg_weight=0.0,
    save_every=500,
    trunc_psi=0.75,
    fp16=False,
    fq_layers=[],
    fq_dict_size=256,
    attn_layers=[],
    hist_method='inverse-quadratic',
    hist_resizing='sampling',
    hist_sigma=0.02,
    hist_bin=64,
    hist_insz=150,
    input_image=None,
    aug_prob=0.0):
  model = Trainer(
    name,
    results_dir,
    models_dir,
    image_size=image_size,
    network_capacity=network_capacity,
    transparent=transparent,
    lr=learning_rate,
    save_every=save_every,
    trunc_psi=trunc_psi,
    fp16=fp16,
    fq_layers=fq_layers,
    fq_dict_size=fq_dict_size,
    attn_layers=attn_layers,
    hist_insz=hist_insz,
    hist_bin=hist_bin,
    hist_sigma=hist_sigma,
    hist_resizing=hist_resizing,
    hist_method=hist_method,
    aug_prob=aug_prob,
  )

  model.load(load_from)

  model.GAN.eval()

  histblock = RGBuvHistBlock(insz=hist_insz, h=hist_bin,
                             resizing=hist_resizing, method=hist_method,
                             sigma=hist_sigma,
                             device=torch.cuda.current_device())
  transform = transforms.Compose([transforms.ToTensor()])

  if generate:

    if target_hist is None:
      raise Exception('No target histogram is given')

    extension = os.path.splitext(target_hist)[1]
    if extension == '.npy':
      hist = np.load(target_hist)
      h = torch.from_numpy(hist).to(device=torch.cuda.current_device())

      recolor_image(model=model, model_name=name, target_hist_name=target_hist,
                    input_image_name=input_image,
                    results_dir=results_dir,
                    target_hist=h, latent_noise=latent_noise,
                    optimize_noise=optimize_noise,
                    add_noise=add_noise, random_styles=random_styles,
                    post_recoloring=post_recoloring,
                    upsampling_output=upsampling_output,
                    upsampling_method=upsampling_method,
                    swapping_levels=swapping_levels,
                    pyramid_levels=pyramid_levels,
                    level_blending=level_blending)

    elif str.lower(extension) == '.jpg' or str.lower(extension) == '.png':
      histblock = RGBuvHistBlock(insz=hist_insz, h=hist_bin,
                                 resizing=hist_resizing, method=hist_method,
                                 sigma=hist_sigma,
                                 device=torch.cuda.current_device())
      transform = transforms.Compose([transforms.ToTensor()])
      img = Image.open(target_hist)
      img = torch.unsqueeze(transform(img), dim=0).to(
        device=torch.cuda.current_device())
      h = histblock(img)
      recolor_image(model=model, model_name=name, target_hist_name=target_hist,
                    input_image_name=input_image,
                    results_dir=results_dir,
                    target_hist=h, latent_noise=latent_noise,
                    optimize_noise=optimize_noise,
                    add_noise=add_noise, random_styles=random_styles,
                    post_recoloring=post_recoloring,
                    upsampling_output=upsampling_output,
                    upsampling_method=upsampling_method,
                    swapping_levels=swapping_levels,
                    pyramid_levels=pyramid_levels,
                    level_blending=level_blending)

    elif extension == '':
      files = [os.path.join(target_hist, f) for f in os.listdir(target_hist) if
               os.path.isfile(os.path.join(target_hist, f))]
      histblock = RGBuvHistBlock(insz=hist_insz, h=hist_bin,
                                 resizing=hist_resizing, method=hist_method,
                                 sigma=hist_sigma,
                                 device=torch.cuda.current_device())
      transform = transforms.Compose([transforms.ToTensor()])
      for f in files:
        extension = os.path.splitext(f)[1]
        if extension == '.npy':
          hist = np.load(f)
          h = torch.from_numpy(hist).to(device=torch.cuda.current_device())
          recolor_image(model=model, model_name=name,
                        target_hist_name=target_hist,
                        results_dir=results_dir,
                        input_image_name=input_image,
                        optimize_noise=optimize_noise,
                        add_noise=add_noise, random_styles=random_styles,
                        target_hist=h, latent_noise=latent_noise,
                        post_recoloring=post_recoloring,
                        upsampling_output=upsampling_output,
                        upsampling_method=upsampling_method,
                        swapping_levels=swapping_levels,
                        pyramid_levels=pyramid_levels,
                        level_blending=level_blending)

        elif (extension == str.lower(extension) == '.jpg' or str.lower(
            extension) == '.png'):
          img = Image.open(f)
          img = torch.unsqueeze(transform(img), dim=0).to(
            device=torch.cuda.current_device())
          h = histblock(img)
          recolor_image(model=model, model_name=name,
                        target_hist_name=target_hist,
                        results_dir=results_dir,
                        input_image_name=input_image,
                        optimize_noise=optimize_noise,
                        add_noise=add_noise, random_styles=random_styles,
                        target_hist=h, latent_noise=latent_noise,
                        post_recoloring=post_recoloring,
                        upsampling_output=upsampling_output,
                        upsampling_method=upsampling_method,
                        swapping_levels=swapping_levels,
                        pyramid_levels=pyramid_levels,
                        level_blending=level_blending)

        else:
          print(f'Warning: File extension of {f} is not supported.')
          continue

    else:
      print('The file extension of target image is not supported.')
      raise NotImplementedError

    return

  ## optimization part

  if vgg_loss_weight > 0:
    vgg_loss = VGGPerceptualLoss(device=torch.cuda.current_device())

  set_requires_grad(model.GAN.SE, False)
  set_requires_grad(model.GAN.HE, False)
  set_requires_grad(model.GAN.GE, False)

  extension = os.path.splitext(input_image)[1]

  if not (str.lower(extension) == '.jpg' or str.lower(extension) == '.png'):
    raise Exception('No target histogram or image is given')

  filename = os.path.basename(os.path.splitext(input_image)[0])

  if not os.path.exists(results_dir):
    os.mkdir(results_dir)
  if not os.path.exists(f'{results_dir}/{name}/{filename}'):
    os.mkdir(f'{results_dir}/{name}/{filename}')

  latent_dim = model.GAN.G.latent_dim
  image_size = model.GAN.G.image_size
  num_layers = model.GAN.G.num_layers

  input_image = Image.open(input_image)
  input_image = input_image.resize((image_size, image_size))
  input_image = torch.unsqueeze(transform(input_image), dim=0).to(
    device=torch.cuda.current_device())
  input_image.requires_grad = False
  in_h = torch.unsqueeze(histblock(input_image), dim=0)
  in_h.requires_grad = False

  get_latents_fn = noise_list
  style = get_latents_fn(1, num_layers - 2, latent_dim)
  in_noise = image_noise(1, image_size)
  w_space = latent_to_w(model.GAN.SE, style)
  histogram = model.GAN.HE(in_h)
  h_w_space = torch.unsqueeze(histogram, dim=1)
  h_w_space = torch.cat((h_w_space, h_w_space), dim=1)
  w_styles = styles_def_to_tensor(w_space)

  x = model.GAN.GE.initial_block.expand(1, -1, -1, -1)
  styles = w_styles.transpose(0, 1)
  hists = h_w_space.transpose(0, 1)
  styles = torch.cat((styles, hists), dim=0)

  rgb = None
  if optimize_noise and latent_noise:
    noise1_list = []
    noise2_list = []
  style1_list = []
  style2_list = []
  torgb_style_list = []
  for i, (style, block) in enumerate(zip(styles, model.GAN.GE.blocks)):
    noise = in_noise
    if block.upsample is not None:
      noise = noise[:, :x.shape[2] * 2, :x.shape[3] * 2, :]
    else:
      noise = noise[:, :x.shape[2], :x.shape[3], :]
    if optimize_noise and latent_noise:
      noise1 = block.to_noise1(noise).permute((0, 3, 2, 1))
      noise2 = block.to_noise2(noise).permute((0, 3, 2, 1))
      noise1_list.append(noise1.detach().requires_grad_())
      noise2_list.append(noise2.detach().requires_grad_())

    if i < (len(model.GAN.GE.blocks) - 2):
      style1 = block.to_style1(style)
      style2 = block.to_style2(style)
      torgb_style = block.to_rgb.to_style(style)
      style1_list.append(style1.detach().requires_grad_())
      style2_list.append(style2.detach().requires_grad_())
      torgb_style_list.append(torgb_style.detach().requires_grad_())
    else:
      style1 = block.to_style1(histogram)
      style2 = block.to_style2(histogram)
      torgb_style = block.to_rgb.to_style(histogram)
      style1_list.append(torch.tensor([]).requires_grad_())
      style2_list.append(torch.tensor([]).requires_grad_())
      torgb_style_list.append(torch.tensor([]).requires_grad_())
    if latent_noise:
      x, rgb = block.forward_(x, rgb, style1, style2, torgb_style,
                              noise1=noise1,
                              noise2=noise2)
    else:
      x, rgb = block.forward_(x, rgb, style1, style2, torgb_style, inoise=noise)

  torchvision.utils.save_image(
    rgb, f'{results_dir}/{name}/{filename}/{filename}_start.jpg', nrow=1)

  if optimize_noise:
    if latent_noise:
      optimizer = optim.Adam(style1_list + style2_list + torgb_style_list +
                             noise1_list + noise2_list, lr=learning_rate)
    else:
      in_noise = in_noise.detach().requires_grad_()
      optimizer = optim.Adam(style1_list + style2_list + torgb_style_list +
                             [in_noise], lr=learning_rate)
  else:
    optimizer = optim.Adam(style1_list + style2_list + torgb_style_list,
                           lr=learning_rate)

  if optimize_noise and latent_noise:
    in_noise = None
  else:
    noise1_list = noise2_list = None

  for training_step in range(num_train_steps):

    rgb = process_image(model, histogram, style1_list, style2_list,
                        torgb_style_list, noise1_list=noise1_list,
                        noise2_list=noise2_list, in_noise=in_noise)

    if pixel_loss == 'L1':
      rec_loss = pixel_loss_weight * torch.mean(torch.abs(input_image - rgb))
    elif pixel_loss == 'L2':
      rec_loss = pixel_loss_weight * mse(input_image, rgb)
    if vgg_loss_weight:
      latent_loss = vgg_loss_weight * vgg_loss(input_image, rgb)
      loss = rec_loss + latent_loss
    else:
      loss = rec_loss
    if optimize_noise:
      if latent_noise:
        for i, (noise1, noise2) in enumerate(zip(noise1_list, noise2_list)):
          if i == 0:
            noise_loss = noise_reg_weight * (noise1.mean() ** 2 +
                                              noise2.mean() ** 2)
          else:
            noise_loss += noise_reg_weight * (noise1.mean() ** 2 +
                                              noise2.mean() ** 2)
        noise_loss = noise_loss / (len(noise1_list))
      else:
        noise_loss = noise_reg_weight * in_noise.mean() ** 2
      loss = loss + noise_loss

    else:
      noise_loss = torch.tensor(0)

    for i, (style1, style2) in enumerate(zip(style1_list, style2_list)):
      if i == 0:
        style_loss = style_reg_weight * (style1.mean() ** 2 + style2.mean()
                                         ** 2)
      elif i < (len(style1_list) - 2):
        style_loss += style_reg_weight * (
              style1.mean() ** 2 + style2.mean() ** 2)
      style_loss = style_loss / (len(style1_list) - 2)

    loss = loss + style_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if vgg_loss_weight:
      print(f'Optimization step {training_step + 1}, '
            f'rec. loss = {rec_loss.item()}, vgg loss = {latent_loss.item()}, '
            f'rec. noise reg loss = {noise_loss.item()}, style reg loss ='
            f' {style_loss.item()}')

    else:
      print(f'Optimization step {training_step + 1}, rec. loss = '
            f'{rec_loss.item()}, rec. noise reg loss = {noise_loss.item()}, '
            f'style reg loss = {style_loss.item()}')

    if (training_step + 1) % save_every == 0:

      if latent_noise:
        projected_image = process_image(model, histogram, style1_list,
                                        style2_list, torgb_style_list,
                                        noise1_list=noise1_list,
                                        noise2_list=noise2_list)
      else:
        projected_image = process_image(model, histogram, style1_list,
                                        style2_list, torgb_style_list,
                                        in_noise=in_noise)

      torchvision.utils.save_image(
        projected_image, f'{results_dir}/{name}/{filename}/{filename}_'
                         f'{training_step + 1}.jpg', nrow=1)

      if latent_noise:
        data = {'style1_list': style1_list, 'style2_list': style2_list,
                'torgb_style_list': torgb_style_list, 'noise1_list':
                  noise1_list, 'noise2_list': noise2_list}
      else:
        data = {'style1_list': style1_list, 'style2_list': style2_list,
                'torgb_style_list': torgb_style_list, 'in_noise': in_noise}

      with open(f'{results_dir}/{name}/{filename}/{filename}_'
                f'{training_step + 1}.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  if latent_noise:
    data = {'style1_list': style1_list, 'style2_list': style2_list,
            'torgb_style_list': torgb_style_list, 'noise1_list':
              noise1_list, 'noise2_list': noise2_list}
  else:
    data = {'style1_list': style1_list, 'style2_list': style2_list,
            'torgb_style_list': torgb_style_list, 'in_noise': in_noise}

  with open(f'{results_dir}/{name}/{filename}/{filename}_final.pickle',
            'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  print('End of optimization.')

  with open(f'{results_dir}/{name}/{filename}/{filename}_final.pickle',
            'rb') as handle:
    data = pickle.load(handle)
    style1_list = data['style1_list']
    style2_list = data['style2_list']
    torgb_style_list = data['torgb_style_list']
    if optimize_noise and latent_noise:
      noise1_list = data['noise1_list']
      noise2_list = data['noise2_list']

      projected_image = process_image(model, histogram, style1_list,
                                      style2_list, torgb_style_list,
                                      noise1_list=noise1_list,
                                      noise2_list=noise2_list)
    elif optimize_noise:
      in_noise = data['in_noise']
      projected_image = process_image(model, histogram, style1_list,
                                      style2_list, torgb_style_list,
                                      in_noise=in_noise)
    else:
      projected_image = process_image(model, histogram, style1_list,
                                      style2_list, torgb_style_list,
                                      in_noise=in_noise)

  torchvision.utils.save_image(
    projected_image, f'{results_dir}/{name}/{filename}/'
                     f'{filename}_final.jpg', nrow=1)


def get_args():
  parser = argparse.ArgumentParser(description='Project into HistoGAN latent.')
  parser.add_argument('--results_dir', dest='results_dir',
                      default='./results_projection_to_latent')
  parser.add_argument('--models_dir', dest='models_dir', default='./models')
  parser.add_argument('--input_image', dest='input_image', default=None)
  parser.add_argument('--target_hist', dest='target_hist', default=None)
  parser.add_argument('--generate', dest='generate',
                      default=False, type=bool)
  parser.add_argument('--name', dest='name', default='histoGAN_model')
  parser.add_argument('--image_size', dest='image_size', default=256, type=int)
  parser.add_argument('--network_capacity', dest='network_capacity', default=16,
                      type=int)
  parser.add_argument('--load_from', dest='load_from', default=-1)
  parser.add_argument('--transparent', dest='transparent', default=False)
  parser.add_argument('--num_train_steps', dest='num_train_steps',
                      default=2000, type=int)
  parser.add_argument('--learning_rate', dest='learning_rate', default=1e-1,
                      type=float)
  parser.add_argument('--latent_noise', dest='latent_noise', default=False,
                      type=bool)
  parser.add_argument('--optimize_noise', dest='optimize_noise', default=False,
                      type=bool, help='Use it for noise optimization. In '
                                      'testing, use it to load saved noise '
                                      'even when training without noise '
                                      'optimization.')
  parser.add_argument('--add_noise', dest='add_noise', default=False)
  parser.add_argument('--random_styles', dest='random_styles', nargs='+',
                      default=[], type=int)
  parser.add_argument('--save_every', dest='save_every', default=100, type=int)
  parser.add_argument('--fp 16', dest='fp16', default=False)
  parser.add_argument('--fq_layers', dest='fq_layers', default=[])
  parser.add_argument('--fq_dict_size', dest='fq_dict_size', default=256,
                      type=int)
  parser.add_argument('--pixel_loss', dest='pixel_loss', default='L1',
                      help='L1 or L2')
  parser.add_argument('--vgg_loss_weight', dest='vgg_loss_weight',
                      default=0.001, type=float)
  parser.add_argument('--pixel_loss_weight', dest='pixel_loss_weight',
                      default=1.0, type=float)
  parser.add_argument('--noise_reg_weight', dest='noise_reg_weight',
                      default=0.0, type=float)
  parser.add_argument('--style_reg_weight', dest='style_reg_weight',
                      default=0.0, type=float)
  parser.add_argument('--trunc_psi', dest='trunc_psi', default=0.75,
                      type=float)
  parser.add_argument('--attn_layers', dest='attn_layers', default=[])
  parser.add_argument('--gpu', dest='gpu', default=0, type=int)
  parser.add_argument('--hist_bin', dest='hist_bin', default=64, type=int)
  parser.add_argument('--hist_insz', dest='hist_insz', default=150, type=int)
  parser.add_argument('--hist_method', dest='hist_method',
                      default='inverse-quadratic')
  parser.add_argument('--hist_resizing', dest='hist_resizing',
                      default='interpolation')
  parser.add_argument('--hist_sigma', dest='hist_sigma', default=0.02,
                      type=float)
  parser.add_argument('--upsampling_output', dest='upsampling_output',
                      default=False, type=bool,
                      help='TESTING PHASE: Applies a guided upsampling '
                           'post-processing step.')
  parser.add_argument('--upsampling_method', dest='upsampling_method',
                      default='pyramid', type=str,
                      help='TESTING PHASE: BGU or pyramid.')
  parser.add_argument('--pyramid_levels', dest='pyramid_levels',
                      default=6, type=int,
                      help='TESTING PHASE: when --upsampling_output True and '
                           '--upsampling_method is pyramid, this controls the '
                           'number of levels in the Laplacian pymraid.')
  parser.add_argument('--swapping_levels', dest='swapping_levels',
                      default=1, type=int,
                      help='TESTING PHASE: when --upsampling_output True and '
                           '--upsampling_method is pyramid, this controls the '
                           'number of levels to swap.')
  parser.add_argument('--level_blending', dest='level_blending',
                      default=False, type=bool,
                      help='TESTING PHASE: when --upsampling_output True and '
                           '--upsampling_method is pyramid, this allows to '
                           'blend between pyramid levels.')
  parser.add_argument('--post_recoloring',
                      dest='post_recoloring',
                      default=False, type=bool,
                      help='TESTING PHASE: Applies post-recoloring to '
                           'reduce artifacts. It is recommended if initial '
                           'results have some color bleeding/artifacts.')
  parser.add_argument('--aug_prob', dest='aug_prob', default=0.0, type=float,
                      help='There is no augmentation here, but if the trained '
                           'model originally was trained with aug_prob > 0.0, '
                           'this should be the same here.')

  return parser.parse_args()


if __name__ == "__main__":
  args = get_args()
  torch.cuda.set_device(args.gpu)

  assert args.pixel_loss == 'L1' or args.pixel_loss == 'L2', (
    'pixel loss should be either L1 or L2')
  assert args.vgg_loss_weight >= 0, 'vgg loss weight should be >= 0'

  project_to_latent(
    results_dir=args.results_dir,
    models_dir=args.models_dir,
    name=args.name,
    latent_noise=args.latent_noise,
    optimize_noise=args.optimize_noise,
    add_noise=args.add_noise,
    random_styles=args.random_styles,
    load_from=args.load_from,
    image_size=args.image_size,
    target_hist=args.target_hist,
    generate=args.generate,
    network_capacity=args.network_capacity,
    transparent=args.transparent,
    num_train_steps=args.num_train_steps,
    learning_rate=args.learning_rate,
    save_every=args.save_every,
    fp16=args.fp16,
    post_recoloring=args.post_recoloring,
    upsampling_output=args.upsampling_output,
    upsampling_method=args.upsampling_method,
    swapping_levels=args.swapping_levels,
    pyramid_levels=args.pyramid_levels,
    level_blending=args.level_blending,
    pixel_loss=args.pixel_loss,
    vgg_loss_weight=args.vgg_loss_weight,
    pixel_loss_weight=args.pixel_loss_weight,
    style_reg_weight=args.style_reg_weight,
    noise_reg_weight=args.noise_reg_weight,
    trunc_psi=args.trunc_psi,
    fq_layers=args.fq_layers,
    fq_dict_size=args.fq_dict_size,
    attn_layers=args.attn_layers,
    hist_method=args.hist_method,
    hist_resizing=args.hist_resizing,
    hist_sigma=args.hist_sigma,
    hist_bin=args.hist_bin,
    hist_insz=args.hist_insz,
    input_image=args.input_image,
    aug_prob=args.aug_prob
  )
