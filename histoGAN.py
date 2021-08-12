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

from tqdm import tqdm
from histoGAN import Trainer, NanException
from histogram_classes.RGBuvHistBlock import RGBuvHistBlock
from datetime import datetime
import torch
import argparse
from retry.api import retry_call
import os
from PIL import Image
from torchvision import transforms
import numpy as np


SCALE = 1 / np.sqrt(2.0)


def train_from_folder(
    data='./dataset/',
    results_dir='./results',
    models_dir='./models',
    name='test',
    new=False,
    load_from=-1,
    image_size=128,
    network_capacity=16,
    transparent=False,
    batch_size=2,
    gradient_accumulate_every=8,
    num_train_steps=150000,
    learning_rate=2e-4,
    num_workers=None,
    save_every=1000,
    generate=False,
    save_noise_latent=False,
    target_noise_file=None,
    target_latent_file=None,
    num_image_tiles=8,
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
    alpha=2,
    target_hist=None,
    aug_prob=0.0,
    dataset_aug_prob=0.0,
    aug_types=None):


  model = Trainer(
    name,
    results_dir,
    models_dir,
    batch_size=batch_size,
    gradient_accumulate_every=gradient_accumulate_every,
    image_size=image_size,
    network_capacity=network_capacity,
    transparent=transparent,
    lr=learning_rate,
    num_workers=num_workers,
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
    dataset_aug_prob=dataset_aug_prob,
    aug_types=aug_types
  )

  if not new:
    model.load(load_from)
  else:
    model.clear()

  if generate:

    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    if save_noise_latent and not os.path.exists('temp'):
      os.mkdir('./temp')
    if save_noise_latent and not os.path.exists(f'./temp/{name}'):
      os.mkdir(f'./temp/{name}')
    if target_hist is None:
      raise Exception('No target histogram or image is given')
    extension = os.path.splitext(target_hist)[1]
    if extension == '.npy':
      hist = np.load(target_hist)
      h = torch.from_numpy(hist).to(device=torch.cuda.current_device())
      if num_image_tiles > 1:
        num_image_tiles = num_image_tiles - num_image_tiles % 2
        for i in range(int(np.log2(num_image_tiles))):
          h = torch.cat((h, h), dim=0)
      samples_name = ('generated-' +
                      f'{os.path.basename(os.path.splitext(target_hist)[0])}'
                      f'-{timestamp}')
      model.evaluate(samples_name, hist_batch=h,
                     num_image_tiles=num_image_tiles,
                     save_noise_latent=save_noise_latent,
                     load_noise_file=target_noise_file,
                     load_latent_file=target_latent_file)
      print(f'sample images generated at {results_dir}/{name}/{samples_name}')
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
      if num_image_tiles > 1:
        num_image_tiles = num_image_tiles - num_image_tiles % 2
        for i in range(int(np.log2(num_image_tiles))):
          h = torch.cat((h, h), dim=0)
      samples_name = ('generated-' +
                      f'{os.path.basename(os.path.splitext(target_hist)[0])}'
                      f'-{timestamp}')
      model.evaluate(samples_name, hist_batch=h,
                     num_image_tiles=num_image_tiles,
                     save_noise_latent=save_noise_latent,
                     load_noise_file=target_noise_file,
                     load_latent_file=target_latent_file)
      print(f'sample images generated at {results_dir}/{name}/{samples_name}')
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
        elif (extension == str.lower(extension) == '.jpg' or str.lower(
            extension) == '.png'):
          img = Image.open(f)
          img = torch.unsqueeze(transform(img), dim=0).to(
            device=torch.cuda.current_device())
          h = histblock(img)
        else:
          print(f'Warning: File extension of {f} is not supported.')
          continue
        if num_image_tiles > 1:
          num_image_tiles = num_image_tiles - num_image_tiles % 2
          for i in range(int(np.log2(num_image_tiles))):
            h = torch.cat((h, h), dim=0)
        samples_name = ('generated-' +
                        f'{os.path.basename(os.path.splitext(f)[0])}'
                        f'-{timestamp}')
        model.evaluate(samples_name, hist_batch=h,
                       num_image_tiles=num_image_tiles,
                       save_noise_latent=save_noise_latent,
                       load_noise_file=target_noise_file,
                       load_latent_file=target_latent_file)
        print(f'sample images generated at {results_dir}/{name}/'
              f'{samples_name}')
    else:
      print('The file extension of target image is not supported.')
      raise NotImplementedError
    return

  print('\nStart training....\n')
  print(f'Alpha = {alpha}')
  model.set_data_src(data)
  for _ in tqdm(range(num_train_steps - model.steps), mininterval=10.,
                desc=f'{name}<{data}>'):
    retry_call(model.train, fargs=[alpha], tries=3, exceptions=NanException)

    if _ % 50 == 0:
      model.print_log()


def get_args():
  parser = argparse.ArgumentParser(description='Train/Test HistoGAN.')
  parser.add_argument('--data', dest='data', default='./dataset/')
  parser.add_argument('--results_dir', dest='results_dir',
                      default='./results_HistoGAN')
  parser.add_argument('--models_dir', dest='models_dir', default='./models')
  parser.add_argument('--target_hist', dest='target_hist', default=None)
  parser.add_argument('--name', dest='name', default='histoGAN_model')
  parser.add_argument('--new', dest='new', default=False)
  parser.add_argument('--load_from', dest='load_from', default=-1)
  parser.add_argument('--image_size', dest='image_size', default=256, type=int)
  parser.add_argument('--network_capacity', dest='network_capacity', default=16,
                      type=int)
  parser.add_argument('--transparent', dest='transparent', default=False)
  parser.add_argument('--batch_size', dest='batch_size', default=2, type=int)
  parser.add_argument('--gradient_accumulate_every',
                      dest='gradient_accumulate_every', default=8, type=int)
  parser.add_argument('--num_train_steps', dest='num_train_steps',
                      default=1500000, type=int)
  parser.add_argument('--learning_rate', dest='learning_rate', default=2e-4,
                      type=float)
  parser.add_argument('--num_workers', dest='num_workers', default=None)
  parser.add_argument('--save_every', dest='save_every', default=5000,
                      type=int)
  parser.add_argument('--generate', dest='generate', default=False)
  parser.add_argument('--save_noise_latent', dest='save_n_l', default=False)
  parser.add_argument('--target_noise_file', dest='target_n', default=None)
  parser.add_argument('--target_latent_file', dest='target_l', default=None)
  parser.add_argument('--num_image_tiles', dest='num_image_tiles',
                      default=16, type=int)
  parser.add_argument('--trunc_psi', dest='trunc_psi', default=0.75,
                      type=float)
  parser.add_argument('--fp 16', dest='fp16', default=False)
  parser.add_argument('--fq_layers', dest='fq_layers', default=[])
  parser.add_argument('--fq_dict_size', dest='fq_dict_size', default=256,
                      type=int)
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
  parser.add_argument('--alpha', dest='alpha', default=2, type=float)
  parser.add_argument('--aug_prob', dest='aug_prob', default=0.0, type=float,
                      help='Probability of discriminator augmentation. It '
                           'applies operations specified in --aug_types.')
  parser.add_argument('--dataset_aug_prob', dest='dataset_aug_prob',
                      default=0.0, type=float,
                      help='Probability of dataset augmentation. It applies '
                           'random cropping')
  parser.add_argument('--aug_types', dest='aug_types',
                      default=['translation', 'cutout'], nargs='+',
                      help='Options include: translation, cutout, and color')

  return parser.parse_args()


if __name__ == "__main__":
  args = get_args()
  torch.cuda.set_device(args.gpu)
  train_from_folder(
    data=args.data,
    results_dir=args.results_dir,
    models_dir=args.models_dir,
    name=args.name,
    new=args.new,
    load_from=args.load_from,
    image_size=args.image_size,
    network_capacity=args.network_capacity,
    transparent=args.transparent,
    batch_size=args.batch_size,
    gradient_accumulate_every=args.gradient_accumulate_every,
    num_train_steps=args.num_train_steps,
    learning_rate=args.learning_rate,
    num_workers=args.num_workers,
    save_every=args.save_every,
    generate=args.generate,
    save_noise_latent=args.save_n_l,
    target_noise_file=args.target_n,
    target_latent_file=args.target_l,
    num_image_tiles=args.num_image_tiles,
    trunc_psi=args.trunc_psi,
    fp16=args.fp16,
    fq_layers=args.fq_layers,
    fq_dict_size=args.fq_dict_size,
    attn_layers=args.attn_layers,
    hist_method=args.hist_method,
    hist_resizing=args.hist_resizing,
    hist_sigma=args.hist_sigma,
    hist_bin=args.hist_bin,
    hist_insz=args.hist_insz,
    target_hist=args.target_hist,
    alpha=args.alpha,
    aug_prob=args.aug_prob,
    dataset_aug_prob=args.dataset_aug_prob,
    aug_types=args.aug_types
  )
