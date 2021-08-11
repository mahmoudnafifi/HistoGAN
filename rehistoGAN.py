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
from ReHistoGAN import recoloringTrainer
from histoGAN import Trainer, NanException
from datetime import datetime
import torch
import argparse
from retry.api import retry_call
import os
from PIL import Image
from torchvision import transforms
import torchvision
import numpy as np
import copy
from utils.face_preprocessing import face_extraction
from histogram_classes.RGBuvHistBlock import RGBuvHistBlock


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


def hist_interpolation(hists):
  ratios = torch.abs(torch.rand(hists.shape[0])).to(
    device=torch.cuda.current_device())
  ratios = ratios / torch.sum(ratios)
  out_hist = hists[0, :, :, :, :] * ratios[0]
  for i in range(hists.shape[0] - 1):
    out_hist = out_hist + hists[i + 1, :, :, :, :] * ratios[i + 1]
  return out_hist


def process_image(model, name, input_image, target_hist, image_size=256,
                  upsampling_output=False,
                  upsampling_method='pyramid',
                  swapping_levels=1,
                  pyramid_levels=5,
                  level_blending=False,
                  post_recoloring=False,
                  sampling=True,
                  target_number=1, results_dir='./results_ReHistoGAN/',
                  hist_insz=150, hist_bin=64,
                  hist_method='inverse-quadratic', hist_resizing='sampling',
                  hist_sigma=0.02):

  img = Image.open(input_image)

  original_img = np.array(img) / 255

  if upsampling_output:
    width, height = img.size
    if width > image_size or height > image_size:
      resizing_mode = 'upscaling'
    elif width < image_size or height < image_size:
      resizing_mode = 'downscaling'
    else:
      resizing_mode = 'none'
  else:
    resizing_mode = None
    width = None
    height = None

  if width != image_size or height != image_size:
    img = img.resize((image_size, image_size))

  now = datetime.now()
  timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")

  postfix = round(np.random.rand() * 1000)
  transform = transforms.Compose([
    transforms.Lambda(convert_transparent_to_rgb),
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Lambda(expand_greyscale(3))
  ])

  img = torch.unsqueeze(transform(img), dim=0).to(
    device=torch.cuda.current_device())

  if target_hist is None:
    if sampling:
      target_histograms = np.load('histogram_data/histograms.npy')
      target_histograms = torch.tensor(target_histograms).to(
        device=torch.cuda.current_device())

      for j in range(target_number):
        inds = np.random.randint(0, high=target_histograms.shape[0],
                                 size=5)
        h = hist_interpolation(
          target_histograms[inds, :, :, :, :])
        with torch.no_grad():
          samples_name = f'{j}-output-{timestamp}-{postfix}'
          model.evaluate(samples_name, image_batch=img,
                         hist_batch=h,
                         resizing=resizing_mode,
                         resizing_method=upsampling_method,
                         swapping_levels=swapping_levels,
                         pyramid_levels=pyramid_levels,
                         level_blending=level_blending,
                         original_size=[width, height],
                         input_image_name=input_image,
                         original_image=original_img,
                         save_input=False,
                         post_recoloring=post_recoloring)
        print(f'recolored images generated '
              f'at {results_dir}/{name}/{samples_name}')

    else:
      raise Exception('No target histogram is given.')

  else:

    extension = os.path.splitext(target_hist)[1]
    if str.lower(extension) == '.npy':
      hist = np.load(target_hist)
      h = torch.from_numpy(hist).to(device=torch.cuda.current_device())
      samples_name = ('output-' +
                      f'{os.path.basename(os.path.splitext(target_hist)[0])}'
                      f'-{timestamp}-{postfix}')
      with torch.no_grad():
        model.evaluate(samples_name,
                       image_batch=img, hist_batch=h,
                       resizing=resizing_mode,
                       resizing_method=upsampling_method,
                       swapping_levels=swapping_levels,
                       pyramid_levels=pyramid_levels,
                       level_blending=level_blending,
                       original_size=[width, height],
                       original_image=original_img,
                       input_image_name=input_image,
                       save_input=False,
                       post_recoloring=post_recoloring)
        print(f'recolored images generated at {results_dir}/{name}/'
              f'{samples_name}')
    elif str.lower(extension) == '.jpg' or str.lower(extension) == '.png':
      histblock = RGBuvHistBlock(insz=hist_insz, h=hist_bin,
                                 resizing=hist_resizing,
                                 method=hist_method,
                                 sigma=hist_sigma,
                                 device=torch.cuda.current_device())
      transform = transforms.Compose([transforms.ToTensor()])
      img_hist = Image.open(target_hist)
      img_hist = torch.unsqueeze(
        transform(img_hist), dim=0).to(
        device=torch.cuda.current_device())
      with torch.no_grad():
        h = histblock(img_hist)
        samples_name = (
            'output-' +
            f'{os.path.basename(os.path.splitext(target_hist)[0])}'
            f'-{timestamp}-{postfix}')
        model.evaluate(samples_name, image_batch=img,
                       hist_batch=h,
                       resizing=resizing_mode,
                       resizing_method=upsampling_method,
                       swapping_levels=swapping_levels,
                       pyramid_levels=pyramid_levels,
                       level_blending=level_blending,
                       original_size=[width, height],
                       original_image=original_img,
                       input_image_name=input_image,
                       save_input=False,
                       post_recoloring=post_recoloring)
        print(f'recolored images generated at {results_dir}/{name}/'
              f'{samples_name}')

    elif extension == '':
      files = [os.path.join(target_hist, f) for f in os.listdir(target_hist)
               if
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
          img_hist = Image.open(f)
          img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(
            device=torch.cuda.current_device())
          h = histblock(img_hist)
        else:
          print(f'Warning: File extension of {f} is not supported.')
          continue
        with torch.no_grad():
          samples_name = ('output-' +
                          f'{os.path.basename(os.path.splitext(f)[0])}'
                          f'-{timestamp}-{postfix}')
          model.evaluate(samples_name, image_batch=img,
                         hist_batch=h,
                         resizing=resizing_mode,
                         resizing_method=upsampling_method,
                         swapping_levels=swapping_levels,
                         pyramid_levels=pyramid_levels,
                         level_blending=level_blending,
                         original_size=[width, height],
                         original_image=original_img,
                         input_image_name=input_image,
                         save_input=False,
                         post_recoloring=post_recoloring)
          print(f'recolored images generated at {results_dir}/{name}/'
                f'{samples_name}')

def train_from_folder(
    data='./dataset/',
    results_dir='./results_ReHistoGAN/',
    models_dir='./models/',
    histGAN_models_dir='./models/',
    name='test',
    new=False,
    load_from=-1,
    image_size=128,
    network_capacity=16,
    transparent=False,
    load_histogan_weights=True,
    batch_size=2,
    sampling=True,
    gradient_accumulate_every=8,
    num_train_steps=200000,
    learning_rate=2e-4,
    num_workers=None,
    save_every=10000,
    generate=False,
    trunc_psi=0.75,
    fp16=False,
    skip_conn_to_GAN=False,
    fq_layers=[],
    fq_dict_size=256,
    attn_layers=[],
    hist_method='inverse-quadratic',
    hist_resizing='sampling',
    hist_sigma=0.02,
    hist_bin=64,
    hist_insz=150,
    rec_loss='laplacian',
    alpha=32,
    beta=1.5,
    gamma=4,
    fixed_gan_weights=False,
    initialize_gan=False,
    variance_loss=False,
    target_hist=None,
    internal_hist=False,
    histoGAN_model_name=None,
    input_image=None,
    target_number=None,
    change_hyperparameters=False,
    change_hyperparameters_after=100000,
    upsampling_output=False,
    upsampling_method='pyramid',
    swapping_levels=1,
    pyramid_levels=6,
    level_blending=False,
    post_recoloring=False):
  model = recoloringTrainer(
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
    rec_loss=rec_loss,
    fixed_gan_weights=fixed_gan_weights,
    skip_conn_to_GAN=skip_conn_to_GAN,
    initialize_gan=initialize_gan,
    variance_loss=variance_loss,
    internal_hist=internal_hist,
    change_hyperparameters=change_hyperparameters,
    change_hyperparameters_after=change_hyperparameters_after
  )

  if not new:
    status = model.load(load_from)
    if load_histogan_weights and status == -1:
      if histoGAN_model_name is not None:
        GAN_model_name = histoGAN_model_name
      else:
        GAN_model_name = name.replace('_rehistoGAN', '_histoGAN')
      if os.path.exists(os.path.join(models_dir, name)):
        model_HistGAN = Trainer(
          GAN_model_name,
          results_dir,
          histGAN_models_dir,
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
        )
        model_HistGAN.load(args.load_from)
        model.GAN.G.blocks[0] = copy.deepcopy(model_HistGAN.GAN.GE.blocks[-2])
        model.GAN.G.blocks[1] = copy.deepcopy(model_HistGAN.GAN.GE.blocks[-1])
        model.GAN.H = copy.deepcopy(model_HistGAN.GAN.HE)

      else:
        raise Exception('GAN does not exist!')

  else:
    model.clear()
    if load_histogan_weights:
      if os.path.exists(os.path.join(models_dir, name)):
        if histoGAN_model_name is not None:
          GAN_model_name = histoGAN_model_name
        else:
          GAN_model_name = name.replace('_histoGAN', '_rehistoGAN')
        model_HistGAN = Trainer(
          GAN_model_name,
          results_dir,
          histGAN_models_dir,
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
        )
        model_HistGAN.load(args.load_from)
        model.GAN.G.blocks[0] = copy.deepcopy(model_HistGAN.GAN.GE.blocks[-2])
        model.GAN.G.blocks[1] = copy.deepcopy(model_HistGAN.GAN.GE.blocks[-1])
        model.GAN.H = copy.deepcopy(model_HistGAN.GAN.HE)
      else:
        raise Exception('GAN does not exist!')

  if generate:
    if input_image is None:
      raise Exception('No input image is given')

    extension = os.path.splitext(input_image)[1]
    if (extension == str.lower(extension) == '.jpg' or str.lower(
        extension) == '.png'):
      process_image(model, name, input_image, target_hist, image_size=256,
                    upsampling_output=upsampling_output,
                    upsampling_method=upsampling_method,
                    swapping_levels=swapping_levels,
                    pyramid_levels=pyramid_levels,
                    level_blending=level_blending,
                    post_recoloring=post_recoloring,
                    sampling=sampling,
                    target_number=target_number, results_dir=results_dir,
                    hist_insz=hist_insz, hist_bin=hist_bin,
                    hist_method=hist_method, hist_resizing=hist_resizing,
                    hist_sigma=hist_sigma)
    elif extension == '':
      files = [os.path.join(input_image, f) for f in
               os.listdir(input_image) if os.path.isfile(os.path.join(
          input_image, f))]
      for f in files:
        extension = os.path.splitext(f)[1]
        if (extension == str.lower(extension) == '.jpg' or str.lower(
            extension) == '.png'):
          process_image(model, name, f, target_hist, image_size=256,
                        upsampling_output=upsampling_output,
                        upsampling_method=upsampling_method,
                        swapping_levels=swapping_levels,
                        pyramid_levels=pyramid_levels,
                        level_blending=level_blending,
                        post_recoloring=post_recoloring,
                        sampling=sampling,
                        target_number=target_number, results_dir=results_dir,
                        hist_insz=hist_insz, hist_bin=hist_bin,
                        hist_method=hist_method, hist_resizing=hist_resizing,
                        hist_sigma=hist_sigma)

    else:
      raise Exception('File extension is not supported!')

    return

  print('\nStart training....\n')
  print(f'Alpha = {alpha}')
  print(f'Beta = {beta}')
  print(f'Gamma = {gamma}')

  model.set_data_src(data, not fixed_gan_weights)
  for _ in tqdm(range(num_train_steps - model.steps),
                mininterval=10., desc=f'{name}<{data}>'):
    retry_call(model.train, fargs=[alpha, beta, gamma], tries=3,
               exceptions=NanException)

    if _ % 50 == 0:
      model.print_log()


def get_args():
  parser = argparse.ArgumentParser(description='Train/Test ReHistoGAN.')
  parser.add_argument('--data', dest='data', default='./dataset/')
  parser.add_argument('--results_dir', dest='results_dir',
                      default='./results_ReHistoGAN')
  parser.add_argument('--models_dir', dest='models_dir', default='./models')
  parser.add_argument('--histGAN_models_dir', dest='histGAN_models_dir',
                      default='./models',
                      help='directory of pre-trained HistoGAN model')
  parser.add_argument('--histoGAN_model_name', dest='histoGAN_model_name',
                      default=None, type=str)
  parser.add_argument('--target_hist', dest='target_hist', default=None)
  parser.add_argument('--input_image', dest='input_image',
                      default='./input_images/other-objects/')
  parser.add_argument('--face_extraction', dest='face_extraction',
                      default=False, type=bool)
  parser.add_argument('--name', dest='name', default='reHistoGAN_model')
  parser.add_argument('--sampling', dest='sampling', default=False, type=bool,
                      help='for testing mode, if no target histogram is '
                           'given, use sampling process')
  parser.add_argument('--target_number', dest='target_number', default=50,
                      type=int,
                      help='number of recolored images; ignore if you specify '
                           'a target histogram')
  parser.add_argument('--new', dest='new', default=False)
  parser.add_argument('--load_from', dest='load_from', default=-1)
  parser.add_argument('--image_size', dest='image_size', default=256,
                      type=int)
  parser.add_argument('--network_capacity', dest='network_capacity',
                      default=16, type=int)
  parser.add_argument('--transparent', dest='transparent', default=False)
  parser.add_argument('--batch_size', dest='batch_size', default=2, type=int)
  parser.add_argument('--gradient_accumulate_every',
                      dest='gradient_accumulate_every', default=8, type=int)
  parser.add_argument('--num_train_steps', dest='num_train_steps',
                      default=200000, type=int)
  parser.add_argument('--learning_rate', dest='learning_rate', default=2e-4,
                      type=float)
  parser.add_argument('--num_workers', dest='num_workers', default=None)
  parser.add_argument('--save_every', dest='save_every', default=10000,
                      type=int)
  parser.add_argument('--trunc_psi', dest='trunc_psi', default=0.75,
                      type=float)
  parser.add_argument('--fp16', dest='fp16', default=False)
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
                      default='sampling')
  parser.add_argument('--hist_sigma', dest='hist_sigma', default=0.02,
                      type=float)
  parser.add_argument('--generate', dest='generate', default=False)
  parser.add_argument('--alpha', dest='alpha', default=32, type=float)
  parser.add_argument('--beta', dest='beta', default=1.5, type=float)
  parser.add_argument('--gamma', dest='gamma', default=2, type=float)
  parser.add_argument('--change_hyperparameters',
                      dest='change_hyperparameters', default=False, type=bool)
  parser.add_argument('--change_hyperparameters_after',
                      dest='change_hyperparameters_after', default=100000,
                      type=int)
  parser.add_argument('--rec_loss', dest='rec_loss', default='laplacian',
                      type=str,
                      help='reconstruction loss (sobel or laplacian)')
  parser.add_argument('--internal_hist', dest='internal_hist', default=False,
                      type=bool, help='Internal histogram injection. This was '
                                      'an ablation on a different design; not '
                                      'what we did in the official ReHistoGAN')
  parser.add_argument('--skip_conn_to_GAN', dest='skip_conn_to_GAN',
                      default=True, type=bool,
                      help='See Figures 4 and 6 in the paper.')
  parser.add_argument('--fixed_gan_weights', dest='fixed_gan_weights',
                      default=False,
                      help="To fix weights of the HistoGAN's head")
  parser.add_argument('--load_histoGAN_weights', dest='load_histoGAN_weights',
                      default=False, help="To load weights of HistoGAN's head")
  parser.add_argument('--initialize_gan', dest='initialize_gan',
                      default=True, type=bool)
  parser.add_argument('--variance_loss', dest='variance_loss',
                      default=True, type=bool)
  parser.add_argument('--upsampling_output',
                      dest='upsampling_output',
                      default=False, type=bool,
                      help='TESTING PHASE: Applies guided upsampling. It '
                           'is recommended if input image > 256x256.')
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

  return parser.parse_args()


if __name__ == "__main__":
  args = get_args()
  torch.cuda.set_device(args.gpu)

  if args.generate and args.face_extraction:
    if args.input_image is None:
      raise Exception('No input image is given')
    extension = os.path.splitext(args.input_image)[1]
    if (extension == str.lower(extension) == '.jpg' or str.lower(
      extension) == '.png'):
      face_extraction(args.input_image)
      input_image = f'./temp-faces/{os.path.split(args.input_image)[-1]}'
    elif extension == '':

      files = [os.path.join('./temp-faces/', f) for f in
               os.listdir('./temp-faces/') if os.path.isfile(os.path.join(
          './temp-faces/', f))]
      for f in files:
        os.remove(f)

      files = [os.path.join(args.input_image, f) for f in
               os.listdir(args.input_image) if os.path.isfile(os.path.join(
          args.input_image, f))]
      for f in files:
        extension = os.path.splitext(f)[1]
        if (extension == str.lower(extension) == '.jpg' or str.lower(
            extension) == '.png'):
          face_extraction(f)
      input_image = './temp-faces/'
    else:
      raise Exception('File extension is not supported!')
  else:
    input_image = args.input_image

  train_from_folder(
    data=args.data,
    results_dir=args.results_dir,
    models_dir=args.models_dir,
    name=args.name,
    new=args.new,
    histGAN_models_dir=args.histGAN_models_dir,
    load_from=args.load_from,
    load_histogan_weights=args.load_histoGAN_weights,
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
    beta=args.beta,
    gamma=args.gamma,
    skip_conn_to_GAN=args.skip_conn_to_GAN,
    fixed_gan_weights=args.fixed_gan_weights,
    sampling=args.sampling,
    rec_loss=args.rec_loss,
    initialize_gan=args.initialize_gan,
    variance_loss=args.variance_loss,
    input_image=input_image,
    internal_hist=args.internal_hist,
    histoGAN_model_name=args.histoGAN_model_name,
    target_number=args.target_number,
    change_hyperparameters=args.change_hyperparameters,
    change_hyperparameters_after=args.change_hyperparameters_after,
    upsampling_output=args.upsampling_output,
    upsampling_method=args.upsampling_method,
    swapping_levels=args.swapping_levels,
    pyramid_levels=args.pyramid_levels,
    level_blending=args.level_blending,
    post_recoloring=args.post_recoloring
  )
