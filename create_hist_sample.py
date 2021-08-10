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

from histogram_classes.RGBuvHistBlock import RGBuvHistBlock
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from os.path import splitext, join, basename, exists
from os import mkdir

filename = './target_images/1.jpg'
output_dir = './histograms/'

if exists(output_dir) is False:
  mkdir(output_dir)

torch.cuda.set_device(0)
histblock = RGBuvHistBlock(insz=250, h=64,
                           resizing='sampling',
                           method='inverse-quadratic',
                           sigma=0.02,
                           device=torch.cuda.current_device())
transform = transforms.Compose([transforms.ToTensor()])

img_hist = Image.open(filename)
img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(
  device=torch.cuda.current_device())
histogram = histblock(img_hist)
histogram = histogram.cpu().numpy()
np.save(join(output_dir, basename(splitext(filename)[0]) + '.npy'), histogram)

