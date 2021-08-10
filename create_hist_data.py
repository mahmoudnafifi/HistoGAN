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
from os import listdir
from os.path import isfile, join


def hist_interpolation(hist1, hist2):
  ratio = torch.rand(1)
  return hist1 * ratio + hist2 * (1 - ratio)


torch.cuda.set_device(0)

histblock = RGBuvHistBlock(insz=250, h=64,
                           resizing='sampling',
                           method='inverse-quadratic',
                           sigma=0.02,
                           device=torch.cuda.current_device())
transform = transforms.Compose([transforms.ToTensor()])

files = [join('histogram_data', f) for f in listdir('histogram_data') if
         isfile(join('histogram_data', f))]
first = True
for f in files:
  img_hist = Image.open(f)
  img_hist = torch.unsqueeze(transform(img_hist), dim=0).to(
    device=torch.cuda.current_device())
  h = histblock(img_hist)
  if first:
    histograms = h
    first = False
  else:
    histograms = torch.cat((histograms, h), dim=0)
histograms = torch.unsqueeze(histograms, dim=1)
histograms = histograms.cpu().numpy()
np.save('histogram_data/histograms.npy', histograms)

