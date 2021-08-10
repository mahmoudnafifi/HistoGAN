"""
##### Copyright 2021 Mahmoud Afifi.

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
####
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


EPS = 1e-6


class LabHistBlock(nn.Module):
  def __init__(self, h=64, insz=150, resizing='interpolation',
               method='inverse-quadratic', sigma=0.02, intensity_scale=False,
               hist_boundary=None, device='cuda'):
    """ Computes the Lab (CIELAB) histogram feature of a given image.
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
        Equation 2). Default value is False.
      hist_boundary: a list of histogram boundary values. Default is [0, 1].

    Methods:
      forward: accepts input image and returns its histogram feature. Note that
        unless the method is 'thresholding', this is a differentiable function
        and can be easily integrated with the loss function. As mentioned in the
         paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
         training.
    """
    super(LabHistBlock, self).__init__()
    self.h = h
    self.insz = insz
    self.device = device
    self.resizing = resizing
    self.method = method
    self.intensity_scale = intensity_scale
    if hist_boundary is None:
      hist_boundary = [0, 1]
    hist_boundary.sort()
    self.hist_boundary = hist_boundary
    if self.method == 'thresholding':
      self.eps = (abs(hist_boundary[0]) + abs(hist_boundary[1])) / h
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
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]
    X = torch.unbind(x_sampled, dim=0)
    hists = torch.zeros((x_sampled.shape[0], 1, self.h, self.h)).to(
      device=self.device)
    for l in range(L):
      I = torch.t(torch.reshape(X[l], (3, -1)))
      if self.intensity_scale:
        Il = torch.unsqueeze(I[:, 0], dim=1)
      else:
        Il = 1

      Ia = torch.unsqueeze(I[:, 1], dim=1)
      Ib = torch.unsqueeze(I[:, 2], dim=1)

      diff_a = abs(Ia - torch.unsqueeze(torch.tensor(np.linspace(
        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
        dim=0).to(self.device))
      diff_b = abs(Ib - torch.unsqueeze(torch.tensor(np.linspace(
        self.hist_boundary[0], self.hist_boundary[1], num=self.h)),
        dim=0).to(self.device))

      if self.method == 'thresholding':
        diff_a = torch.reshape(diff_a, (-1, self.h)) <= self.eps / 2
        diff_b = torch.reshape(diff_b, (-1, self.h)) <= self.eps / 2
      elif self.method == 'RBF':
        diff_a = torch.pow(torch.reshape(diff_a, (-1, self.h)),
                           2) / self.sigma ** 2
        diff_b = torch.pow(torch.reshape(diff_b, (-1, self.h)),
                           2) / self.sigma ** 2
        diff_a = torch.exp(-diff_a)  # Gaussian
        diff_b = torch.exp(-diff_b)
      elif self.method == 'inverse-quadratic':
        diff_a = torch.pow(torch.reshape(diff_a, (-1, self.h)),
                           2) / self.sigma ** 2
        diff_b = torch.pow(torch.reshape(diff_b, (-1, self.h)),
                           2) / self.sigma ** 2
        diff_a = 1 / (1 + diff_a)  # Inverse quadratic
        diff_b = 1 / (1 + diff_b)

      diff_a = diff_a.type(torch.float32)
      diff_b = diff_b.type(torch.float32)
      a = torch.t(Il * diff_a)

      hists[l, 0, :, :] = torch.mm(a, diff_b)

    # normalization
    hists_normalized = hists / (
        ((hists.sum(dim=1)).sum(dim=1)).sum(dim=1).view(-1, 1, 1, 1) + EPS)

    return hists_normalized