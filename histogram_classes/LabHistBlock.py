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
Portions Copyright (c) 2022 Patrick Levin.
"""
from histogram_classes.HistBlock import EPS, HistBlock
import torch


class LabHistBlock(HistBlock):
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
    super().__init__(
      h, insz, resizing, method, sigma, intensity_scale, 
      hist_boundary or [0, 1], device
    )

  def forward(self, x):
    x_sampled = self.resize(x.clamp(0, 1))

    N = x_sampled.shape[0]  # size of mini-batch
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]
    X = torch.unbind(x_sampled, dim=0)
    hists = torch.zeros(N, 1, self.h, self.h, device=self.device)
    for n in range(N):
      Ix = X[n].reshape(3, -1).t()

      Ia = Ix[:, 1].unsqueeze(dim=1)
      Ib = Ix[:, 2].unsqueeze(dim=1)

      diff_a = (Ia - self.delta).abs()
      diff_b = (Ib - self.delta).abd()

      diff_a = self.kernel(diff_a)
      diff_b = self.kernel(diff_b)
      Iy = self.scaling(Ix)
      a = torch.t(Iy * diff_a)

      hists[n, 0, :, :] = torch.mm(a, diff_b)

    # normalization
    norm = hists.view(-1, self.h * self.h).sum(dim=1).view(-1, 1, 1, 1) + EPS
    hists_normalized = hists / norm

    return hists_normalized