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


class RGBuvHistBlock(HistBlock):
  def __init__(self, h=64, insz=150, resizing='interpolation',
               method='inverse-quadratic', sigma=0.02, intensity_scale=True,
               hist_boundary=None, green_only=False, device='cuda'):
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
      hist_boundary: a list of histogram boundary values. Default is [-3, 3].
      green_only: boolean variable to use only the log(g/r), log(g/b) channels.
        Default is False.

    Methods:
      forward: accepts input image and returns its histogram feature. Note that
        unless the method is 'thresholding', this is a differentiable function
        and can be easily integrated with the loss function. As mentioned in the
         paper, the 'inverse-quadratic' was found more stable than 'RBF' in our
         training.
    """
    super().__init__(
      h, insz, resizing, method, sigma, intensity_scale,
      hist_boundary or [-3, 3], device
    )
    self.green_only = green_only

  def forward(self, x):
    x_sampled = self.resize(x.clamp(0, 1))

    N = x_sampled.shape[0]  # size of mini-batch
    if x_sampled.shape[1] > 3:
      x_sampled = x_sampled[:, :3, :, :]
    X = torch.unbind(x_sampled, dim=0)
    C = 1 + int(not self.green_only) * 2
    hists = torch.zeros(N, C, self.h, self.h, device=self.device)
    for n in range(N):
      Ix = X[n].reshape(3, -1).t()
      Iy = self.scaling(Ix)
      if not self.green_only:
        Du, Dv = self._diff_uv(Ix, i=0, j=1, k=2)
        a = (Iy * Du).t()
        hists[n, 0, :, :] = torch.mm(a, Dv)

      Du, Dv = self._diff_uv(Ix, i=1, j=0, k=2)
      a = (Iy * Du).t()
      hists[n, int(not self.green_only), :, :] = torch.mm(a, Dv)

      if not self.green_only:
        Du, Dv = self._diff_uv(Ix, i=2, j=0, k=1)
        a = (Iy * Du).t()
        hists[n, 2, :, :] = torch.mm(a, Dv)

    # normalization
    norm = hists.view(-1, C * self.h * self.h).sum(dim=1).view(-1, 1, 1, 1)
    hists_normalized = hists / (norm + EPS)

    return hists_normalized

  def _diff_uv(self, X: torch.Tensor, i: int, j: int, k: int):
    U = ((X[:, i] + EPS).log() - (X[:, j] + EPS).log()).unsqueeze(dim=1)
    V = ((X[:, i] + EPS).log() - (X[:, k] + EPS).log()).unsqueeze(dim=1)
    Du = (U - self.delta).abs()
    Dv = (V - self.delta).abs()
    Du = self.kernel(Du)
    Dv = self.kernel(Dv)
    return Du, Dv
