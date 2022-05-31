"""Base class for color histogram blocks.

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
from functools import partial
from typing import Callable, List, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


_Device = Union[str, torch.device]


EPS = 1e-6


class HistBlock(nn.Module):
    """Histogram block for calculating colour histograms for 3-channel
    tensors.

    Args:
        h (int): histogram dimension size (number of bins).

        insz (int): Maximum size of the input image; if it is larger than
            this size, the image will be resized to (`insz`, `insz`).

        resizing (str): resizing method if applicable. Options are:
            'interpolation' and 'sampling'.

        method (str): the method used to count the number of pixels for
            each bin in the histogram feature. Options are:
            'thresholding', 'RBF' (radial basis function), and
            'inverse-quadratic'

        sigma (float): if the method value is 'RBF' or 'inverse-quadratic',
            then this is the sigma parameter of the kernel function.

        intensity_scale (Scaling): intensity scale method to obtain scaling
            values (I_y in  Equation 2).

        hist_boundary (list[int]): A list of histogram boundary values.
            The list must have two entries; additional values are ignored
            if present.

        device (str|device): computation device (name or instance)

    Methods:
        forward: accepts input image and returns its histogram feature.
            Note that unless the method is `Method.THRESHOLDING`, this is a
            differentiable function and can be easily integrated with
            the loss function. As mentioned in the paper,
            `Method.INVERSE_QUADTRATIC` was found more stable than
            `Method.RADIAL_BASIS_FUNCTION`.
    """
    def __init__(
        self,
        h: int,
        insz: int,
        resizing: str,
        method: str,
        sigma: float,
        intensity_scale: str,
        hist_boundary: List[int],
        device: _Device
    ) -> None:
        super().__init__()
        hist_boundary.sort()
        start, end = hist_boundary[:2]
        self.h = h
        self.device = torch.device(device)
        self.resize = _get_resizing(resizing, h, insz, self.device)
        self.kernel = _get_kernel(method, h, sigma, hist_boundary)
        self.scaling = _get_scaling(intensity_scale)
        self.delta = torch.linspace(
            start, end, steps=h, device=self.device, dtype=torch.float32
        ).unsqueeze(dim=0)

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


_KernelMethod = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------- Factory Functions -----------------------------

def _get_resizing(
    mode: str, h: int, max_size: int, device: _Device
) -> _KernelMethod:
    if mode == 'interpolation':
        return partial(_resizing_interpolate, max_size)
    elif mode == 'sampling':
        return partial(_resizing_sample, h, max_size, device)
    else:
        raise ValueError(
            f'Unknown resizing method: "{mode}". Supported methods are '
            '"interpolation" or "sampling"'
        )


def _get_scaling(intensity_scale: bool):
    return _intensity_scaling if intensity_scale else _no_scaling 


def _get_kernel(
    method: str, h: int, sigma: float, boundary: Sequence[int]
) -> _KernelMethod: 
    if method == 'thresholding':
        eps = (boundary[1] - boundary[0]) / (2 * h)
        return partial(_thresholding_kernel, h, eps)
    elif method == 'RBF':
        inv_sigma_sq = 1 / sigma ** 2
        return partial(_rbf_kernel, h, inv_sigma_sq)
    elif method == 'inverse-quadratic':
        inv_sigma_sq = 1 / sigma ** 2
        return partial(_inverse_quadratic_kernel, h, inv_sigma_sq)
    else:
        raise ValueError(
            f'Unknown kernel method: "{method}". Supported methods are '
            '"thresholding", "RBF", or "inverse-quadratic".'
        )


# ----------------------------- Resizing Kernels -----------------------------

def _resizing_interpolate(max_size: int, X: torch.Tensor) -> torch.Tensor:
    H, W = X.shape[2:]
    if H > max_size or W > max_size:
        return F.interpolate(
            X, size=(max_size, max_size), mode='bilinear', align_corners=False
        )
    return X


def _resizing_sample(
    h: int, max_size: int, device: _Device, X: torch.Tensor
) -> torch.Tensor:
    H, W = X.shape[2:]
    if H > max_size or W > max_size:
        index_H = torch.linspace(0, H - H/h, h, dtype=torch.int32).to(device)
        index_W = torch.linspace(0, W - W/h, h, dtype=torch.int32).to(device)
        sampled = X.index_select(dim=2, index=index_H)
        return sampled.index_select(dim=3, index=index_W)
    return X


# ---------------------------- Scaling Functions -----------------------------

def _no_scaling(_: torch.Tensor) -> int:
    return 1


def _intensity_scaling(X: torch.Tensor) -> torch.Tensor:
    XX = X ** 2
    return (XX[:, 0] + XX[:, 1] + XX[:, 2] + EPS).sqrt().unsqueeze(dim=1)


# ----------------------------- Kernel Functions -----------------------------

def _thresholding_kernel(h: int, eps: float, X: torch.Tensor) -> torch.Tensor:
    return (X.reshape(-1, h) <= eps).float()


def _rbf_kernel(h: int, inv_sigma_sq: float, X: torch.Tensor) -> torch.Tensor:
    Y = (X.reshape(-1, h) ** 2) * inv_sigma_sq
    return (-Y).exp()


def _inverse_quadratic_kernel(
    h: int, inv_sigma_sq: float, X: torch.Tensor
) -> torch.Tensor:
    Y = (X.reshape(-1, h) ** 2) * inv_sigma_sq
    return 1. / (1. + Y)
