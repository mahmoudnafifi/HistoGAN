import cv2 as cv
import torch
import numpy as np
import utils.imresize as resize


def pyramid_upsampling(target, reference, levels=5, swapping_levels=1,
                       blending=False):

  target = torch.squeeze(target, dim=0).cpu().detach().numpy()
  target = target.transpose(1, 2, 0)
  target[target < 0] = 0.0
  target[target > 1] = 1.0
  reference = torch.squeeze(reference, dim=0).cpu().detach().numpy()
  reference = reference.transpose(1, 2, 0)

  h, w, _ = reference.shape
  if w % (2 ** levels) == 0:
    new_size_w = w
  else:
    new_size_w = w + (2 ** levels) - w % (2 ** levels)

  if h % (2 ** levels) == 0:
    new_size_h = h
  else:
    new_size_h = h + (2 ** levels) - h % (2 ** levels)

  new_size = (new_size_h, new_size_w)
  if not ((h, w) == new_size):
    reference = resize.imresize(reference, output_shape=new_size)


  target = resize.imresize(target, output_shape=reference.shape[:2])

  target = target.astype(float)
  reference = reference.astype(float)

  # generate Gaussian pyramid for target
  G = target.copy()
  gpA = [G]
  for i in range(levels):
    G = cv.pyrDown(G)
    gpA.append(G)

  # generate Gaussian pyramid for reference
  G = reference.copy()
  gpB = [G]
  for i in range(levels):
    G = cv.pyrDown(G)
    gpB.append(G)

  # generate Laplacian Pyramid for reference
  lpB = [gpB[levels - 1]]
  for i in range(levels - 1, 0, -1):
    GE = cv.pyrUp(gpB[i])
    L = cv.subtract(gpB[i - 1], GE)
    lpB.append(L)

  # generate Laplacian Pyramid for target
  lpA = [gpA[levels - 1]]
  for i in range(levels - 1, 0, -1):
    GE = cv.pyrUp(gpA[i])
    L = cv.subtract(gpA[i - 1], GE)
    lpA.append(L)

  # now reconstruct
  for i in range(swapping_levels):
    lpB[i] = lpA[i]
  if blending:
    weights = np.linspace(0.0, 1.0, levels - swapping_levels + 1)
    for i in range(swapping_levels, levels):
      lpB[i] = (1 - weights[i]) * lpA[i] + weights[i] * lpB[i]


  output = lpB[0]
  for i in range(1, levels):
    output = cv.pyrUp(output)
    output = cv.add(output, lpB[i])


  output = torch.unsqueeze(torch.from_numpy(output.transpose(2, 0, 1)), dim=0)
  return output
