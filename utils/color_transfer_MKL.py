import numpy as np

EPS = 2.2204e-16


def color_transfer_MKL(source, target):
  assert len(source.shape) == 3, 'Images should have 3 dimensions'
  assert source.shape[-1] == 3, 'Images should have 3 channels'
  X0 = np.reshape(source, (-1, 3), 'F')
  X1 = np.reshape(target, (-1, 3), 'F')
  A = np.cov(X0, rowvar=False)
  B = np.cov(X1, rowvar=False)
  T = MKL(A, B)
  mX0 = np.mean(X0, axis=0)
  mX1 = np.mean(X1, axis=0)
  XR = (X0 - mX0) @ T + mX1
  IR = np.reshape(XR, source.shape, 'F')
  IR = np.real(IR)
  IR[IR > 1] = 1
  IR[IR < 0] = 0
  return IR


def MKL(A, B):
  Da2, Ua = np.linalg.eig(A)

  Da2 = np.diag(Da2)
  Da2[Da2 < 0] = 0
  Da = np.sqrt(Da2 + EPS)
  C = Da @ np.transpose(Ua) @ B @ Ua @ Da
  Dc2, Uc = np.linalg.eig(C)

  Dc2 = np.diag(Dc2)
  Dc2[Dc2 < 0] = 0
  Dc = np.sqrt(Dc2 + EPS)
  Da_inv = np.diag(1 / (np.diag(Da)))
  T = Ua @ Da_inv @ Uc @ Dc @ np.transpose(Uc) @ Da_inv @ np.transpose(Ua)
  return T
