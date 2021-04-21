import numpy as np
from numba import njit, stencil


# signal = np.dot(C, sigma)
@njit
def compute_signal (n, sigma, C, i, j):

  signal = np.zeros(n, dtype=np.int32)

  for k in range(C.size):
    signal[i[k]] += C[k] * sigma[j[k]]

  return signal


# state = 1, with prob=f(signal)
#       = 0, with 1-prob
@njit
def update_state (n, beta, signal, numActive):

  newSigma = np.zeros(n, dtype=np.int8)
  rand = np.random.rand(n)

  for i in range(n):
    if rand[i] < 1./ (1. + np.exp(-2.*beta * (signal[i]-0.5))):
      newSigma[i] = 1
      numActive += 1

  return newSigma, numActive


# A(t+1) = sigma*(1-alpha) + A(t)*alpha
@njit
def update_average_activity (n, sigma, alpha, avgActivity):

  newAvgActivity = np.empty(n, dtype=np.float32)
  par = 1. - alpha

  for i in range(n):
    newAvgActivity[i] = sigma[i]*par + avgActivity[i]*alpha

  return newAvgActivity


@stencil
def compute_branching_par (arr):

  par = 0.

  if arr[-1] != 0:
    par = arr[0] / arr[-1]

  return par
