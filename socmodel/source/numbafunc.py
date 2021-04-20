from random import random

import numpy as np
from numba import njit, stencil


@njit
def compute_neuron_signal (i, n, sigma, C):
  signal = 0
  for j in range(n):
    signal += C[i,j] * sigma[j]
  return signal

@njit
def compute_new_state (n, sigma, C, beta, numActive):
  newSigma = np.zeros(n, dtype=np.int8)
  for i in range(n):
    signal = compute_neuron_signal(i=i, n=n, sigma=sigma, C=C)
    if random() < 1./ (1. + np.exp(-2.*beta * (signal-0.5))):
      newSigma[i] = 1
      numActive += 1
  return newSigma, numActive

@stencil
def compute_branching_par (arr):
  par = 0.
  if arr[-1] != 0:
    par = arr[0] / arr[-1]
  return par
