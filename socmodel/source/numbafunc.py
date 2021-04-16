from random import random

import numpy as np
from numba import njit, stencil


@njit
def compute_signal (i, n, sigma, C):

  signal = 0

  for j in range(n):

    signal += C[i,j] * sigma[j]

  return signal


@njit
def compute_average_activity (S, A, alpha):

  avgActivity = S * (1. - alpha) + A * alpha

  return avgActivity


@njit
def compute_new_state (n, sigma, C, alpha, beta, avgActivity, numActive):

  newSigma = np.zeros(n, dtype=np.int8)

  for i in range(n):

    signal = compute_signal(i=i, n=n, sigma=sigma, C=C)

    if random() < 1. / (1. + np.exp(-2. * beta * (signal - 0.5))):
      newSigma[i] = 1
      numActive += 1

  avgActivity = compute_average_activity(S=newSigma, A=avgActivity, alpha=alpha)

  return newSigma, avgActivity, numActive


@stencil
def compute_branching_parameter (arr):

  par = 0.

  if arr[-1] != 0:
    par = arr[0] / arr[-1]

  return par
