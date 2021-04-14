from random import random

import numpy as np
from numba import njit, stencil
from tqdm import trange

from socmodel.source.state import ZerosState
from socmodel.source.connectivity import ZerosConnectivity


class Network:

  '''
  Parameters
  ----------
    n : int
      Size of the model (number of neurons).
      It must be greater or equal than 1

    alpha : float
      Temporal memory of the model.
      It must be between 0 (no memory) and 1 (full memory)

    beta : float
      Inverse temperature of the model.
      It must be greater or equal than 0

    T : int
      Number of steps for the evolution time scale: it represents the time scale
      separation between fast neurons dynamics (state evolution) and slow change
      in the network topology (connectivity evolution).
      It must be greater of equal than 1

    state_init : BaseState, default=ZerosState()
      State vector initializer

    connectivity_init : BaseConnectivity, default=ZerosConnectivity()
      Connectivity matrix initializer
  '''

  def __init__ (self, n, alpha, beta, T,
                state_init=ZerosState(),
                connectivity_init=ZerosConnectivity()):

    self.n                 = n
    self.alpha             = alpha
    self.beta              = beta
    self.T                 = T
    self.state_init        = state_init
    self.connectivity_init = connectivity_init

    self._check_parameters()
    self._set_initial_conditions()


  def __repr__ (self):

    class_name = self.__class__.__qualname__
    params = list(self.__init__.__code__.co_varnames)
    params.remove('self')
    args = ', '.join([f'{key}={getattr(self, key)}' for key in params])

    return f'{class_name}({args})'


  def _check_parameters (self):

    if not isinstance(self.n, int):
      raise TypeError('Invalid "n" passed. "n" must be an int.')

    if not self.n >= 1:
      raise ValueError('Invalid "n" passed. "n" must be greater or equal than 1.')

    if not 0. <= self.alpha <= 1.:
      raise ValueError('Invalid "alpha" passed. "alpha" must be a float in [0,1].')

    if not self.beta >= 0.:
      raise ValueError('Invalid "beta" passed. "beta" must be a float greater or equal that 0.')

    if not isinstance(self.T, int):
      raise TypeError('Invalid "T" passed. "T" must be an int.')

    if not self.T >= 1:
      raise ValueError('Invalid "T" passed. "T" must be greater or equal that 1.')


  def _set_initial_conditions (self):

    self.sigma = self.state_init.get(size=self.n)
    self.C = self.connectivity_init.get(shape=(self.n,self.n))

    self.avgActivity = np.empty(self.n, dtype=np.float32)
    self.avgActivity = self.sigma
    self.epsilon = 1e-9

    self.linksPlus = np.sum(self.C == 1)
    self.linksMinus = np.sum(self.C == -1)


  @staticmethod
  @stencil
  def _compute_branching_parameter (arr):

    par = 0.

    if arr[-1] != 0:
      par = arr[0] / arr[-1]

    return par


  @staticmethod
  @njit
  def _compute_new_state (n, sigma, C, alpha, beta, avgActivity, numActive):

    newSigma = np.zeros(n, dtype=np.int8)

    for i in range(n):

      signal = 0
      for j in range(n):
        signal += C[i,j] * sigma[j]

      if random() < 1./ (1. + np.exp(-2.*beta * (signal-0.5))):
        newSigma[i] = 1
        numActive += 1

    avgActivity = newSigma * (1. - alpha) + avgActivity * alpha

    return newSigma, avgActivity, numActive


  def _evolve_state (self):

    numActive = 0

    for _ in range(self.T):

      self.sigma, self.avgActivity, numActive = self._compute_new_state(
        n           = self.n,
        sigma       = self.sigma,
        C           = self.C,
        alpha       = self.alpha,
        beta        = self.beta,
        avgActivity = self.avgActivity,
        numActive   = numActive
      )

    return numActive


  def _add_random_linkPlus (self, i):

    indices = np.where(self.C[i] == 0)[0]
    indices = indices[indices != i]

    if indices.size:
      j = np.random.choice(indices)
      self.C[i,j] = 1
      self.linksPlus += 1


  def _add_random_linkMinus (self, i):

    indices = np.where(self.C[i] == 0)[0]
    indices = indices[indices != i]

    if indices.size:
      j = np.random.choice(indices)
      self.C[i,j] = -1
      self.linksMinus += 1


  def _remove_random_link (self, i):

    indices = np.where(self.C[i] != 0)[0]

    if indices.size:
      j = np.random.choice(indices)
      l = self.C[i,j]
      if l == 1: self.linksPlus -= 1
      if l == -1: self.linksMinus -= 1
      self.C[i,j] = 0


  def _evolve_connectivity (self):

    i = np.random.randint(low=0, high=self.n)
    A = self.avgActivity[i]

    if A < self.epsilon:
      self._add_random_linkPlus(i)

    elif A > (1. - self.epsilon):
      self._add_random_linkMinus(i)

    else:
      self._remove_random_link(i)


  def run (self, evolution_steps, progressbar=True):

    avgActive = np.empty(evolution_steps, dtype=np.float32)
    degPlus = np.empty(evolution_steps, dtype=np.float32)
    degMinus = np.empty(evolution_steps, dtype=np.float32)

    for i in trange(evolution_steps, desc='Simulation: ', disable=(not progressbar)):

      avgActive[i] = self._evolve_state()
      self._evolve_connectivity()
      degPlus[i] = self.linksPlus
      degMinus[i] = self.linksMinus

    avgActive /= (self.T * self.n)
    degPlus /= self.n
    degMinus /= self.n
    branchPar = self._compute_branching_parameter(avgActive)

    return degPlus, degMinus, branchPar, avgActive