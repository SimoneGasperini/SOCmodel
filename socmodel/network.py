from collections import deque

import numpy as np
from numba import njit
from tqdm import trange

from socmodel.state import ZerosState
from socmodel.connectivity import ZerosConnectivity


class Network ():

  '''
  Parameters
  ----------
    n : int, default=100
      Size of the model (number of neurons). It must be greater or equal than 1

    state_init : BaseState, default=ZerosState()
      State vector initialization object

    connectivity_init : BaseConnectivity, default=ZerosConnectivity()
      Connectivity matrix initialization object

    beta : float, default=1.
      Inverse temperature of the model. It must be greater or equal than 0

    W : int, default=1
      Number of last time steps to be considered in computing the average
      neurons activity (it represents the temporal memory window of the neurons).
      It must be greater of equal than 1

    T : int, default=1
      Number of time steps to wait between each single neuron rewiring
      (it represents the time scale separation between fast neurons dynamics
      and slow change in their connectivity). It must be greater of equal than 1

    seed : int, default=None
      Random seed
  '''

  def __init__ (self, n=100,
                state_init=ZerosState(),
                connectivity_init=ZerosConnectivity(),
                beta=1., W=1, T=1, seed=None):

    self.n = n
    self.state_init = state_init
    self.connectivity_init = connectivity_init
    self.beta = beta
    self.W = W
    self.T = T
    self.seed = seed

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

    if not self.beta >= 0.:
      raise ValueError('Invalid "beta" passed. "beta" must be a float greater or equal that 0.')

    if not isinstance(self.W, int):
      raise TypeError('Invalid "W" passed. "W" must be an int.')

    if not self.W >= 1:
      raise ValueError('Invalid "W" passed. "W" must be greater or equal that 1.')

    if not isinstance(self.T, int):
      raise TypeError('Invalid "T" passed. "T" must be an int.')

    if not self.T >= 1:
      raise ValueError('Invalid "T" passed. "T" must be greater or equal that 1.')


  def _set_initial_conditions (self):

    np.random.seed(self.seed)

    self.sigma = self.state_init.get(size=self.n)
    self.C = self.connectivity_init.get(shape=(self.n,self.n))

    self._history = deque(maxlen=self.W)
    self._history.append(self.sigma)

    self.linksPlus = np.sum(self.C == 1)
    self.linksMinus = np.sum(self.C == -1)


  @staticmethod
  @njit
  def _compute_probability (n, sigma, C, beta):

    probability = np.empty(n, dtype=np.float32)

    for i in range(n):
      signal = 0
      for j in range(n):
        signal += C[i,j] * sigma[j]
      probability[i] = 1. / (1. + np.exp(-2.*beta * (signal-0.5)))

    return probability


  def _compute_average_activity (self, i):

    history = np.array(self._history)
    return np.mean(history[:,i])


  def _update_state (self):

    probability = self._compute_probability(n=self.n, sigma=self.sigma, C=self.C, beta=self.beta)
    self.sigma = (np.random.rand(self.n) < probability).astype(np.int8)

    self._history.append(self.sigma)


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


  def _perform_rewiring (self):

    i = np.random.randint(low=0, high=self.n)
    A = self._compute_average_activity(i=i)

    if A == 0.:
      self._add_random_linkPlus(i)

    elif A == 1.:
      self._add_random_linkMinus(i)

    else:
      self._remove_random_link(i)


  def run (self, num_steps):

    degPlus = np.empty(num_steps)
    degMinus = np.empty(num_steps)
    norm = 1. / self.n

    for i in trange(num_steps, desc='Simulation: '):

      self._update_state()

      if (i + 1) % self.T == 0:
        self._perform_rewiring()

      degPlus[i] = self.linksPlus * norm
      degMinus[i] = self.linksMinus * norm

    return degPlus, degMinus
