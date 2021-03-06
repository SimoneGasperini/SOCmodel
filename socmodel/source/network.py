import numpy as np
from scipy import sparse
from tqdm import trange

from socmodel.source.state import ZerosState
from socmodel.source.connectivity import ZerosConnectivity
from socmodel.source.numbafunc import compute_signal
from socmodel.source.numbafunc import update_state
from socmodel.source.numbafunc import update_average_activity
from socmodel.source.numbafunc import compute_branching_par

import warnings
warnings.simplefilter('ignore')


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

    tau : int
      Number of steps for the evolution time scale: it represents the time scale
      separation between fast neurons dynamics (state evolution) and slow change
      in the network topology (connectivity evolution).
      It must be greater of equal than 1

    sigma_init : BaseState, default=ZerosState()
      State vector initializer

    C_init : BaseConnectivity, default=ZerosConnectivity()
      Connectivity matrix initializer
  '''

  def __init__ (self, n, alpha, beta, tau,
                sigma_init=ZerosState(), C_init=ZerosConnectivity()):

    self.n          = n
    self.alpha      = alpha
    self.beta       = beta
    self.tau        = tau
    self.sigma_init = sigma_init
    self.C_init     = C_init

    self._check_parameters()
    self._set_initial_conditions()


  def __repr__ (self):

    class_name = self.__class__.__qualname__
    params = list(self.__init__.__code__.co_varnames)
    params.remove('self')
    args = ', '.join([f'{key}={getattr(self, key)}' for key in params])

    return f'{class_name}({args})'


  def _check_parameters (self):

    if not np.issubdtype(type(self.n), int):
      raise TypeError('Invalid "n" passed. "n" must be an int.')

    if not self.n >= 1:
      raise ValueError('Invalid "n" passed. "n" must be greater or equal than 1.')

    if not 0. <= self.alpha <= 1.:
      raise ValueError('Invalid "alpha" passed. "alpha" must be a float in [0,1].')

    if not self.beta >= 0.:
      raise ValueError('Invalid "beta" passed. "beta" must be a float greater or equal that 0.')

    if not np.issubdtype(type(self.tau), int):
      raise TypeError('Invalid "tau" passed. "tau" must be an int.')

    if not self.tau >= 1:
      raise ValueError('Invalid "tau" passed. "tau" must be greater or equal that 1.')


  def _set_initial_conditions (self):

    self.sigma = self.sigma_init.get(size=self.n)
    self.C = sparse.coo_matrix(self.C_init.get(shape=(self.n,self.n)))

    self.avgActivity = np.empty(self.n, dtype=np.float32)
    self.avgActivity = self.sigma
    self.epsilon = 1e-9

    self.linksPlus = np.sum(self.C == 1)
    self.linksMinus = np.sum(self.C == -1)


  def _update_state (self, numActive):

    signal = compute_signal(n=self.n, sigma=self.sigma,
                            C=self.C.data, i=self.C.row, j=self.C.col)

    return update_state(n=self.n, beta=self.beta, signal=signal, numActive=numActive)


  def _update_average_activity (self):

    return update_average_activity(n=self.n, sigma=self.sigma, alpha=self.alpha,
                                   avgActivity=self.avgActivity)


  def _evolve_state (self):

    numActive = 0

    for _ in range(self.tau):
      self.sigma, numActive = self._update_state(numActive=numActive)
      self.avgActivity = self._update_average_activity()

    return numActive


  def _add_random_linkPlus (self, i):

    indices = np.where(self.C[i].toarray()[0] == 0)[0]
    indices = indices[indices != i]

    if indices.size:
      j = np.random.choice(indices)
      self.C[i,j] = 1
      self.linksPlus += 1


  def _add_random_linkMinus (self, i):

    indices = np.where(self.C[i].toarray()[0] == 0)[0]
    indices = indices[indices != i]

    if indices.size:
      j = np.random.choice(indices)
      self.C[i,j] = -1
      self.linksMinus += 1


  def _remove_random_link (self, i):

    indices = self.C[i].nonzero()[1]

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

    for i in trange(evolution_steps, desc='Simulation: ', disable=(not progressbar), ncols=100):

      avgActive[i] = self._evolve_state()
      self.C = self.C.tocsr()
      self._evolve_connectivity()
      self.C = self.C.tocoo()
      degPlus[i] = self.linksPlus
      degMinus[i] = self.linksMinus

    avgActive /= (self.tau * self.n)
    degPlus /= self.n
    degMinus /= self.n
    branchPar = compute_branching_par(avgActive)

    return degPlus, degMinus, branchPar
