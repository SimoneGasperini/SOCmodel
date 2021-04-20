import numpy as np


class BaseState ():

  '''
  Base class for state vector initialization.
  '''

  def get (self, size):
    '''
    Initialize the state vector according to the specialization.

    Parameters
    ----------
      size : int
        State vector size

    Returns
    -------
      state : array-like
        State vector with the given size
    '''

    raise NotImplementedError


  def __repr__ (self):

    class_name = self.__class__.__qualname__
    try:
      params = super(type(self), self).__init__.__code__.co_varnames
    except AttributeError:
      params = self.__init__.__code__.co_varnames

    params = list(self.__init__.__code__.co_varnames)
    params.remove('self')
    args = ', '.join([f'{key}={getattr(self, key)}' for key in params])

    return f'{class_name}({args})'


class ZerosState (BaseState):

  '''
  Initialize the state vector with all 0s. It corresponds to the generation of
  a state with all the neurons in a resting state.
  '''

  def __init__ (self):
    super(ZerosState, self).__init__()

  def get (self, size):
    return np.zeros(shape=size, dtype=np.int8)


class OnesState (BaseState):

  '''
  Initialize the state vector with all 1s. It corresponds to the generation of
  a state with all the neurons in a firing state.
  '''

  def __init__ (self):
    super(OnesState, self).__init__()

  def get (self, size):
    return np.ones(shape=size, dtype=np.int8)


class RandomState (BaseState):

  '''
  Initialize randomly the state vector with 1s (or 0s) according to the given
  probability. It corresponds to the generation of a state with some of the
  neurons in a firing state and the remaining neurons in a resting state.
  '''

  def __init__ (self, p=0.5):
    self.p = p
    super(RandomState, self).__init__()

  def get (self, size):
    return np.where(np.random.rand(size) < self.p, 1, 0).astype(np.int8)
