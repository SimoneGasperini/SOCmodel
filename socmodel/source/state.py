import numpy as np


class BaseState ():

  '''
  Base class for state vector initialization.
  '''


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
