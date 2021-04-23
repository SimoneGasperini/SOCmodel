import numpy as np


class BaseConnectivity ():

  '''
  Base class for connectivity matrix initialization.
  '''


class ZerosConnectivity (BaseConnectivity):

  '''
  Initialize the connectivity matrix with all 0s. It corresponds to the
  generation of a network with no links.
  '''

  def __init__ (self):
    super(ZerosConnectivity, self).__init__()

  def get (self, shape):
    return np.zeros(shape=shape, dtype=np.int8)


class OnesConnectivity (BaseConnectivity):

  '''
  Initialize the connectivity matrix with all 1s (or -1s) apart the elements
  on the main diagonal (set to 0).
  It corresponds to the generation of a fully connected network with no loops.
  '''

  def __init__ (self, negative=False):
    self.negative = negative
    super(OnesConnectivity, self).__init__()

  def get (self, shape):
    connectivity = np.ones(shape=shape, dtype=np.int8)
    np.fill_diagonal(connectivity, val=0)
    return connectivity if not self.negative else -connectivity


class RandomConnectivity (BaseConnectivity):

  '''
  Initialize randomly the connectivity matrix with 1s, -1s (or 0s) according
  to their link creation probabilities.
  It corresponds to the generation of a directed random network with no loops.
  '''

  def __init__ (self, pPlus=0.5, pMinus=0.5):
    self.pPlus = pPlus
    self.pMinus = pMinus
    super(RandomConnectivity, self).__init__()

  def get (self, shape):
    pZero = 1. - (self.pPlus + self.pMinus)
    connectivity = (np.random.choice([-1,0,1], size=shape,
                                     p=[self.pMinus,pZero,self.pPlus])).astype(np.int8)
    np.fill_diagonal(connectivity, val=0)
    return connectivity
