import numpy as np

from hypothesis import strategies as st
from hypothesis import given

from socmodel.source.connectivity import ZerosConnectivity
from socmodel.source.connectivity import OnesConnectivity
from socmodel.source.connectivity import RandomConnectivity


@given(size = st.integers(min_value=1, max_value=1e3),)
def test_ZerosConnectivity (size):

  connectivity = ZerosConnectivity().get(shape=(size,size))

  assert (connectivity == 0).all()


@given(size = st.integers(min_value=1, max_value=1e3),)
def test_OnesConnectivity (size):

  I = np.eye(size, dtype=bool)
  connectivity1 = OnesConnectivity().get(shape=(size,size))

  assert (connectivity1[I] == 0).all()
  assert (connectivity1[~I] == 1).all()

  connectivity2 = OnesConnectivity(negative=True).get(shape=(size,size))

  assert (connectivity2[I] == 0).all()
  assert (connectivity2[~I] == -1).all()


@given(size   = st.integers(min_value=1, max_value=1e3),
       pPlus  = st.floats(min_value=0., max_value=0.5),
       pMinus = st.floats(min_value=0., max_value=0.5),)
def test_RandomConnectivity (size, pPlus, pMinus):

  I = np.eye(size, dtype=bool)
  connectivity = RandomConnectivity(pPlus=pPlus, pMinus=pMinus).get(shape=(size,size))

  assert (connectivity[I] == 0).all()
  assert ((connectivity[~I] == -1) | (connectivity[~I] == 0) | (connectivity[~I] == 1)).all()
