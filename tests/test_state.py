from hypothesis import strategies as st
from hypothesis import given

from socmodel.source.state import ZerosState
from socmodel.source.state import OnesState
from socmodel.source.state import UniformState


@given(size = st.integers(min_value=1, max_value=1e6),)
def test_ZerosState (size):
  '''
  Test the ZerosState object (it contains only 0s).
  '''

  state = ZerosState().get(size=size)

  assert ((state == 0)).all()


@given(size = st.integers(min_value=1, max_value=1e6),)
def test_OnesState (size):
  '''
  Test the OnesState object (it contains only 1s).
  '''

  state = OnesState().get(size=size)

  assert ((state == 1)).all()


@given(size = st.integers(min_value=1, max_value=1e6),)
def test_UniformState (size):
  '''
  Test the UniformState object (it contains only 0s or 1s).
  '''

  state = UniformState().get(size=size)

  assert ((state == 0) | (state == 1)).all()
