from hypothesis import strategies as st
from hypothesis import given

from socmodel.source.state import ZerosState
from socmodel.source.state import OnesState
from socmodel.source.state import RandomState


@given(size = st.integers(min_value=1, max_value=1e6),)
def test_ZerosState (size):

  state = ZerosState().get(size=size)

  assert (state == 0).all()


@given(size = st.integers(min_value=1, max_value=1e6),)
def test_OnesState (size):

  state = OnesState().get(size=size)

  assert (state == 1).all()


@given(size = st.integers(min_value=1, max_value=1e6),
       p    = st.floats(min_value=0., max_value=1.),)
def test_RandomState (size, p):

  state = RandomState(p=p).get(size=size)

  assert ((state == 0) | (state == 1)).all()
