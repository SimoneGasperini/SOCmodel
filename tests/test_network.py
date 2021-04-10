import numpy as np

from hypothesis import strategies as st
from hypothesis import given, settings

from socmodel.state import ZerosState
from socmodel.state import OnesState
from socmodel.state import UniformState

from socmodel.connectivity import ZerosConnectivity
from socmodel.connectivity import OnesConnectivity
from socmodel.connectivity import RandomConnectivity

from socmodel.network import Network


state_initializers = [ZerosState, OnesState, UniformState]
connectivity_initializers = [ZerosConnectivity, OnesConnectivity, RandomConnectivity]



@given(n                 = st.integers(min_value=1, max_value=1e3),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       beta              = st.floats(min_value=0.),
       W                 = st.integers(min_value=1, max_value=1e4),
       T                 = st.integers(min_value=1),)
def test_constructor (n, state_init, connectivity_init, beta, W, T):

  params = {'n'                 : n,
            'state_init'        : state_init(),
            'connectivity_init' : connectivity_init(),
            'beta'              : beta,
            'W'                 : W,
            'T'                 : T,}
  Network(**params)



@given(n                 = st.integers(min_value=1, max_value=100),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       beta              = st.floats(min_value=0.),
       W                 = st.integers(min_value=1, max_value=100),
       T                 = st.integers(min_value=1, max_value=100),
       steps             = st.integers(min_value=0, max_value=100),)
@settings(deadline=None)
def test_state_evolution (n, state_init, connectivity_init, beta, W, T, steps):

  net = Network(n=n, state_init=state_init(), connectivity_init=connectivity_init(),
                beta=beta, W=W, T=T)

  for _ in range(steps):

    numActive = net._evolve_state()

    assert ((net.sigma == 0) | (net.sigma == 1)).all()
    assert (net.history[-1] == net.sigma).all()
    assert (numActive / net.T) <= net.n



@given(n                 = st.integers(min_value=1, max_value=100),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       beta              = st.floats(min_value=0.),
       W                 = st.integers(min_value=1, max_value=100),
       T                 = st.integers(min_value=1, max_value=100),
       steps             = st.integers(min_value=0, max_value=100),)
@settings(deadline=None)
def test_connectivity_evolution (n, state_init, connectivity_init, beta, W, T, steps):

  net = Network(n=n, state_init=state_init(), connectivity_init=connectivity_init(),
                beta=beta, W=W, T=T)

  for _ in range(steps):

    net._evolve_connectivity()

    assert ((net.C == -1) | (net.C == 0) | (net.C == 1)).all()
    assert (net.C[np.eye(n, dtype=bool)] == 0).all()
    assert np.sum(net.C == 1) == net.linksPlus
    assert np.sum(net.C == -1) == net.linksMinus
