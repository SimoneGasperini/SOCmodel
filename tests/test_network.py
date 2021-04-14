import numpy as np

from hypothesis import strategies as st
from hypothesis import given, settings

from socmodel.source.state import ZerosState
from socmodel.source.state import OnesState
from socmodel.source.state import UniformState

from socmodel.source.connectivity import ZerosConnectivity
from socmodel.source.connectivity import OnesConnectivity
from socmodel.source.connectivity import RandomConnectivity

from socmodel.source.network import Network


state_initializers = [ZerosState, OnesState, UniformState]
connectivity_initializers = [ZerosConnectivity, OnesConnectivity, RandomConnectivity]



@given(n                 = st.integers(min_value=1, max_value=1000),
       alpha             = st.floats(min_value=0., max_value=1.),
       beta              = st.floats(min_value=0.),
       T                 = st.integers(min_value=1),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),)
def test_constructor (n, alpha, beta, T, state_init, connectivity_init):

  Network(n=n, alpha=alpha, beta=beta, T=T,
          state_init=state_init(), connectivity_init=connectivity_init())



@given(n                 = st.integers(min_value=1, max_value=1000),
       alpha             = st.floats(min_value=0., max_value=1.),
       beta              = st.floats(min_value=0.),
       T                 = st.integers(min_value=1, max_value=10),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       steps             = st.integers(min_value=0, max_value=100),)
@settings(deadline=None)
def test_state_evolution (n, alpha, beta, T, state_init, connectivity_init, steps):

  net = Network(n=n, alpha=alpha, beta=beta, T=T,
                state_init=state_init(), connectivity_init=connectivity_init())

  for _ in range(steps):

    numActive = net._evolve_state()

    assert ((net.sigma == 0) | (net.sigma == 1)).all()
    assert ((net.avgActivity >= 0.) & (net.avgActivity <= 1.)).all()
    assert (numActive / net.T) <= net.n



@given(n                 = st.integers(min_value=1, max_value=1000),
       alpha             = st.floats(min_value=0., max_value=1.),
       beta              = st.floats(min_value=0.),
       T                 = st.integers(min_value=1, max_value=10),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       steps             = st.integers(min_value=0, max_value=100),)
@settings(deadline=None)
def test_connectivity_evolution (n, alpha, beta, T, state_init, connectivity_init, steps):

  net = Network(n=n, alpha=alpha, beta=beta, T=T,
                state_init=state_init(), connectivity_init=connectivity_init())

  for _ in range(steps):

    net._evolve_connectivity()

    assert ((net.C == -1) | (net.C == 0) | (net.C == 1)).all()
    assert (net.C[np.eye(n, dtype=bool)] == 0).all()
    assert np.sum(net.C == 1) == net.linksPlus
    assert np.sum(net.C == -1) == net.linksMinus
