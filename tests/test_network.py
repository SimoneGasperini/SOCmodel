import numpy as np

from hypothesis import strategies as st
from hypothesis import given, settings

from socmodel.source.state import ZerosState
from socmodel.source.state import OnesState
from socmodel.source.state import RandomState

from socmodel.source.connectivity import ZerosConnectivity
from socmodel.source.connectivity import OnesConnectivity
from socmodel.source.connectivity import RandomConnectivity

from socmodel.source.network import Network


sigma_initializers = [ZerosState, OnesState, RandomState]
C_initializers = [ZerosConnectivity, OnesConnectivity, RandomConnectivity]



@given(n          = st.integers(min_value=1, max_value=1000),
       alpha      = st.floats(min_value=0., max_value=1.),
       beta       = st.floats(min_value=0.),
       tau        = st.integers(min_value=1),
       sigma_init = st.sampled_from(sigma_initializers),
       C_init     = st.sampled_from(C_initializers),)
def test_constructor (n, alpha, beta, tau, sigma_init, C_init):

  Network(n=n, alpha=alpha, beta=beta, tau=tau,
          sigma_init=sigma_init(), C_init=C_init())



@given(n          = st.integers(min_value=1, max_value=1000),
       alpha      = st.floats(min_value=0., max_value=1.),
       beta       = st.floats(min_value=0.),
       tau        = st.integers(min_value=1, max_value=10),
       sigma_init = st.sampled_from(sigma_initializers),
       C_init     = st.sampled_from(C_initializers),
       steps      = st.integers(min_value=0, max_value=100),)
@settings(deadline=None)
def test_state_evolution (n, alpha, beta, tau, sigma_init, C_init, steps):

  net = Network(n=n, alpha=alpha, beta=beta, tau=tau,
                sigma_init=sigma_init(), C_init=C_init())

  for _ in range(steps):

    numActive = net._evolve_state()

    assert ((net.sigma == 0) | (net.sigma == 1)).all()
    assert ((net.avgActivity >= 0.) & (net.avgActivity <= 1.)).all()
    assert (numActive / net.tau) <= net.n



@given(n          = st.integers(min_value=1, max_value=1000),
       alpha      = st.floats(min_value=0., max_value=1.),
       beta       = st.floats(min_value=0.),
       tau        = st.integers(min_value=1, max_value=10),
       sigma_init = st.sampled_from(sigma_initializers),
       C_init     = st.sampled_from(C_initializers),
       steps      = st.integers(min_value=0, max_value=50),)
@settings(deadline=None)
def test_connectivity_evolution (n, alpha, beta, tau, sigma_init, C_init, steps):

  net = Network(n=n, alpha=alpha, beta=beta, tau=tau,
                sigma_init=sigma_init(), C_init=C_init())
  net.C = net.C.tocsr()

  for _ in range(steps):

    net._evolve_connectivity()

    C = net.C.toarray()
    assert ((C == -1) | (C == 0) | (C == 1)).all()
    assert (C[np.eye(n, dtype=bool)] == 0).all()
    assert np.sum(C == 1) == net.linksPlus
    assert np.sum(C == -1) == net.linksMinus
