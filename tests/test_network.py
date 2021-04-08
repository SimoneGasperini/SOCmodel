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
       T                 = st.integers(min_value=1),
       seed              = st.integers(min_value=1, max_value=2**32 - 1),)
def test_constructor (n, state_init, connectivity_init, beta, W, T, seed):

  params = {'n'                 : n,
            'state_init'        : state_init(),
            'connectivity_init' : connectivity_init(),
            'beta'              : beta,
            'W'                 : W,
            'T'                 : T,
            'seed'              : seed
            }
  network = Network(**params)


@given(n                 = st.integers(min_value=1, max_value=100),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       beta              = st.floats(min_value=0.),)
@settings(deadline=None)
def test_probability (n, state_init, connectivity_init, beta):

  network = Network(n=n,
                    state_init=state_init(),
                    connectivity_init=connectivity_init(),
                    beta=beta)

  probability = network._compute_probability(n=network.n, sigma=network.sigma,
                                             C=network.C, beta=network.beta)
  assert probability.size == n
  assert ((0. <= probability) & (probability <= 1.)).all()

  if beta == 0.:
    assert (probability == 0.5).all()


@given(n                 = st.integers(min_value=1, max_value=100),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       beta              = st.floats(min_value=0.),
       W                 = st.integers(min_value=1, max_value=10),
       steps             = st.integers(min_value=0, max_value=10),)
@settings(deadline=None)
def test_statevector (n, state_init, connectivity_init, beta, W, steps):

  network = Network(n=n,
                    state_init=state_init(),
                    connectivity_init=connectivity_init(),
                    beta=beta,
                    W=W)

  check_state = lambda s : ((s == 0) | (s == 1)).all()
  assert check_state(network.sigma)

  for _ in range(steps):

    network._update_state()
    assert check_state(network.sigma)


@given(n                 = st.integers(min_value=1, max_value=100),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       beta              = st.floats(min_value=0.),
       W                 = st.integers(min_value=1, max_value=10),
       steps             = st.integers(min_value=0, max_value=10),)
@settings(deadline=None)
def test_rewiring (n, state_init, connectivity_init, beta, W, steps):

  network = Network(n=n,
                    state_init=state_init(),
                    connectivity_init=connectivity_init(),
                    beta=beta,
                    W=W)

  def check_connectivity (C):
    c1 = ((C == -1) | (C == 0) | (C == 1)).all()
    c2 = (C[np.eye(n, dtype=bool)] == 0).all()
    return c1 and c2

  for _ in range(steps):

    network._update_state()

    C1 = np.copy(network.C)
    network._perform_rewiring()
    C2 = np.copy(network.C)

    assert network.linksPlus == np.sum(network.C == 1)
    assert network.linksMinus == np.sum(network.C == -1)

    assert check_connectivity(C1)
    assert check_connectivity(C2)
    assert np.sum(C2 - C1) in [-1,0,1]
