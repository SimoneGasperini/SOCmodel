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
       beta              = st.floats(min_value=0.),
       W                 = st.integers(min_value=1, max_value=10),
       steps             = st.integers(min_value=0, max_value=100),)
@settings(deadline=None)
def test_activity_dynamics (n, state_init, connectivity_init, beta, W, steps):

  network = Network(n=n, state_init=state_init(),
                    connectivity_init=connectivity_init(),
                    beta=beta, W=W)

  check_state = lambda net : ((net.sigma == 0) | (net.sigma == 1)).all()

  assert check_state(network)

  for _ in range(steps):

    network.sigma, network.descendants = \
      network._compute_new_state(n=network.n, sigma=network.sigma,
                                 C=network.C, beta=network.beta)

    assert check_state(network)
    assert np.sum(network.sigma) == network.descendants


@given(n                 = st.integers(min_value=1, max_value=100),
       state_init        = st.sampled_from(state_initializers),
       connectivity_init = st.sampled_from(connectivity_initializers),
       beta              = st.floats(min_value=0.),
       W                 = st.integers(min_value=1, max_value=10),
       steps             = st.integers(min_value=0, max_value=50),)
@settings(deadline=None)
def test_network_evolution (n, state_init, connectivity_init, beta, W, steps):

  network = Network(n=n, state_init=state_init(),
                    connectivity_init=connectivity_init(),
                    beta=beta, W=W)

  def check_connectivity (net):
    c1 = ((net.C == -1) | (net.C == 0) | (net.C == 1)).all()
    c2 = (net.C[np.eye(n, dtype=bool)] == 0).all()
    return c1 and c2

  network.run(steps)
  assert check_connectivity(network)

  for _ in range(steps):

    network._perform_rewiring()

    assert check_connectivity(network)
    assert np.sum(network.C == 1) == network.linksPlus
    assert np.sum(network.C == -1) == network.linksMinus
