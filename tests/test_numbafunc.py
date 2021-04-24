import numpy as np
from scipy import sparse

from hypothesis import strategies as st
from hypothesis import given, settings

from socmodel.source.numbafunc import compute_signal as nb_compute_signal
from socmodel.source.numbafunc import update_state as nb_update_state
from socmodel.source.numbafunc import update_average_activity as nb_update_average_activity

py_compute_signal = nb_compute_signal.py_func
py_update_state = nb_update_state.py_func
py_update_average_activity = nb_update_average_activity.py_func


@given(n     = st.integers(min_value=1, max_value=100),
       scale = st.integers(min_value=-1000, max_value=1000),
       shift = st.integers(min_value=-1000, max_value=1000),)
@settings(deadline=None)
def test_compute_signal (n, scale, shift):

  sigma = (np.random.rand(n) * scale + shift).astype(np.int32)
  np_C = (np.random.rand(n,n) * scale + shift).astype(np.int32)

  C_sparse = sparse.coo_matrix(np_C)
  C = C_sparse.data
  i = C_sparse.row
  j = C_sparse.col

  np_result = np.dot(np_C, sigma)
  nb_result = nb_compute_signal(n=n, sigma=sigma, C=C, i=i, j=j)
  py_result = py_compute_signal(n=n, sigma=sigma, C=C, i=i, j=j)

  assert ((np_result == nb_result) & (nb_result == py_result)).all()


@given(n     = st.integers(min_value=1, max_value=100),
       scale = st.integers(min_value=-1000, max_value=1000),
       shift = st.integers(min_value=-1000, max_value=1000),)
@settings(deadline=None)
def test_update_state (n, scale, shift):

  signal = (np.random.rand(n) * scale + shift).astype(np.int32)

  nb_sigma, nb_numActive = nb_update_state(n=n, beta=np.inf, signal=signal, numActive=0)
  py_sigma, py_numActive = py_update_state(n=n, beta=np.inf, signal=signal, numActive=0)

  assert (nb_sigma == py_sigma).all()
  assert np.sum(nb_sigma) == nb_numActive
  assert nb_numActive == py_numActive
  assert np.sum(py_sigma) == py_numActive


@given(n     = st.integers(min_value=1, max_value=100),
       p     = st.floats(min_value=0., max_value=1.),
       alpha = st.floats(min_value=0., max_value=1.),)
@settings(deadline=None)
def test_update_average_activity (n, p, alpha):

  sigma = np.where(np.random.rand(n) < p, 1, 0).astype(np.int8)
  avgActivity = np.random.rand(n).astype(np.float32)

  nb_avgActivity = nb_update_average_activity(n=n, sigma=sigma, alpha=alpha, avgActivity=avgActivity)
  py_avgActivity = py_update_average_activity(n=n, sigma=sigma, alpha=alpha, avgActivity=avgActivity)

  assert (nb_avgActivity == py_avgActivity).all()
