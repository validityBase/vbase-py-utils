"""Numba/JIT Huber M-estimator (RLM) fit, bit-faithful to statsmodels.

This module reproduces ``statsmodels.RLM(y, X, M=HuberT()).fit()`` (defaults:
``scale_est='mad'``, ``cov='H1'``, ``update_scale=True``, ``conv='dev'``,
``tol=1e-8``, ``maxiter=50``). The IRLS loop is compiled with ``numba.njit`` so
the per-fit Python/numpy dispatch overhead and per-iteration temporaries are
eliminated; benchmarks show ~2-3.7x per-fit over the previous pure-numpy loop,
biggest on the small designs (few factors) typical of production. The one op
numba cannot improve is ``np.linalg.pinv`` -- it dispatches to the same LAPACK
SVD as numpy -- so the win comes purely from the elementwise glue.

The IRLS loop mirrors statsmodels exactly, verified against the 0.14.4 source:

* init params via ``WLS(y, X).fit()`` with unit weights == ``pinv(X) @ y``;
* scale via ``scale.mad(resid, center=0)`` == ``median(|resid|) / 0.674489...``
  (note: *uncentered*);
* inner solve via ``_MinimalWLS.fit(method='pinv')`` == ``pinv(sqrt(w)*X) @
  (sqrt(w)*y)`` (SVD whitening -- min-norm solution, so it matches statsmodels
  even on near-collinear designs);
* convergence on the deviance ``sum(rho(resid / wls_scale))`` where ``wls_scale``
  is the ordinary WLS residual variance of the current fit (distinct from the MAD
  scale that drives the weights), stopping when the successive-iteration change is
  ``<= tol`` or ``maxiter`` is reached.

Equivalence to statsmodels is asserted by
``tests/stats/test_handrolled_vs_statsmodels.py`` (atol 1e-8; measured agreement
~1e-10).
"""

import logging
import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# HuberT default tuning constant (statsmodels norms.HuberT().t).
_HUBER_T = 1.345
# Gaussian MAD normalization constant: scipy.stats.norm.ppf(0.75).
_MAD_C = 0.6744897501960817


@njit(cache=True, fastmath=False)
def _huber_weights(z, t):
    """Huber's t IRLS weights: 1 for |z|<=t, else t/|z|."""
    w = np.empty(z.size, dtype=np.float64)
    for i in range(z.size):
        a = abs(z[i])
        if a <= t:
            w[i] = 1.0
        else:
            w[i] = t / a
    return w


@njit(cache=True, fastmath=False)
def _mad_scale(resid):
    """Uncentered MAD scale: median(|resid|) / MAD_C (statsmodels center=0)."""
    return np.median(np.abs(resid)) / _MAD_C


@njit(cache=True, fastmath=False)
def _wls_deviance(resid, w, df_resid, t):
    """Deviance sum(rho(resid / wls_scale)) with the current WLS residual scale.

    wls_scale is the ordinary WLS residual variance of the current fit
    (sqrt(w)*resid), distinct from the MAD scale that drives the weights, matching
    statsmodels ``_MinimalWLS.results().scale``. rho(0)=0, so a non-positive scale
    (perfect fit) yields deviance 0; the scale==0 break in the loop governs
    termination, so guarding here does not change the fitted params.
    """
    ss = 0.0
    for i in range(resid.size):
        wr = math.sqrt(w[i]) * resid[i]
        ss += wr * wr
    wls_scale = ss / df_resid
    if not wls_scale > 0.0:
        return 0.0
    s = 0.0
    for i in range(resid.size):
        z = resid[i] / wls_scale
        a = abs(z)
        if a <= t:
            s += 0.5 * z * z
        else:
            s += a * t - 0.5 * t * t
    return s


# The IRLS loop needs a handful of intermediate arrays (residuals, scale,
# weights, deviance); splitting it would obscure the algorithm.
# pylint: disable=too-many-locals
@njit(cache=True, fastmath=False)
def _fit_core(y, x, t, tol, maxiter):
    """njit IRLS core. Returns ``(params, perfect_fit)``.

    ``perfect_fit`` is True when the loop stopped on a zero MAD scale (statsmodels'
    perfect-fit termination); the caller turns that into the log message the njit
    code cannot emit.
    """
    n, p = x.shape
    df_resid = n - p

    # Initial fit: WLS with unit weights == OLS via pseudo-inverse (SVD).
    beta = np.linalg.pinv(x) @ y
    resid = y - x @ beta
    scale = _mad_scale(resid)

    # iteration == 1 after the initial fit (statsmodels counts init as iter 1).
    dev = _wls_deviance(resid, np.ones(n), df_resid, t)
    iteration = 1
    perfect = False

    while True:
        if scale == 0.0:
            # Perfect fit of the weighted data: statsmodels warns and stops,
            # keeping the last params.
            perfect = True
            break
        w = _huber_weights(resid / scale, t)
        # Weighted design sqrt(w)*X and sqrt(w)*y, built with explicit loops so
        # numba fuses them without allocating temporaries each iteration.
        xw = np.empty((n, p), dtype=np.float64)
        swy = np.empty(n, dtype=np.float64)
        for i in range(n):
            sw_i = math.sqrt(w[i])
            swy[i] = sw_i * y[i]
            for k in range(p):
                xw[i, k] = sw_i * x[i, k]
        beta = np.linalg.pinv(xw) @ swy
        resid = y - x @ beta
        scale = _mad_scale(resid)
        dev_new = _wls_deviance(resid, w, df_resid, t)
        iteration += 1
        # statsmodels _check_convergence: stop when |d_dev| <= tol or maxiter hit.
        if not (abs(dev_new - dev) > tol and iteration < maxiter):
            break
        dev = dev_new

    return beta, perfect


# pylint: disable=too-many-arguments
def fit_huber_rlm_params(
    endog: NDArray[np.floating],
    exog: NDArray[np.floating],
    *,
    t: float = _HUBER_T,
    tol: float = 1e-8,
    maxiter: int = 50,
    label: str | None = None,
) -> NDArray[np.floating]:
    """Fit a Huber-t robust linear model, returning the coefficient vector.

    Bit-faithful reproduction of ``statsmodels.RLM(endog, exog,
    M=HuberT()).fit().params``. ``endog``/``exog`` must already be
    NaN-filtered and (as in the betas code) exponentially time-weighted with the
    constant column appended; this function only runs the IRLS.

    The numerics run in the ``numba.njit`` core :func:`_fit_core`; this wrapper
    coerces dtype/contiguity and emits the perfect-fit warning the compiled core
    cannot log.

    Args:
        endog: 1-d dependent array, shape (n,).
        exog: 2-d design matrix, shape (n, p) (includes the constant column).
        t: Huber tuning constant. Defaults to 1.345.
        tol: Convergence tolerance on the deviance. Defaults to 1e-8.
        maxiter: Maximum IRLS iterations. Defaults to 50.
        label: Optional identifier (e.g. asset name) used only in the perfect-fit
            log message.

    Returns:
        Estimated parameters, shape (p,).
    """
    y = np.ascontiguousarray(endog, dtype=np.float64)
    x = np.ascontiguousarray(exog, dtype=np.float64)
    beta, perfect = _fit_core(y, x, float(t), float(tol), int(maxiter))
    if perfect:
        logger.warning("Perfect fit for %s", label if label is not None else "asset")
    return beta
