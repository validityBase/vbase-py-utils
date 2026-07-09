"""Pure-numpy Huber M-estimator (RLM) fit, bit-faithful to statsmodels.

This module deliberately imports *only* numpy. It reproduces
``statsmodels.RLM(y, X, M=HuberT()).fit()`` (defaults: ``scale_est='mad'``,
``cov='H1'``, ``update_scale=True``, ``conv='dev'``, ``tol=1e-8``,
``maxiter=50``) so a joblib worker that fits with this function does not pull the
statsmodels import floor (see the parallel betas memory profiling notes).

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
"""

import numpy as np
from numpy.typing import NDArray

# HuberT default tuning constant (statsmodels norms.HuberT().t).
_HUBER_T = 1.345
# Gaussian MAD normalization constant: scipy.stats.norm.ppf(0.75).
_MAD_C = 0.6744897501960817


def _huber_rho(z: NDArray[np.floating], t: float) -> NDArray[np.floating]:
    """Huber's t objective rho(z): 0.5 z^2 for |z|<=t, else |z| t - 0.5 t^2."""
    absz = np.abs(z)
    inner = absz <= t
    return np.where(inner, 0.5 * z**2, absz * t - 0.5 * t**2)


def _huber_weights(z: NDArray[np.floating], t: float) -> NDArray[np.floating]:
    """Huber's t IRLS weights: 1 for |z|<=t, else t/|z|."""
    absz = np.abs(z)
    # Avoid division warnings at z==0; those entries are <= t so weight is 1.
    safe = np.where(absz <= t, 1.0, absz)
    return np.where(absz <= t, 1.0, t / safe)


def _mad_scale(resid: NDArray[np.floating]) -> float:
    """Uncentered MAD scale: median(|resid|) / MAD_C (statsmodels center=0)."""
    return float(np.median(np.abs(resid)) / _MAD_C)


def fit_huber_rlm_params(
    endog: NDArray[np.floating],
    exog: NDArray[np.floating],
    *,
    t: float = _HUBER_T,
    tol: float = 1e-8,
    maxiter: int = 50,
) -> NDArray[np.floating]:
    """Fit a Huber-t robust linear model, returning the coefficient vector.

    Bit-faithful reproduction of ``statsmodels.RLM(endog, exog,
    M=HuberT()).fit().params``. ``endog``/``exog`` must already be
    NaN-filtered and (as in the betas code) exponentially time-weighted with the
    constant column appended; this function only runs the IRLS.

    Args:
        endog: 1-d dependent array, shape (n,).
        exog: 2-d design matrix, shape (n, p) (includes the constant column).
        t: Huber tuning constant. Defaults to 1.345.
        tol: Convergence tolerance on the deviance. Defaults to 1e-8.
        maxiter: Maximum IRLS iterations. Defaults to 50.

    Returns:
        Estimated parameters, shape (p,).
    """
    y = np.asarray(endog, dtype=np.float64)
    x = np.asarray(exog, dtype=np.float64)
    n, p = x.shape
    df_resid = n - p

    def _wls_deviance(resid: NDArray[np.floating], w: NDArray[np.floating]) -> float:
        # Deviance uses the ordinary WLS residual scale of the *current* fit,
        # matching statsmodels _MinimalWLS.results().scale, not the MAD scale.
        wresid = np.sqrt(w) * resid
        wls_scale = float(wresid @ wresid) / df_resid
        return float(_huber_rho(resid / wls_scale, t).sum())

    # Initial fit: WLS with unit weights == OLS via pseudo-inverse (SVD).
    beta = np.linalg.pinv(x) @ y
    resid = y - x @ beta
    scale = _mad_scale(resid)

    # iteration == 1 after the initial fit (statsmodels counts init as iter 1).
    dev = _wls_deviance(resid, np.ones(n))
    iteration = 1

    while True:
        if scale == 0.0:
            # Perfect fit of the weighted data: statsmodels warns and stops,
            # keeping the last params.
            break
        w = _huber_weights(resid / scale, t)
        sw = np.sqrt(w)
        beta = np.linalg.pinv(sw[:, None] * x) @ (sw * y)
        resid = y - x @ beta
        scale = _mad_scale(resid)
        dev_new = _wls_deviance(resid, w)
        iteration += 1
        # statsmodels _check_convergence: stop when |d_dev| <= tol or maxiter hit.
        if not (abs(dev_new - dev) > tol and iteration < maxiter):
            break
        dev = dev_new

    return beta
