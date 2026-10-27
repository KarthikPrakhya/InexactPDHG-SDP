"""
Common utilities for Lanczos-based spectrahedron projections.

This module provides shared CPU/GPU infrastructure for:

    - SciPy / CuPy backend detection
    - backend-specific eigsh
    - LinearOperator wrappers
    - matrix-vector application
    - recycled starting vectors
    - Ritz residual computation
    - simplex projection / waterfilling
    - optional matvec counting

Both:
    accurate_lanczos.py
    spectral_warm_start.py

should use these utilities so that the two algorithms differ only
in their algorithmic policy, not their backend implementation.

Supported backends
------------------
CPU:
    NumPy + SciPy

GPU:
    CuPy + CuPyX

No array is intentionally transferred between CPU and GPU.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Optional SciPy backend
# ============================================================================

try:
    from scipy.sparse.linalg import (
        eigsh as scipy_eigsh,
        LinearOperator as ScipyLinearOperator,
    )

    HAS_SCIPY = True

except ImportError:
    scipy_eigsh = None
    ScipyLinearOperator = None
    HAS_SCIPY = False


# ============================================================================
# Optional CuPy / CuPyX backend
# ============================================================================

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import (
        eigsh as cupyx_eigsh,
        LinearOperator as CupyxLinearOperator,
    )

    HAS_CUPY = True

except ImportError:
    cp = None
    cupyx_eigsh = None
    CupyxLinearOperator = None
    HAS_CUPY = False


# ============================================================================
# Backend detection
# ============================================================================

def is_cupy_array(x) -> bool:
    """
    Return True if x is a CuPy ndarray.
    """
    return (
        HAS_CUPY
        and isinstance(x, cp.ndarray)
    )


def is_cupy_object(x) -> bool:
    """
    Detect CuPy / CuPyX objects.

    This includes:
        - cupy.ndarray
        - cupyx LinearOperator
        - CuPy sparse objects
        - other objects implemented in cupy/cupyx modules

    This function does not perform any host/device transfer.
    """
    if not HAS_CUPY or x is None:
        return False

    if isinstance(x, cp.ndarray):
        return True

    module = type(x).__module__

    return (
        module.startswith("cupy")
        or module.startswith("cupyx")
    )


def detect_backend(G, start_value=None) -> str:
    """
    Detect the numerical backend from the actual inputs.

    Priority
    --------
    1. G
    2. start_value = (U, alpha)
    3. NumPy

    Returns
    -------
    "cupy" or "numpy"
    """

    if is_cupy_object(G):
        return "cupy"

    if start_value is not None:
        try:
            U = start_value[0]

            if is_cupy_object(U):
                return "cupy"

        except Exception:
            pass

    return "numpy"


def get_xp(backend):
    """
    Return the array module corresponding to a backend.

    Returns
    -------
    numpy module or cupy module
    """

    if backend == "cupy":

        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy/CuPyX backend requested, "
                "but CuPy could not be imported."
            )

        return cp

    if backend == "numpy":

        if not HAS_SCIPY:
            raise RuntimeError(
                "NumPy backend requested, "
                "but SciPy could not be imported."
            )

        return np

    raise ValueError(
        f"Unknown backend '{backend}'. "
        "Expected 'numpy' or 'cupy'."
    )


def get_eigsh(backend):
    """
    Return the backend-specific eigsh implementation.
    """

    if backend == "cupy":

        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy/CuPyX backend requested, "
                "but CuPy is unavailable."
            )

        return cupyx_eigsh

    if backend == "numpy":

        if not HAS_SCIPY:
            raise RuntimeError(
                "SciPy backend requested, "
                "but SciPy is unavailable."
            )

        return scipy_eigsh

    raise ValueError(
        f"Unknown backend '{backend}'."
    )


def get_linear_operator_class(backend):
    """
    Return the backend-specific LinearOperator class.
    """

    if backend == "cupy":

        if not HAS_CUPY:
            raise RuntimeError(
                "CuPy/CuPyX backend requested, "
                "but CuPy is unavailable."
            )

        return CupyxLinearOperator

    if backend == "numpy":

        if not HAS_SCIPY:
            raise RuntimeError(
                "SciPy backend requested, "
                "but SciPy is unavailable."
            )

        return ScipyLinearOperator

    raise ValueError(
        f"Unknown backend '{backend}'."
    )


# ============================================================================
# Matrix-vector interface
# ============================================================================

def matvec(G, x):
    """
    Apply G to x.

    Supports:
        - SciPy LinearOperator
        - CuPy/CuPyX LinearOperator
        - sparse matrices
        - dense matrices
        - objects implementing @

    The returned array remains on the same backend as x/G.
    """

    if hasattr(G, "matvec"):
        return G.matvec(x)

    if hasattr(G, "dot"):
        return G.dot(x)

    return G @ x


# ============================================================================
# Counted LinearOperator
# ============================================================================

def make_counted_operator(
    G,
    backend,
    dtype,
    matvec_counter,
):
    """
    Wrap G in a backend-specific LinearOperator that counts
    matrix-vector products.

    Parameters
    ----------
    G :
        Original matrix/operator.

    backend :
        "numpy" or "cupy".

    dtype :
        Operator dtype.

    matvec_counter :
        Mutable one-element list, e.g.

            counter = [0]

        The counter is incremented in-place.

    Notes
    -----
    G is assumed symmetric, so rmatvec uses the same operation.
    """

    def _matvec(x):

        matvec_counter[0] += 1

        return matvec(G, x)

    def _rmatvec(x):

        # G is assumed symmetric.
        matvec_counter[0] += 1

        return matvec(G, x)

    LinearOperator = get_linear_operator_class(backend)

    return LinearOperator(
        G.shape,
        matvec=_matvec,
        rmatvec=_rmatvec,
        dtype=dtype,
    )


# ============================================================================
# Starting vector / warm-start utilities
# ============================================================================

def get_initial_v0(start_value, xp):
    """
    Extract a normalized dominant vector from

        start_value = (U_prev, alpha_prev)

    Parameters
    ----------
    start_value : tuple or None
        Factorized previous iterate.

    xp :
        NumPy or CuPy module.

    Returns
    -------
    v0 : array or None

    Notes
    -----
    Only U_prev[:, 0] is used.

    No host/device transfer is performed.
    """

    if start_value is None:
        return None

    try:
        U_prev = start_value[0]

        if U_prev is None:
            return None

        if U_prev.ndim != 2:
            return None

        if U_prev.shape[1] == 0:
            return None

        v0 = U_prev[:, 0].copy()

        norm_v0 = xp.linalg.norm(v0)

        # Scalar extraction only.
        if float(norm_v0) == 0.0:
            return None

        v0 = v0 / norm_v0

        return v0

    except Exception:
        return None


def random_starting_vector(n, xp, dtype=None, seed=None):
    """
    Generate a normalized random starting vector on the
    selected backend.

    The resulting array is created directly on the backend.

    Parameters
    ----------
    n : int
        Dimension.

    xp :
        NumPy or CuPy.

    dtype :
        Optional dtype.

    seed :
        Optional random seed.

    Returns
    -------
    v0 : backend-native array
    """

    if xp is np:

        rng = np.random.default_rng(seed)

        if dtype is None:
            dtype = np.float64

        v0 = rng.standard_normal(
            n
        ).astype(dtype, copy=False)

    else:

        if seed is not None:
            # This operates on the CuPy RNG state.
            xp.random.seed(seed)

        if dtype is None:
            dtype = xp.float64

        v0 = xp.random.standard_normal(
            n,
            dtype=dtype,
        )

    norm_v0 = xp.linalg.norm(v0)

    return v0 / norm_v0


def get_or_create_initial_v0(
    start_value,
    n,
    xp,
    dtype=None,
    seed=None,
):
    """
    Return the recycled starting vector if available;
    otherwise generate a backend-native random vector.
    """

    v0 = get_initial_v0(
        start_value,
        xp,
    )

    if v0 is not None:
        return v0

    return random_starting_vector(
        n=n,
        xp=xp,
        dtype=dtype,
        seed=seed,
    )


# ============================================================================
# Ritz residuals
# ============================================================================

def compute_eig_residuals(
    G,
    U,
    eigvals,
    xp,
    relative=False,
):
    """
    Compute Ritz eigenpair residuals.

    Absolute residual:

        ||G u_i - lambda_i u_i||_2

    If relative=True:

        ||G u_i - lambda_i u_i||_2
        --------------------------------
        max(1, |lambda_i|)

    Parameters
    ----------
    G :
        Symmetric matrix/operator.

    U :
        Eigenvectors, shape (n, k).

    eigvals :
        Eigenvalues, shape (k,).

    xp :
        NumPy or CuPy.

    relative :
        Whether to return relative residuals.

    Returns
    -------
    residuals :
        Backend-native array of shape (k,).
    """

    m = eigvals.shape[0]

    residuals = xp.empty(
        m,
        dtype=xp.float64,
    )

    for i in range(m):

        ui = U[:, i]

        r = (
            matvec(G, ui)
            - eigvals[i] * ui
        )

        residual = xp.linalg.norm(r)

        if relative:
            residual /= max(
                1.0,
                abs(float(eigvals[i])),
            )

        residuals[i] = residual

    return residuals

# ============================================================================
# Boundary residual certificate
# ============================================================================

def boundary_residual_certificate(
    G,
    vals,
    vecs,
    theta,
    rank,
    xp,
    eta=1e-2,
):
    """
    Check whether the two Ritz pairs surrounding the water level
    are sufficiently resolved.

    The criterion is

        ||G u_r - lambda_r u_r||
            <= eta * (lambda_r - theta)

        ||G u_{r+1} - lambda_{r+1} u_{r+1}||
            <= eta * (theta - lambda_{r+1})

    Returns
    -------
    certificate_ok : bool
    certificate : dict
    """

    k = len(vals)

    # We need both lambda_r and lambda_{r+1}.
    if rank <= 0:
        return False, {
            "certificate_ok": False,
            "reason": "no_active_eigenvalue",
        }

    if rank >= k:
        return False, {
            "certificate_ok": False,
            "reason": "insufficient_spectrum",
            "rank": rank,
            "k": k,
        }

    lambda_r = float(vals[rank - 1])
    lambda_r1 = float(vals[rank])

    u_r = vecs[:, rank - 1]
    u_r1 = vecs[:, rank]

    # Exactly TWO additional G matvecs.
    r_r = G @ u_r - lambda_r * u_r
    r_r1 = G @ u_r1 - lambda_r1 * u_r1

    rho_r = float(xp.linalg.norm(r_r))
    rho_r1 = float(xp.linalg.norm(r_r1))

    margin_r = lambda_r - float(theta)
    margin_r1 = float(theta) - lambda_r1

    if margin_r <= 0 or margin_r1 <= 0:
        return False, {
            "certificate_ok": False,
            "reason": "invalid_boundary_margin",
            "lambda_r": lambda_r,
            "lambda_r1": lambda_r1,
            "theta": float(theta),
            "rho_r": rho_r,
            "rho_r1": rho_r1,
        }

    ratio_r = rho_r / margin_r
    ratio_r1 = rho_r1 / margin_r1

    upper_ok = ratio_r <= eta
    lower_ok = ratio_r1 <= eta

    certificate_ok = upper_ok and lower_ok

    certificate = {
        "certificate_ok": certificate_ok,
        "lambda_r": lambda_r,
        "lambda_r1": lambda_r1,
        "theta": float(theta),
        "rho_r": rho_r,
        "rho_r1": rho_r1,
        "margin_r": margin_r,
        "margin_r1": margin_r1,
        "ratio_r": ratio_r,
        "ratio_r1": ratio_r1,
        "eta": eta,
        "upper_ok": upper_ok,
        "lower_ok": lower_ok,
    }

    return certificate_ok, certificate


# ============================================================================
# Waterfilling / simplex projection
# ============================================================================

def waterfill(
    lam,
    radius,
    xp,
    active_tol=1e-12,
):
    """
    Project eigenvalues onto the nonnegative simplex.

    Given eigenvalues lambda, compute

        alpha_i = max(lambda_i - theta, 0)

    such that

        sum_i alpha_i = radius.

    Parameters
    ----------
    lam : array
        Eigenvalues.

    radius : float
        Trace constraint.

    xp :
        NumPy or CuPy.

    active_tol : float
        Threshold used to construct the active mask.

    Returns
    -------
    theta :
        Water level. Backend scalar.

    alpha :
        Waterfilled eigenvalues. Same backend as lam.

    active :
        Boolean backend-native mask.
    """

    if lam.size == 0:
        raise ValueError(
            "Cannot waterfill an empty eigenvalue array."
        )

    # Work on the same device/backend.
    lam_sorted = xp.sort(lam)[::-1]

    # Cumulative sums.
    cssv = xp.cumsum(lam_sorted) - radius

    # Avoid constructing anything on CPU.
    ind = xp.arange(
        1,
        lam_sorted.size + 1,
        dtype=lam_sorted.dtype,
    )

    cond = (
        lam_sorted
        - cssv / ind
        > 0
    )

    if not bool(xp.any(cond)):

        theta = (
            lam_sorted[0]
            - radius
        )

    else:

        rho = int(
            xp.nonzero(cond)[0][-1]
        ) + 1

        theta = (
            cssv[rho - 1]
            / rho
        )

    alpha = xp.maximum(
        lam_sorted - theta,
        0,
    )

    active = (
        alpha > active_tol
    )

    return theta, alpha, active


# ============================================================================
# Boundary / waterfilling completeness
# ============================================================================

def waterfill_boundary_ratio(
    alpha,
    xp,
):
    """
    Compute the relative size of the last computed coefficient:

        alpha[-1] / max(1, |alpha[0]|)

    This is useful for determining whether the computed partial
    spectrum extends sufficiently below the water level.

    Returns a Python float because this value is normally used
    for control flow/logging.
    """

    alpha_k = float(alpha[-1])
    alpha_1 = float(alpha[0])

    return (
        alpha_k
        / max(1.0, abs(alpha_1))
    )


def is_waterfill_complete(
    alpha,
    xp,
    boundary_tol,
):
    """
    Determine whether the computed spectrum appears sufficient
    for waterfilling.

    No array transfer is performed.
    """

    ratio = waterfill_boundary_ratio(
        alpha,
        xp,
    )

    return ratio <= boundary_tol


# ============================================================================
# Spectral sorting
# ============================================================================

def sort_eigenpairs(
    eigvals,
    eigvecs,
    xp,
    descending=True,
):
    """
    Sort eigenpairs without transferring arrays.

    Parameters
    ----------
    eigvals :
        Backend-native eigenvalues.

    eigvecs :
        Backend-native eigenvectors.

    xp :
        NumPy or CuPy.

    descending :
        Sort largest first if True.

    Returns
    -------
    eigvals_sorted, eigvecs_sorted
    """

    if descending:
        idx = xp.argsort(eigvals)[::-1]
    else:
        idx = xp.argsort(eigvals)

    return (
        eigvals[idx],
        eigvecs[:, idx],
    )


# ============================================================================
# Simple spectral statistics
# ============================================================================

def active_rank(
    alpha,
    xp,
    active_tol=1e-12,
):
    """
    Return the number of active waterfilled coefficients.

    Only the integer scalar is transferred to host memory.
    """

    active = alpha > active_tol

    return int(
        xp.sum(active)
    )


def trace_error(
    alpha,
    radius,
    xp,
):
    """
    Compute absolute trace error.

    Only the final scalar is transferred to host memory.
    """

    return abs(
        float(xp.sum(alpha))
        - radius
    )


# ============================================================================
# Validation helpers
# ============================================================================

def validate_backend_consistency(
    G,
    start_value,
    backend,
):
    """
    Check that G and start_value are compatible with backend.

    This performs no array transfer.
    """

    if backend == "numpy":

        if is_cupy_object(G):
            raise TypeError(
                "G is a CuPy/CuPyX object but "
                "backend='numpy'."
            )

        if start_value is not None:
            U = start_value[0]

            if is_cupy_object(U):
                raise TypeError(
                    "start_value contains CuPy arrays but "
                    "backend='numpy'."
                )

    elif backend == "cupy":

        if not is_cupy_object(G):
            raise TypeError(
                "backend='cupy' requires G to be a "
                "CuPy/CuPyX object."
            )

        if start_value is not None:
            U = start_value[0]

            if U is not None and not is_cupy_object(U):
                raise TypeError(
                    "backend='cupy' requires U_prev to "
                    "be a CuPy array."
                )

    else:
        raise ValueError(
            f"Unknown backend '{backend}'."
        )