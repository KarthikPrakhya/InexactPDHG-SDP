"""
Low-accuracy adaptive Lanczos spectral warm start.
The algorithm progressively computes more of the dominant
spectrum until the available eigenvalues are sufficiently below
the water level.
"""

import time

from .lanczos_utils import (
    detect_backend,
    get_xp,
    get_eigsh,
    make_counted_operator,
    get_initial_v0,
    compute_eig_residuals,
    waterfill,
    sort_eigenpairs,
    waterfill_boundary_ratio,
    trace_error,
    boundary_residual_certificate,
)


def spectral_warm_start(
    A,
    n,
    R,
    D=None,
    U_prev=None,
    alpha_prev=None,
    outer_iter_num=0,
    k_init=20,
    k_step=20,
    k_cap=None,
    sws_tol=1e-2,
    boundary_tol=1e-4,
    maxiter=None,
    backend="auto",
    verbose=False,
    timing=False,
    sync_gpu_for_timing=False,
    boundary_certificate=False,
    boundary_eta=1e-2,
):
    """
    Low-accuracy adaptive Lanczos spectral warm start.

    Parameters
    ----------
    A :
        Symmetric projection operator.

    n : int
        Matrix dimension.

    R : float
        Spectrahedron trace/radius.

    D :
        Retained for compatibility with the previous SWS interface.
        Not used by Lanczos SWS.

    U_prev : array, optional
        Previous factorized projection.

    alpha_prev : array, optional
        Previous spectral coefficients.

        Retained as part of the warm-start interface. Lanczos
        currently uses U_prev[:, 0] as its recycled starting
        direction.

    outer_iter_num : int
        PDHG outer iteration.

    k_init : int
        Initial number of eigenpairs.

    k_step : int
        Increase in spectral dimension when waterfilling is
        incomplete.

    k_cap : int, optional
        Maximum number of eigenpairs.

    sws_tol : float
        Low-accuracy Lanczos tolerance.

    boundary_tol : float
        Relative threshold for deciding whether the available
        spectrum is sufficient for the warm start.

    maxiter : int
        Maximum Lanczos iterations per eigsh call.

    backend : {"auto", "numpy", "cupy"}
        Backend selection.

        "auto" detects the backend from A/U_prev.

    verbose : bool
        Print diagnostics.

    timing : bool, optional
        If True, record per-round and total elapsed times.
        Defaults to False.

    sync_gpu_for_timing : bool, optional
        If True and the backend is CuPy, synchronize the current
        CUDA stream before measuring elapsed times. This makes
        timing more accurate but adds GPU synchronization overhead.
        Defaults to False.

    boundary_certificate : bool, optional
        If True, compute and return the boundary residual certificate.
        Defaults to False.

    boundary_eta : float, optional
        Relative boundary-residual threshold.
        Defaults to 1e-2.

    Returns
    -------
    U, alpha, logging_ws
        Factorized warm start:

            X_WS = U diag(alpha) U^T
    """

    measure_timing = bool(timing)
    sync_for_timing = bool(sync_gpu_for_timing)

    start_time = time.perf_counter() if measure_timing else None

    # ==================================================================
    # Construct start_value exactly as used by accurate Lanczos
    # ==================================================================

    start_value = None

    if U_prev is not None:
        start_value = (
            U_prev,
            alpha_prev,
        )

    # ==================================================================
    # Backend
    # ==================================================================

    if backend == "auto":

        backend = detect_backend(
            A,
            start_value,
        )

    elif backend not in ("numpy", "cupy"):

        raise ValueError(
            "backend must be 'auto', 'numpy', or 'cupy'."
        )

    xp = get_xp(backend)
    eigsh_fn = get_eigsh(backend)

    # ==================================================================
    # Dimensions
    # ==================================================================

    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError(
            f"A has shape {A.shape}, expected ({n}, {n})."
        )

    if n < 2:
        raise ValueError(
            "Lanczos warm start requires n >= 2."
        )

    # eigsh requires k < n
    k_cap_effective = min(
        n - 1,
        1000 if k_cap is None else k_cap,
    )

    if k_cap_effective < 1:
        raise ValueError(
            "k_cap must be at least 1."
        )

    k = min(
        k_init,
        k_cap_effective,
    )

    # ==================================================================
    # Recycle previous dominant direction
    # ==================================================================

    v0 = get_initial_v0(
        start_value,
        xp,
    )

    # ==================================================================
    # Counted operator
    # ==================================================================

    matvec_counter = [0]

    counted_A = make_counted_operator(
        G=A,
        backend=backend,
        dtype=A.dtype,
        matvec_counter=matvec_counter,
    )

    # ==================================================================
    # Storage
    # ==================================================================

    final_vals = None
    final_vecs = None
    final_alpha = None
    final_theta = None
    final_active = None
    final_residuals = None

    final_boundary_ratio = None
    waterfill_complete = False

    round_count = 0
    total_lanczos_time = 0.0

    # ==================================================================
    # Adaptive low-accuracy Lanczos
    # ==================================================================

    while True:

        round_count += 1

        if verbose:
            print(
                f"[SWS] backend={backend} "
                f"round={round_count} "
                f"k={k} "
                f"tol={sws_tol:.1e}"
            )

        round_start = time.perf_counter() if measure_timing else None

        # --------------------------------------------------------------
        # Low-accuracy Lanczos
        # --------------------------------------------------------------

        eigsh_kwargs = dict(
            k=k,
            which="LA",
            tol=sws_tol,
            # maxiter=maxiter,
            # ncv=min(
            #     n,
            #     max(2 * k + 20, 40),
            # ),
        )

        if v0 is not None:
            eigsh_kwargs["v0"] = v0

        vals, vecs = eigsh_fn(
            counted_A,
            **eigsh_kwargs,
        )

        # --------------------------------------------------------------
        # Measure elapsed time (optional) with optional GPU sync
        # --------------------------------------------------------------

        lanczos_time = None

        if measure_timing:
            if backend == "cupy" and sync_for_timing:
                xp.cuda.get_current_stream().synchronize()

            lanczos_time = (
                time.perf_counter()
                - round_start
            )

            total_lanczos_time += lanczos_time

        # --------------------------------------------------------------
        # Sort descending
        # --------------------------------------------------------------

        vals, vecs = sort_eigenpairs(
            vals,
            vecs,
            xp,
            descending=True,
        )

        # --------------------------------------------------------------
        # Ritz residuals
        # --------------------------------------------------------------

        # residuals = compute_eig_residuals(
        #     A,
        #     vecs,
        #     vals,
        #     xp,
        #     relative=True,
        # )

        # max_residual = float(
        #     xp.max(residuals)
        # )

        # mean_residual = float(
        #     xp.mean(residuals)
        # )

        # --------------------------------------------------------------
        # Waterfilling
        # --------------------------------------------------------------

        theta, alpha, active = waterfill(
            vals,
            R,
            xp,
            active_tol=1e-12,
        )

        rank = int(
            xp.sum(active)
        )

        # --------------------------------------------------------------
        # Boundary criterion
        # --------------------------------------------------------------

        relative_boundary = (
            waterfill_boundary_ratio(
                alpha,
                xp,
            )
        )

        waterfill_complete = (
            relative_boundary <= boundary_tol
        )

        # --------------------------------------------------------------
        # Save current result
        # --------------------------------------------------------------

        final_vals = vals
        final_vecs = vecs
        final_alpha = alpha
        final_theta = theta
        final_active = active
        # final_residuals = residuals
        final_boundary_ratio = relative_boundary

        if verbose and measure_timing:
            print(
                f"[SWS] rank={rank} "
                f"theta={float(theta):.6e} "
                f"alpha_k={float(alpha[-1]):.3e} "
                f"boundary={relative_boundary:.3e} "
                # f"max_res={max_residual:.3e} "
                f"matvecs={matvec_counter[0]} "
                f"time={lanczos_time:.3f}s"
            )
        elif verbose and not measure_timing:
            print(
                f"[SWS] rank={rank} "
                f"theta={float(theta):.6e} "
                f"alpha_k={float(alpha[-1]):.3e} "
                f"boundary={relative_boundary:.3e} "
                # f"max_res={max_residual:.3e} "
                f"matvecs={matvec_counter[0]} "
            )
        # --------------------------------------------------------------
        # Stop: spectrum is sufficient
        # --------------------------------------------------------------

        if waterfill_complete:

            if verbose:
                print(
                    "[SWS] waterfilling complete."
                )

            break

        # --------------------------------------------------------------
        # Stop: k cap
        # --------------------------------------------------------------

        if k >= k_cap_effective:

            if verbose:
                print(
                    "[SWS] k_cap reached before "
                    "waterfilling completion."
                )

            break

        # --------------------------------------------------------------
        # Increase k
        # --------------------------------------------------------------

        k_new = min(
            k + k_step,
            k_cap_effective,
        )

        if k_new == k:
            break

        k = k_new

        # --------------------------------------------------------------
        # Recycle dominant Ritz vector
        # --------------------------------------------------------------

        v0 = vecs[:, 0].copy()

    # ==================================================================
    # Final active factor
    # ==================================================================

    mask = final_alpha > 1e-12

    U = final_vecs[:, mask]
    alpha = final_alpha[mask]

    rank = int(
        xp.sum(mask)
    )

    # ==================================================================
    # Active residuals
    # ==================================================================

    # active_residuals = (
    #     final_residuals[mask]
    # )

    # if active_residuals.size > 0:

    #     final_max_residual = float(
    #         xp.max(active_residuals)
    #     )

    #     final_mean_residual = float(
    #         xp.mean(active_residuals)
    #     )

    # else:

    #     final_max_residual = float("inf")
    #     final_mean_residual = float("inf")

    # ==================================================================
    # Trace error
    # ==================================================================

    tr_error = trace_error(
        alpha,
        R,
        xp,
    )

    # ==================================================================
    # Total timing (optional)
    # ==================================================================

    total_time = None

    if measure_timing:
        if backend == "cupy" and sync_for_timing:
            xp.cuda.get_current_stream().synchronize()

        total_time = (
            time.perf_counter()
            - start_time
        )

    # ==================================================================
    # Boundary Certificate
    # ==================================================================

    boundary_certificate_ok = None
    boundary_cert = None
    boundary_cert_matvecs = 0

    if boundary_certificate:
        before = matvec_counter[0]

        boundary_certificate_ok, boundary_cert = (
            boundary_residual_certificate(
                G=counted_A,
                vals=final_vals,
                vecs=final_vecs,
                theta=final_theta,
                rank=rank,
                xp=xp,
                eta=boundary_eta,
            )
        )

        boundary_cert_matvecs = (
            matvec_counter[0] - before
        )

    algorithm_matvec_count = (
        matvec_counter[0] - boundary_cert_matvecs
    )

    # ==================================================================
    # Logging
    # ==================================================================

    logging_ws = {
        "backend": backend,
        "method": "spectral_warm_start_lanczos",

        "outer_iter_num": outer_iter_num,

        "sws_tol": sws_tol,
        "maxiter": maxiter,

        "k": len(final_vals),
        "k_cap": k_cap_effective,
        "k_init": k_init,
        "k_step": k_step,

        "rank_ws": rank,
        "theta": float(final_theta),

        "trace_error": tr_error,

        # "max_eig_residual": final_max_residual,
        # "mean_eig_residual": final_mean_residual,

        "waterfill_complete": waterfill_complete,

        "alpha_k": float(final_alpha[-1]),
        "relative_boundary": final_boundary_ratio,
        "boundary_tol": boundary_tol,

        "matvec_count": algorithm_matvec_count,
        "lanczos_rounds": round_count,
        "lanczos_time": total_lanczos_time if measure_timing else None,
        "total_time": total_time,
        "boundary_certificate_matvecs": boundary_cert_matvecs,
        "boundary_certificate_ok": boundary_certificate_ok,
        "boundary_certificate": boundary_cert,
    }

    if verbose:
        print()
        print("[SWS] ========================================")
        print("[SWS] Spectral warm start complete")
        print("[SWS] ========================================")
        print(f"[SWS] backend       = {backend}")
        print(f"[SWS] k            = {len(final_vals)}")
        print(f"[SWS] rank         = {rank}")
        print(f"[SWS] theta        = {float(final_theta):.6e}")
        # print(
        #     f"[SWS] max residual = "
        #     f"{final_max_residual:.3e}"
        # )
        print(
            f"[SWS] boundary     = "
            f"{final_boundary_ratio:.3e}"
        )
        print(
            f"[SWS] matvecs      = "
            f"{matvec_counter[0]}"
        )
        if measure_timing:
            print(
                f"[SWS] Lanczos time = "
                f"{total_lanczos_time:.3f}s"
            )
            print(
                f"[SWS] total time   = "
                f"{total_time:.3f}s"
            )
        print("[SWS] ========================================")

    return U, alpha, logging_ws




# import time
# import numpy as np
# from algorithms.projections.projection_spectrahedron import projection_spectrahedron

# # ============================================================
# # CPU backends
# # ============================================================

# try:
#     from scipy.sparse.linalg import (
#         lobpcg,
#         eigsh,
#         LinearOperator as ScipyLinearOperator,
#     )
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False

# # ============================================================
# # GPU backends
# # ============================================================

# try:
#     import cupy as cp
#     from cupyx.scipy.sparse.linalg import (
#         lobpcg as cupyx_lobpcg,
#         eigsh as cupyx_eigsh,
#         LinearOperator as CupyxLinearOperator,
#     )
#     HAS_CUPYX = True
# except ImportError:
#     HAS_CUPYX = False

# # ============================================================
# # Utilities
# # ============================================================

# def get_xp(backend="auto"):
#     if backend == "cupyx":
#         if not HAS_CUPYX:
#             raise RuntimeError("CuPy backend requested but unavailable")
#         return cp

#     if backend == "numpy":
#         return np

#     if HAS_CUPYX:
#         return cp

#     return np


# def prepare_X(vecs, k, xp, n):
#     k_eff = min(k, vecs.shape[1])

#     X = vecs[:, :k_eff]

#     if k_eff < k:
#         extra = xp.random.standard_normal((n, k - k_eff))
#         X = xp.concatenate([X, extra], axis=1)

#     return X


# # ============================================================
# # Projection onto simplex
# # ============================================================

# def compute_tau(lam, R, xp):
#     lo = lam.min() - 1
#     hi = lam.max()

#     for _ in range(60):
#         mid = 0.5 * (lo + hi)

#         s = xp.maximum(lam - mid, 0).sum()

#         if s > R:
#             lo = mid
#         else:
#             hi = mid

#     return 0.5 * (lo + hi)


# # ============================================================
# # Regime classification
# # ============================================================

# def classify_regime(rank, n,
#                     low_thresh=0.12,
#                     high_thresh=0.28):

#     ratio = rank / n

#     if ratio <= low_thresh:
#         return "low"

#     elif ratio <= high_thresh:
#         return "medium"

#     return "high"


# # ============================================================
# # Preconditioner
# # ============================================================

# def build_preconditioner(D, xp, n):

#     def matvec(x):
#         x = x.reshape(-1)
#         return D * (D * x)
    
#     def rmatvec(x):
#         x = x.reshape(-1)
#         return matvec(x)

#     def matmat(X):
#         return D[:, None] * (D[:, None] * X)

#     if xp is np:
#         return ScipyLinearOperator(
#             (n, n),
#             matvec=matvec,
#             rmatvec=rmatvec,
#             matmat=matmat,
#         )
#     else:
#         return CupyxLinearOperator(
#             (n, n),
#             matvec=matvec,
#             rmatvec=rmatvec,
#             matmat=matmat,
#         )


# # ============================================================
# # Dense fallback
# # ============================================================

# def dense_projection(A, R):

#     return projection_spectrahedron(A, diameter=R)


# # ============================================================
# # Compute eigenvalue residuals
# # ============================================================
# def compute_eig_residuals(A_mv, U, eigvals, xp):

#     residuals = []

#     for i in range(len(eigvals)):
#         ui = U[:, i]
#         lam = eigvals[i]

#         r = A_mv(ui) - lam * ui

#         rel = xp.linalg.norm(r) / max(1.0, abs(lam))

#         residuals.append(rel)

#     max_eig_residual = max(residuals)
#     mean_eig_residual = sum(residuals) / len(residuals)

#     return max_eig_residual, mean_eig_residual


# # ============================================================
# # Main Spectral Warm Start Routine
# # ============================================================

# def spectral_warm_start(
#     A,
#     n,
#     R,
#     D=None,
#     U_prev=None,
#     alpha_prev=None,
#     outer_iter_num=0,
#     k_init=20,
#     k_step=20,
#     k_cap=None,
#     max_ws_iters=20,
#     eig_tol_low=1e-3, # 3
#     eig_tol_medium=1e-5, # 5
#     eig_tol_high=1e-7, # 7
#     backend="auto",
#     verbose=True,
# ):

#     xp = get_xp(backend)

#     if xp is cp:
#         backend = "cupyx"
#     else:
#         backend = "numpy"

#     if k_cap is None:
#         k_cap = min(n, 1000)

#     # ========================================================
#     # Initial rank guess
#     # ========================================================

#     k = min(k_init, k_cap)
#     rank_est = k

#     # ========================================================
#     # Preconditioner
#     # ========================================================

#     M = None

#     if D is not None:
#         M = build_preconditioner(D, xp, n)

#     # ========================================================
#     # Initial block
#     # ========================================================

#     if U_prev is not None:

#         if U_prev.shape[1] >= k:
#             X = U_prev[:, :k]

#         else:
#             extra = xp.random.standard_normal(
#                 (n, k - U_prev.shape[1])
#             )
#             X = xp.concatenate([U_prev, extra], axis=1)

#     else:
#         X = xp.random.standard_normal((n, k))

#     # ========================================================
#     # SWS loop
#     # ========================================================

#     for fr_iter in range(max_ws_iters):

#         # ====================================================
#         # Determine regime
#         # ====================================================

#         regime = classify_regime(rank_est, n)

#         if verbose:
#             print(
#                 f"[SWS] rank={rank_est} "
#                 f"regime={regime}"
#             )

#         # ----------------------------------------------------
#         # Regime-dependent tolerance
#         # ----------------------------------------------------

#         if regime == "low":
#             eig_tol = eig_tol_low
#             maxiter = 2 # 30

#         elif regime == "medium":
#             eig_tol = eig_tol_medium
#             maxiter = 5 # 80

#         else:
#             eig_tol = eig_tol_high
#             maxiter = 7 # 150

#         if verbose:
#             print(
#                 f"[SWS] iter={fr_iter} "
#                 f"k={k} "
#                 f"regime={regime} "
#                 f"eig_tol={eig_tol}"
#             )

#         # ====================================================
#         # HIGH-RANK REGIME
#         # ====================================================

#         if regime == "high":

#             # ------------------------------------------------
#             # Otherwise:
#             # large partial solve
#             # ------------------------------------------------
#             k_dense = min(n, max(k, int(0.5 * n)))
#             X = prepare_X(X, k_dense, xp, n)
            
#             try: 
#                 if backend == "cupyx":

#                     idx = np.argsort(alpha_prev)

#                     # Sorted eigenvalues
#                     alpha_sorted = alpha_prev[idx]

#                     # Reorder eigenvectors accordingly
#                     U_sorted = U_prev[:, idx]

#                     v0 = U_sorted[:, 0]

#                     vals, vecs = cupyx_eigsh(
#                         A,
#                         k=k,
#                         which="LA",
#                         tol=eig_tol,
#                         v0 = v0,
#                         maxiter=maxiter
#                     )

#                     idx = cp.argsort(-vals)

#                     vals = vals[idx]
#                     vecs = vecs[:, idx]

#                 else:

#                     vals, vecs = eigsh(
#                         A,
#                         k=k,
#                         which="LA",
#                         tol=eig_tol,
#                     )

#                     idx = np.argsort(-vals)

#                     vals = vals[idx]
#                     vecs = vecs[:, idx]
#             except:
#                 vecs, vals = dense_projection(A, R)

#             if np.any(np.isnan(vals)) or np.isany(np.isnan(vecs)):
#                 vecs, vals = dense_projection(A, R)

#             max_eig_residual, mean_eig_residual = compute_eig_residuals(A, vecs, vals, xp)

#             tau = compute_tau(vals, R, xp)

#             alpha = xp.maximum(vals - tau, 0)

#             mask = alpha > 1e-12

#             rank_est = int(mask.sum())

#             logging = {
#                 "fr_iter": fr_iter, 
#                 "rank_ws": rank_est,
#                 "max_eig_residual": max_eig_residual,
#                 "mean_eig_residual": mean_eig_residual,
#                 "theta": tau}

#             if rank_est == 0:
#                 print('0 rank in warm start')
#             return vecs[:, mask], alpha[mask], logging

#         # ====================================================
#         # LOW / MEDIUM PARTIAL EIGENSOLVE
#         # ====================================================

#         elif regime == "medium":
#             try:
#                 if backend == "cupyx":
#                     v0 = X

#                     vals, vecs = cupyx_eigsh(
#                         A,
#                         k=k,
#                         which="LA",
#                         tol=eig_tol,
#                         maxiter=maxiter
#                     )

#                     idx = cp.argsort(-vals)

#                     vals = vals[idx]
#                     vecs = vecs[:, idx]

#                 else:

#                     vals, vecs = eigsh(
#                         A,
#                         k=k,
#                         which="LA",
#                         tol=eig_tol,
#                         maxiter=maxiter
#                     )

#                     idx = np.argsort(-vals)

#                     vals = vals[idx]
#                     vecs = vecs[:, idx]
#             except:
#                 vecs, vals = dense_projection(A, R)

#         else: # Low Regime
#             try:
#                 if backend == "cupyx":

#                     vals, vecs = cupyx_lobpcg(
#                         A,
#                         X,
#                         M=M,
#                         tol=eig_tol,
#                         maxiter=maxiter,
#                         largest=True,
#                     )

#                     idx = cp.argsort(-vals)

#                     vals = vals[idx]
#                     vecs = vecs[:, idx]

#                 else:

#                     vals, vecs = lobpcg(
#                         A,
#                         X,
#                         M=M,
#                         tol=eig_tol,
#                         maxiter=maxiter,
#                         largest=True,
#                     )

#                     idx = np.argsort(-vals)

#                     vals = vals[idx]
#                     vecs = vecs[:, idx]


#             except Exception as e:

#                 if verbose:
#                     print(f"[SWS] LOBPCG failed: {e}")

#                 # fallback eigsh

#                 try: 
#                     if backend == "cupyx":

#                         vals, vecs = cupyx_eigsh(
#                             A,
#                             k=k,
#                             which="LA",
#                             tol=eig_tol,
#                             maxiter=maxiter
#                         )

#                         idx = cp.argsort(-vals)

#                         vals = vals[idx]
#                         vecs = vecs[:, idx]

#                     else:

#                         vals, vecs = eigsh(
#                             A,
#                             k=k,
#                             which="LA",
#                             tol=eig_tol,
#                             maxiter=maxiter
#                         )

#                         idx = np.argsort(-vals)

#                         vals = vals[idx]
#                         vecs = vecs[:, idx]
#                 except Exception as e:

#                     if verbose:
#                         print(f"[SWS] eigsh failed: {e}")

#                     vecs, vals = dense_projection(A, R)
                
#         # ====================================================
#         # Simplex projection
#         # ====================================================

#         max_eig_residual, mean_eig_residual = compute_eig_residuals(A, vecs, vals, xp)

#         tau = compute_tau(vals, R, xp)

#         alpha = xp.maximum(vals - tau, 0)

#         mask = alpha > 1e-12

#         rank_est = int(mask.sum())

#         logging = {
#             "fr_iter": fr_iter, 
#             "rank_ws": rank_est,
#             "max_eig_residual": max_eig_residual,
#             "mean_eig_residual": mean_eig_residual,
#             "theta": tau}
        
#         if rank_est == 0:
#             print('0 rank in warm start')

#         # ====================================================
#         # Mixing / recycling
#         # ====================================================

#         # if (
#         #     U_prev is not None
#         #     and alpha_prev is not None
#         # ):

#         #     w = min(outer_iter_num / 20000.0, 1.0)

#         #     k_mix = min(
#         #         vecs.shape[1],
#         #         U_prev.shape[1],
#         #     )

#         #     vecs[:, :k_mix] = (
#         #         (1 - w) * vecs[:, :k_mix]
#         #         + w * U_prev[:, :k_mix]
#         #     )

#         #     alpha[:k_mix] = (
#         #         (1 - w) * alpha[:k_mix]
#         #         + w * alpha_prev[:k_mix]
#         #     )

#         # ====================================================
#         # Spectral stopping criterion
#         # ====================================================

#         spectral_tol = 1e-8 * alpha[0]

#         # IMPORTANT:
#         # stop when boundary eigenvalue is tiny
#         #
#         # means:
#         # no significant truncated spectrum remains
#         #

#         if alpha[-1] <= spectral_tol:

#             if verbose:
#                 print("[SWS] spectral convergence reached")

#             return vecs[:, mask], alpha[mask], logging

#         # ====================================================
#         # Medium regime:
#         # stop expanding aggressively
#         # ====================================================

#         if regime == "medium":

#             if rank_est >= 0.9 * k:

#                 k = min(k + k_step, k_cap)

#                 X = prepare_X(vecs, k, xp, n)

#                 continue

#             return vecs[:, mask], alpha[mask], logging

#         # ====================================================
#         # Low regime:
#         # continue adaptive growth
#         # ====================================================

#         if rank_est >= 0.9 * k:

#             k_new = min(k + k_step, k_cap)

#             if k_new == k:
#                 return vecs[:, mask], alpha[mask], logging

#             k = k_new

#             X = prepare_X(vecs, k, xp, n)

#         else:

#             return vecs[:, mask], alpha[mask], logging

#     # ========================================================
#     # Max iterations reached
#     # ========================================================

#     return vecs[:, mask], alpha[mask], logging


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# import numpy as np

# # Try SciPy LOBPCG
# try:
#     from scipy.sparse.linalg import lobpcg, LinearOperator as ScipyLinearOperator, eigsh
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False

# # Try CuPy LOBPCG
# try:
#     import cupy as cp
#     from cupyx.scipy.sparse.linalg import lobpcg as cupyx_lobpcg
#     from cupyx.scipy.sparse.linalg import eigsh as cupyx_eigsh
#     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator
#     HAS_CUPYX = True
# except ImportError:
#     HAS_CUPYX = False
# # try:
# #     from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, eigsh as scipy_eigsh
# #     HAS_SCIPY = True
# # except ImportError:
# #     HAS_SCIPY = False

# # try:
# #     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator, eigsh as cupyx_eigsh
# #     import cupy as xp
# #     HAS_CUPYX = True
# # except ImportError:
# #     HAS_CUPYX = False


# def prepare_X(vecs_np, k, xp, n):
#     k_eff = min(k, vecs_np.shape[1])

#     if xp is np:
#         X = vecs_np[:, :k_eff].copy()
#     else:
#         X = cp.asarray(vecs_np[:, :k_eff])

#     if k_eff < k:
#         extra = xp.random.randn(n, k - k_eff)
#         X = xp.concatenate([X, extra], axis=1)

#     if X.ndim == 1:
#         X = X.reshape(-1, 1)

#     return X

# # ----------------------------
# # Utility: tau computation
# # ----------------------------
# def compute_tau(lam, R):
#     lo, hi = lam.min() - 1, lam.max()
#     for _ in range(50):
#         mid = (lo + hi) / 2
#         s = np.sum(np.maximum(lam - mid, 0))
#         if s > R:
#             lo = mid
#         else:
#             hi = mid
#     return (lo + hi) / 2


# # ----------------------------
# # Preconditioner builder
# # ----------------------------
# # def build_preconditioner(D, xp, n):
# #     """
# #     M ≈ A^{-1} using diagonal scaling:
# #         M(x) = D * (D * x)
# #     """
# #     def matvec(x):
# #         return D * (D * x)

# #     if xp is np:
# #         return ScipyLinearOperator((n, n), matvec=matvec)
# #     else:
# #         return CupyxLinearOperator((n, n), matvec=matvec)
# def build_preconditioner(D, xp, n):
#     def matvec(x):
#         x = x.reshape(-1)
#         return (D * (D * x)).reshape(-1)

#     def matmat(X):
#         return D[:, None] * (D[:, None] * X)

#     if xp is np:
#         return ScipyLinearOperator((n, n), matvec=matvec, matmat=matmat)
#     else:
#         return CupyxLinearOperator((n, n), matvec=matvec, matmat=matmat)


# # ----------------------------
# # Rank heuristic for k_init
# # ----------------------------
# def estimate_rank_wsom_trace(traceA, R, n, k_cap=210):
#     """
#     Rough heuristic:
#         rank ≈ trace(A) / avg_eig
#     but we don't know avg eig → assume ~ trace / n

#     → rank ≈ R / (trace/n) = n * R / trace
#     """
#     if traceA <= 0:
#         return min(20, k_cap)
#     # print(f"[SWS] Estimating initial k from trace: trace={traceA}, R={R}, n={n}")
#     k_est = int(np.ceil(n * R / traceA))
#     k_est = max(10, k_est)
#     return min(k_est, k_cap)


# # ----------------------------
# # Mixing schedule
# # ----------------------------
# def mixing_weight(outer_iter_num, max_outer=20000):
#     """
#     Smooth ramp from 0 → 1 over outer iterations
#     """
#     t = min(outer_iter_num / max_outer, 1.0)
#     return t


# # ----------------------------
# # Main SWS warm start
# # ----------------------------
# def facial_reduction_warm_start(
#     A,
#     n,
#     R,
#     D=None,
#     traceA=None,
#     U_prev=None,
#     alpha_prev=None,
#     outer_iter_num=0,
#     k_init=None,
#     max_ws_iters=20,
#     eig_tol=1e-6,
#     k_step=10,
#     k_cap=210,
#     verbose=True,
#     backend="auto",
# ):

#     # ----------------------------
#     # Backend selection
#     # ----------------------------
#     if backend == "cupyx" and HAS_CUPYX:
#         xp = cp
#         lobpcg_fn = cupyx_lobpcg
#         eigsh_fn = cupyx_eigsh
#     elif backend == "auto" and HAS_CUPYX:
#         xp = cp
#         lobpcg_fn = cupyx_lobpcg
#         eigsh_fn = cupyx_eigsh
#     else:
#         xp = np
#         lobpcg_fn = lobpcg
#         eigsh_fn = eigsh
#     # ----------------------------
#     # Initial k
#     # ----------------------------
#     if k_init is None:
#         if traceA is not None:
#             k = estimate_rank_wsom_trace(traceA, R, n, k_cap)
#         else:
#             k = min(20, k_cap)
#     else:
#         k = min(k_init, k_cap)

#     # ----------------------------
#     # Preconditioner
#     # ----------------------------
#     M = None
#     if D is not None:
#         M = build_preconditioner(D, xp, n)

#     num_fr_iters = 0

#     # Initial guess
#     X = None
#     # X = xp.random.randn(n, k)

#     while True:
#         if verbose:
#             print(f"[SWS] Trying k = {k}")

#         # ----------------------------
#         # Build initial block X
#         # ----------------------------
#         if X is None:
#             if U_prev is not None:
#                 # pad or trim
#                 if U_prev.shape[1] >= k:
#                     X = U_prev[:, :k]
#                 else:
#                     extra = xp.random.randn(n, k - U_prev.shape[1])
#                     X = xp.concatenate([U_prev, extra], axis=1)
#             else:
#                 X = xp.random.randn(n, k)

#         # ----------------------------
#         # Run LOBPCG
#         # ----------------------------
#         try:
#             vals, vecs = lobpcg_fn(
#                 A,
#                 X,
#                 M=M,
#                 tol=eig_tol,
#                 maxiter=50,
#                 largest=True,
#             )
#         except Exception:
#             if verbose:
#                 print(f"[SWS] LOBPCG failed for k = {k}")
#             try:
#                 # fallback: random restart
#                 X = xp.random.randn(n, k)
#                 vals, vecs = lobpcg_fn(
#                     A,
#                     X,
#                     M=M,
#                     tol=eig_tol,
#                     maxiter=50,
#                     largest=True,
#                 )
#             except Exception:
#                 if verbose:
#                     print(f"[SWS] LOBPCG failed again for k = {k}, falling back to eigsh")
#                 vals, vecs = eigsh_fn(A, k=k, which='LA', tol=eig_tol)
#         # try:
#         #     if backend == 'cupyx' and HAS_CUPYX or (backend == 'auto' and HAS_CUPYX):
#         #         vals, vecs = cupyx_eigsh(A, k=k, which='LA', tol=eig_tol)
#         #     else:
#         #         vals, vecs = scipy_eigsh(A, k=k, which='LA', tol=eig_tol)
#         # except Exception:
#         #     # Fallback: power iteration for top-k
#         #     Q = xp.random.randn(n, k)
#         #     for _ in range(10):
#         #         Z = xp.column_stack([A.matvec(Q[:, i]) for i in range(k)])
#         #         Q, _ = xp.linalg.qr(Z)

#         #     B = Q.T @ xp.column_stack([A.matvec(Q[:, i]) for i in range(k)])
#         #     vals, vecs_small = xp.linalg.eigh(B)

#         #     idx = xp.argsort(-vals)
#         #     vals = vals[idx]
#         #     vecs = Q @ vecs_small[:, idx]

#         # sort descending
#         idx = xp.argsort(-vals)
#         vals = vals[idx]
#         vecs = vecs[:, idx]

#         # ----------------------------
#         # Projection step
#         # ----------------------------
#         tau = compute_tau(vals, R)
#         alpha = xp.maximum(vals - tau, 0)

#         # ----------------------------
#         # Mixing with previous iterate
#         # ----------------------------
#         if U_prev is not None and alpha_prev is not None:
#             w = mixing_weight(outer_iter_num)

#             # align dimensions
#             k_prev = U_prev.shape[1]
#             k_curr = vecs.shape[1]

#             k_min = min(k_prev, k_curr)

#             vecs[:, :k_min] = (
#                 (1 - w) * vecs[:, :k_min] + w * U_prev[:, :k_min]
#             )
#             alpha[:k_min] = (
#                 (1 - w) * alpha[:k_min] + w * alpha_prev[:k_min]
#             )

#         num_fr_iters += 1

#         if alpha.sum() >= R or k == n or num_fr_iters >= max_ws_iters:
#             break

#         # ----------------------------
#         # Increase k
#         # ----------------------------
#         k = min(k + k_step, k_cap)

#         # recycle eigenvectors
#         X = X = prepare_X(vecs, k, xp, n) # xp.array(vecs_np[:, :k])

#     mask = alpha > 1e-12
#     return vecs[:, mask], alpha[mask]

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 
# import numpy as np

# # Try to import scipy and cupyx LinearOperator
# try:
#     from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, eigsh as scipy_eigsh
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False

# try:
#     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator, eigsh as cupyx_eigsh
#     import cupy as xp
#     HAS_CUPYX = True
# except ImportError:
#     HAS_CUPYX = False

# def compute_tau(lam, R):
#     lo, hi = lam.min() - 1, lam.max()
#     for _ in range(50):
#         mid = (lo + hi) / 2
#         s = np.sum(np.maximum(lam - mid, 0))
#         if s > R:
#             lo = mid
#         else:
#             hi = mid
#     return (lo + hi) / 2

# def facial_reduction_warm_start(A, n, R, k_init=20, max_ws_iters=20, eig_tol=1e-6, k_step=10, verbose=True, backend='auto'):
#     k = k_init

#     num_fr_iters = 0

#     while True:
#         if verbose:
#             print(f"[SWS] Trying k = {k}")

#         try:
#             if backend == 'cupyx' and HAS_CUPYX or (backend == 'auto' and HAS_CUPYX):
#                 vals, vecs = cupyx_eigsh(A, k=k, which='LA', tol=eig_tol)
#             else:
#                 vals, vecs = scipy_eigsh(A, k=k, which='LA', tol=eig_tol)
#         except Exception:
#             # Fallback: power iteration for top-k
#             Q = xp.random.randn(n, k)
#             for _ in range(10):
#                 Z = xp.column_stack([A.matvec(Q[:, i]) for i in range(k)])
#                 Q, _ = xp.linalg.qr(Z)

#             B = Q.T @ xp.column_stack([A.matvec(Q[:, i]) for i in range(k)])
#             vals, vecs_small = xp.linalg.eigh(B)

#             idx = xp.argsort(-vals)
#             vals = vals[idx]
#             vecs = Q @ vecs_small[:, idx]

#         idx = np.argsort(-vals)
#         vals = vals[idx]
#         vecs = vecs[:, idx]

#         tau = compute_tau(vals, R)
#         alpha = np.maximum(vals - tau, 0)

#         num_fr_iters += 1
#         if alpha.sum() >= R or k == n or num_fr_iters >= max_ws_iters:
#             break

#         k = min(k + k_step, n)

#     mask = alpha > 1e-12
#     return vecs[:, mask], alpha[mask]


# import numpy as np

# # --- Backend detection ---
# try:
#     import cupy as cp
#     from cupyx.scipy.sparse.linalg import eigsh as cupyx_eigsh
#     HAS_GPU = True
# except ImportError:
#     HAS_GPU = False

# try:
#     from scipy.sparse.linalg import eigsh as scipy_eigsh
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False


# def get_backend(backend):
#     if backend in ['gpu', 'auto'] and HAS_GPU:
#         return cp, cupyx_eigsh
#     return np, scipy_eigsh


# # --- Stable dual solve (prefix only, no noisy tail) ---
# def solve_tau_prefix(xp, lam, R):
#     S = 0.0
#     for r in range(1, len(lam) + 1):
#         S += lam[r - 1]
#         tau = (S - R) / r

#         # KKT stopping condition
#         if r == len(lam) or tau >= lam[r]:
#             return tau, r

#     return tau, len(lam)


# # --- Warm-start vector selection ---
# def get_init_vector(xp, U_prev):
#     if U_prev is None:
#         return None
#     # use dominant direction as starting vector
#     return U_prev[:, 0]


# # --- Main function ---
# def facial_reduction_warm_start(
#     A,
#     n,
#     R,
#     k_init=20,
#     max_ws_iters=20,
#     eig_tol=1e-6,
#     k_step=10,
#     verbose=True,
#     backend='auto',
#     U_prev=None,
#     alpha_prev=None
# ):
#     xp, eigsh = get_backend(backend)

#     k = k_init

#     for it in range(max_ws_iters):

#         if verbose:
#             print(f"[SWS] Iter {it}, solving top-{k} eigenpairs")

#         # --- Warm start ---
#         v0 = get_init_vector(xp, U_prev)

#         # --- Compute top-k eigenpairs of A ---
#         try:
#             vals, vecs = eigsh(A, k=k, which='LA', tol=eig_tol, v0=v0)
#         except TypeError:
#             # cupyx may not support v0
#             vals, vecs = eigsh(A, k=k, which='LA', tol=eig_tol)

#         # --- Sort descending ---
#         idx = xp.argsort(-vals)
#         lam = vals[idx]
#         vecs = vecs[:, idx]

#         # --- Solve dual using only prefix ---
#         tau, r = solve_tau_prefix(xp, lam, R)

#         if verbose:
#             print(f"[SWS] tau={float(tau):.6f}, rank={r}")

#         # --- Check if we have enough eigenvalues ---
#         if r < k:
#             if tau >= lam[r]:
#                 break

#         # Otherwise, we need more eigenvalues
#         if k >= n:
#             break

#         k = min(k + k_step, n)

#     # --- Final solution ---
#     alpha = xp.maximum(lam[:r] - tau, 0)
#     U = vecs[:, :r]

#     return U, alpha


# import numpy as np

# # --- Backend detection ---
# try:
#     import cupy as cp
#     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator
#     HAS_GPU = True
# except ImportError:
#     HAS_GPU = False


# def get_xp(backend):
#     if backend in ['gpu', 'auto'] and HAS_GPU:
#         return cp
#     return np


# # --- Active-set tau solver (core fix) ---
# def solve_tau_active_set(xp, lam, R):
#     """
#     lam must be sorted descending (largest first)
#     Uses only top eigenvalues to determine tau and rank.
#     """
#     cumsum = xp.cumsum(lam)

#     for r in range(1, len(lam) + 1):
#         tau = (cumsum[r-1] - R) / r

#         # Active set condition
#         if r == len(lam) or tau >= lam[r]:
#             return tau, r

#     return tau, len(lam)


# # --- Krylov expansion (reusable) ---
# def expand_krylov(A, xp, Q, k_target):
#     while Q.shape[1] < k_target:
#         # Generate new block
#         Z = xp.column_stack([A.matvec(Q[:, i]) for i in range(Q.shape[1])])

#         # Orthogonalize against existing Q
#         Z = Z - Q @ (Q.T @ Z)

#         # QR of new directions
#         Z, _ = xp.linalg.qr(Z)

#         # Append
#         Q = xp.hstack([Q, Z])

#         # Truncate if overshoot
#         if Q.shape[1] > k_target:
#             Q = Q[:, :k_target]

#     return Q

# # --- Subspace mixing ---
# def mix_subspaces(xp, Q_new, U_prev, outer_iter):
#     if U_prev is None:
#         return Q_new

#     # Exponential schedule → favors new subspace early
#     tau_mix = 5.0
#     w_max = 0.9
#     w = w_max * (1 - xp.exp(-outer_iter / tau_mix))

#     Q = xp.hstack([
#         xp.sqrt(1 - w) * Q_new,
#         xp.sqrt(w) * U_prev
#     ])

#     Q, _ = xp.linalg.qr(Q)
#     return Q


# # --- Main algorithm ---
# def facial_reduction_warm_start(
#     A,
#     n,
#     R,
#     k_init=20,
#     max_ws_iters=20,
#     eig_tol=1e-6,
#     k_step=10,
#     verbose=True,
#     backend='auto',
#     U_prev=None,
#     alpha_prev=None,
#     outer_iter=0
# ):
#     xp = get_xp(backend)

#     k = k_init

#     # --- Initialize subspace ---
#     Q = xp.random.randn(n, k)
#     Q, _ = xp.linalg.qr(Q)

#     # Optional: incorporate previous iterate early
#     if U_prev is not None:
#         Q = mix_subspaces(xp, Q, U_prev, outer_iter)

#     for it in range(max_ws_iters):

#         if verbose:
#             print(f"[SWS] Iter {it}, subspace dim = {Q.shape[1]}")

#         # --- Expand Krylov subspace (reuse previous Q!) ---
#         Q = expand_krylov(A, xp, Q, k)

#         # --- Rayleigh–Ritz projection ---
#         AQ = xp.column_stack([A.matvec(Q[:, i]) for i in range(Q.shape[1])])
#         B = Q.T @ AQ

#         lam, V_small = xp.linalg.eigh(B)

#         # Sort descending
#         idx = xp.argsort(-lam)
#         lam = lam[idx]
#         V_small = V_small[:, idx]

#         # --- Solve dual using only top eigenvalues ---
#         k_trust = min(20, len(lam))
#         lam_trust = lam[:k_trust]

#         tau, r = solve_tau_active_set(xp, lam_trust, R)

#         if verbose:
#             print(f"[SWS] tau = {float(tau):.6f}, rank = {r}")

#         # --- Check active-set consistency ---
#         if r < len(lam):
#             if tau >= lam[r]:
#                 break
#         else:
#             # Need more eigenvalues
#             pass

#         # Increase subspace size
#         k = min(k + k_step, n)

#         if k >= n:
#             break

#     # --- Final solution ---
#     alpha = xp.maximum(lam[:r] - tau, 0)

#     U = Q @ V_small[:, :r]

#     return U, alpha




# import numpy as np

# # Backend selection
# try:
#     import cupy as cp
#     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator
#     HAS_GPU = True
# except ImportError:
#     HAS_GPU = False


# def get_backend(backend):
#     if backend in ['gpu', 'auto'] and HAS_GPU:
#         return cp
#     return np


# def compute_tau(xp, lam, R):
#     lo, hi = lam.min() - 1, lam.max()
#     for _ in range(50):
#         mid = (lo + hi) / 2
#         s = xp.sum(xp.maximum(lam - mid, 0))
#         if s > R:
#             lo = mid
#         else:
#             hi = mid
#     return (lo + hi) / 2


# def block_lanczos(A, xp, n, m, Q_init=None, n_iter=10):
#     """
#     Build Krylov subspace once (no restart)
#     """
#     if Q_init is None:
#         Q = xp.random.randn(n, m)
#     else:
#         Q = Q_init

#     Q, _ = xp.linalg.qr(Q)

#     for _ in range(n_iter):
#         Z = xp.column_stack([A.matvec(Q[:, i]) for i in range(Q.shape[1])])
#         Q, _ = xp.linalg.qr(Z)

#     return Q


# def estimate_rank_wsom_spectrum(xp, lam, R):
#     tau = compute_tau(xp, lam, R)
#     alpha = xp.maximum(lam - tau, 0)
#     rank = int(xp.sum(alpha > 1e-10))
#     return rank, tau, alpha

# def robust_rank_estimate(xp, lam, R, r_max=None):
#     tau = compute_tau(xp, lam, R)
#     alpha = xp.maximum(lam - tau, 0)

#     eps = 1e-6
#     delta = 1e-3

#     # 1. Relative threshold
#     threshold = eps * max(float(lam[0]), float(tau), 1.0)
#     mask = alpha > threshold

#     # 2. Energy truncation
#     alpha_pos = alpha[mask]
#     if alpha_pos.size == 0:
#         return 0

#     cumsum = xp.cumsum(alpha_pos)
#     total = cumsum[-1]
#     r_energy = int(xp.searchsorted(cumsum, (1 - delta) * total)) + 1

#     r = min(int(mask.sum()), r_energy)

#     # 3. Optional cap
#     if r_max is not None:
#         r = min(r, r_max)

#     return r, tau, alpha


# def mix_subspaces(xp, Q_new, U_prev=None, weight=0.5):
#     if U_prev is None:
#         return Q_new

#     # Blend bases
#     Q = xp.hstack([
#         xp.sqrt(1 - weight) * Q_new,
#         xp.sqrt(weight) * U_prev
#     ])

#     Q, _ = xp.linalg.qr(Q)
#     return Q


# def facial_reduction_warm_start(
#     A,
#     n,
#     R,
#     outer_iter=0,
#     U_prev=None,
#     alpha_prev=None,
#     m=40,
#     lanczos_iters=8,
#     backend='auto',
#     verbose=True
# ):
#     xp = get_backend(backend)

#     if verbose:
#         print("[SWS] Running block Lanczos warm start")

#     # Step 1: build Krylov subspace once
#     Q = block_lanczos(A, xp, n, m, n_iter=lanczos_iters)

#     # Step 2: mix with previous iterate (adaptive weight)
#     if U_prev is not None:
#         def compute_weight(outer_iter, T=20000, w_max=0.9):
#             t = outer_iter
#             return min(w_max, (t / T)**2)
#         weight = compute_weight(outer_iter)
#         print(f"[SWS] Mixing with previous subspace, weight = {weight:.2f}")
#         Q = mix_subspaces(xp, Q, U_prev, weight)

#     # Step 3: Rayleigh-Ritz projection
#     AQ = xp.column_stack([A.matvec(Q[:, i]) for i in range(Q.shape[1])])
#     B = Q.T @ AQ

#     lam, V_small = xp.linalg.eigh(B)

#     idx = xp.argsort(-lam)
#     lam = lam[idx]
#     V_small = V_small[:, idx]

#     # Step 4: rank estimation via soft-threshold
#     rank, tau, alpha = robust_rank_estimate(xp, lam, R) # estimate_rank_wsom_spectrum(xp, lam, R)

#     if verbose:
#         print(f"[SWS] Estimated rank = {rank}")

#     # Step 5: lift eigenvectors
#     U = Q @ V_small[:, :rank]
#     alpha = alpha[:rank]

#     return U, alpha


# =================================================================================

# import numpy as np

# # Try to import scipy and cupyx LinearOperator
# try:
#     from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, eigsh as scipy_eigsh
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False

# try:
#     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator, eigsh as cupyx_eigsh
#     import cupy as xp
#     HAS_CUPYX = True
# except ImportError:
#     HAS_CUPYX = False

# def compute_tau(lam, R):
#     lo, hi = lam.min() - 1, lam.max()
#     for _ in range(50):
#         mid = (lo + hi) / 2
#         s = np.sum(np.maximum(lam - mid, 0))
#         if s > R:
#             lo = mid
#         else:
#             hi = mid
#     return (lo + hi) / 2

# def facial_reduction_warm_start(A, n, R, k_init=20, max_ws_iters=20, eig_tol=1e-6, k_step=10, verbose=True, backend='auto'):
#     k = k_init

#     num_fr_iters = 0

#     while True:
#         if verbose:
#             print(f"[SWS] Trying k = {k}")

#         try:
#             if backend == 'cupyx' and HAS_CUPYX or (backend == 'auto' and HAS_CUPYX):
#                 vals, vecs = cupyx_eigsh(A, k=k, which='LA', tol=eig_tol)
#             else:
#                 vals, vecs = scipy_eigsh(A, k=k, which='LA', tol=eig_tol)
#         except Exception:
#             # Fallback: power iteration for top-k
#             Q = xp.random.randn(n, k)
#             for _ in range(10):
#                 Z = xp.column_stack([A.matvec(Q[:, i]) for i in range(k)])
#                 Q, _ = xp.linalg.qr(Z)

#             B = Q.T @ xp.column_stack([A.matvec(Q[:, i]) for i in range(k)])
#             vals, vecs_small = xp.linalg.eigh(B)

#             idx = xp.argsort(-vals)
#             vals = vals[idx]
#             vecs = Q @ vecs_small[:, idx]

#         idx = np.argsort(-vals)
#         vals = vals[idx]
#         vecs = vecs[:, idx]

#         tau = compute_tau(vals, R)
#         alpha = np.maximum(vals - tau, 0)

#         num_fr_iters += 1
#         if alpha.sum() >= R or k == n or num_fr_iters >= max_ws_iters:
#             break

#         k = min(k + k_step, n)

#     mask = alpha > 1e-12
#     return vecs[:, mask], alpha[mask]

# ================================================================================================
# import numpy as np

# # Try to import scipy and cupyx LinearOperator
# try:
#     from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, eigsh as scipy_eigsh
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False

# try:
#     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator, eigsh as cupyx_eigsh
#     import cupy as xp
#     HAS_CUPYX = True
# except ImportError:
#     HAS_CUPYX = False

# def orthonormalize(V):
#     xp = np if isinstance(V, np.ndarray) else __import__('cupy')
#     Q, _ = xp.linalg.qr(V)
#     return Q

# def restrict_operator(A, U):
#     # Returns LinearOperator for U^T A U
#     def matvec(x):
#         return U.T @ (A @ (U @ x))
#     n = U.shape[1]
#     if isinstance(A, CupyxLinearOperator):
#         return CupyxLinearOperator((n, n), matvec=matvec, dtype=float)
#     else:
#         return ScipyLinearOperator((n, n), matvec=matvec, dtype=float)

# def facial_reduction_warm_start(
#     A, n, R,
#     max_ws_iters=20,
#     eig_tol=1e-6,
#     neg_tol=-1e-8,
#     k_init=20,
#     k_step=10,
#     backend='auto',
# ):
#     """
#     A: LinearOperator representing symmetric matrix
#     n: dimension
#     R: trace constraint
#     """


#     # Backend selection

#     # Start with full space
#     U = xp.eye(n)

#     for it in range(max_ws_iters):
#         def matvec(x):
#             return U.T @ (A @ (U @ x))
        
#         if backend == 'cupyx' and HAS_CUPYX or (backend == 'auto' and HAS_CUPYX):
#             A_U = CupyxLinearOperator((U.shape[1], U.shape[1]), matvec=matvec, dtype=float)
#         else:
#             A_U = ScipyLinearOperator((U.shape[1], U.shape[1]), matvec=matvec, dtype=float)

#         # smallest eigenvalue
#         try:
#             if backend == 'cupyx' and HAS_CUPYX or (backend == 'auto' and HAS_CUPYX):
#                 vals, vecs = cupyx_eigsh(A_U, k=1, which='SA', tol=eig_tol)
#             elif backend == 'scipy' and HAS_SCIPY:
#                 vals, vecs = scipy_eigsh(A_U, k=1, which='SA', tol=eig_tol)
#             lam_min = vals[0]
#             v = vecs[:, 0]
#         except Exception:
#             # Use power iteration
#             v = xp.random.randn(U.shape[1])
#             v = v / xp.linalg.norm(v)
#             for _ in range(100):
#                 v = A_U @ v
#                 v = v / xp.linalg.norm(v)
#             lam_min = v @ (A_U @ v)

#         W = []  # before loop
        
#         if lam_min < neg_tol:
#             w = U @ v
#             w = w / xp.linalg.norm(w)

#             # Orthogonalize against previous W
#             for w_prev in W:
#                 w = w - (w_prev @ w) * w_prev

#             norm_w = xp.linalg.norm(w)
#             if norm_w < 1e-10:
#                 print("[FR] direction already removed (numerical), stopping")
#                 break

#             w = w / norm_w
#             W.append(w)

#             # Rebuild U as orthogonal complement of W
#             W_mat = xp.column_stack(W)  # n x k
#             Q, _ = xp.linalg.qr(W_mat)

#             # Null space via projection
#             U = xp.eye(n) - Q @ Q.T
#             U = orthonormalize(U)

#             print(f"[FR] iter={it}, removed NEW direction, λ_min={lam_min:.2e}")
#         else:
#             print(f"[FR] converged at iter={it}, λ_min={lam_min:.2e}")
#             break

#     # Now compute leading eigenpairs in reduced space
#     def matvec2(x):
#         return U.T @ (A @ (U @ x))
#     if backend == 'cupyx' and HAS_CUPYX or (backend == 'auto' and HAS_CUPYX):
#         A_U = CupyxLinearOperator((U.shape[1], U.shape[1]), matvec=matvec2, dtype=float)
#     else:
#         A_U = ScipyLinearOperator((U.shape[1], U.shape[1]), matvec=matvec2, dtype=float)

#     k = min(k_init, U.shape[1])
#     while True:
#         try:
#             if backend == 'cupyx' and HAS_CUPYX or (backend == 'auto' and HAS_CUPYX):
#                 vals, vecs = cupyx_eigsh(A_U, k=k, which='LA', tol=eig_tol)
#             else:
#                 vals, vecs = scipy_eigsh(A_U, k=k, which='LA', tol=eig_tol)
#         except Exception:
#             # Fallback: power iteration for top-k
#             Q = xp.random.randn(n, k)
#             for _ in range(10):
#                 Z = xp.column_stack([A_U.matvec(Q[:, i]) for i in range(k)])
#                 Q, _ = xp.linalg.qr(Z)

#             B = Q.T @ xp.column_stack([A_U.matvec(Q[:, i]) for i in range(k)])
#             vals, vecs_small = xp.linalg.eigh(B)

#             idx = xp.argsort(-vals)
#             vals = vals[idx]
#             vecs = Q @ vecs_small[:, idx]

#         # sort descending
#         idx = xp.argsort(-vals)
#         vals = vals[idx]
#         vecs = vecs[:, idx]

#         # find threshold τ for trace constraint
#         alpha = xp.maximum(vals, 0)
#         if alpha.sum() >= R or k == U.shape[1]:
#             break

#         k = min(k + k_step, U.shape[1])

#     # find τ by bisection
#     def compute_tau(lam, R):
#         lo, hi = float(xp.min(lam)) - 1, float(xp.max(lam))
#         for _ in range(50):
#             mid = (lo + hi) / 2
#             s = float(xp.sum(xp.maximum(lam - mid, 0)))
#             if s > R:
#                 lo = mid
#             else:
#                 hi = mid
#         return (lo + hi) / 2

#     tau = compute_tau(vals, R)
#     alpha = xp.maximum(vals - tau, 0)

#     # keep only positive entries
#     mask = alpha > 1e-12
#     alpha = alpha[mask]
#     V = vecs[:, mask]

#     # map back to full space
#     U_final = U @ V

#     return U_final, alpha


# import numpy as xp
# from torch.linalg import eigvals

# # Try to import scipy and cupyx LinearOperator
# try:
#     from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator, eigsh as scipy_eigsh
#     HAS_SCIPY = True
# except ImportError:
#     HAS_SCIPY = False

# try:
#     from cupyx.scipy.sparse.linalg import LinearOperator as CupyxLinearOperator
#     import cupy as xp
#     HAS_CUPYX = True
# except ImportError:
#     HAS_CUPYX = False


# # -----------------------------
# # Utility: wrap matvec into LinearOperator
# # -----------------------------

# def make_linear_operator(matvec, n, backend='auto'):
#     """
#     backend: 'auto', 'scipy', or 'cupyx'
#     """
#     if backend == 'cupyx' or (backend == 'auto' and HAS_CUPYX):
#         if not HAS_CUPYX:
#             raise ImportError("cupyx is not available")
#         return CupyxLinearOperator((n, n), matvec=matvec)
#     elif backend == 'scipy' or (backend == 'auto' and HAS_SCIPY):
#         return ScipyLinearOperator((n, n), matvec=matvec)
#     else:
#         # fallback: emulate operator
#         class SimpleOp:
#             def __init__(self, matvec, n):
#                 self.matvec = matvec
#                 self.shape = (n, n)

#         return SimpleOp(matvec, n)


# # -----------------------------
# # Top-k eigenpairs (robust)
# # -----------------------------

# def top_k_eigenpairs(op, n, k, maxiter=200, backend='auto'):
#     """
#     Returns largest k eigenvalues/eigenvectors of symmetric operator
#     backend: 'auto', 'scipy', or 'cupyx'
#     """
#     k = min(k, n - 1)

#     # Use cupyx if op is a CupyxLinearOperator or backend is 'cupyx'
#     if (backend == 'cupyx' or (backend == 'auto' and HAS_CUPYX and isinstance(op, CupyxLinearOperator))):
#         # Use cupyx eigsh if available
#         try:
#             from cupyx.scipy.sparse.linalg import eigsh as cupyx_eigsh
#             v0 = xp.random.randn(n)
#             v0 /= xp.linalg.norm(v0)
#             vals, vecs = cupyx_eigsh(
#                 op,
#                 k=k,
#                 which='LA',
#                 v0=v0,
#                 maxiter=maxiter,
#                 tol=1e-6,
#             )
#             # vals = xp.asnumpy(vals)
#             # vecs = xp.asnumpy(vecs)
#             idx = xp.argsort(-vals)
#             return vals[idx], vecs[:, idx]
#         except Exception:
#             pass  # fallback below

#     if (backend == 'scipy' or (backend == 'auto' and HAS_SCIPY and isinstance(op, ScipyLinearOperator))):
#         try:
#             v0 = xp.random.randn(n)
#             v0 /= xp.linalg.norm(v0)
#             vals, vecs = scipy_eigsh(
#                 op,
#                 k=k,
#                 which='LA',   # largest algebraic
#                 v0=v0,
#                 maxiter=maxiter,
#                 tol=1e-6,
#             )
#             idx = xp.argsort(-vals)
#             return vals[idx], vecs[:, idx]
#         except Exception:
#             pass  # fallback below

#     # -------- fallback: randomized power iteration --------
#     Q = xp.random.randn(n, k)
#     for _ in range(10):
#         Z = xp.column_stack([op.matvec(Q[:, i]) for i in range(k)])
#         Q, _ = xp.linalg.qr(Z)

#     B = Q.T @ xp.column_stack([op.matvec(Q[:, i]) for i in range(k)])
#     vals, vecs_small = xp.linalg.eigh(B)

#     idx = xp.argsort(-vals)
#     vals = vals[idx]
#     vecs = Q @ vecs_small[:, idx]

#     return vals, vecs


# # -----------------------------
# # Solve for lambda (bisection)
# # -----------------------------
# def find_lambda(eigvals, R, tol=1e-10):
#     lo = 0.0
#     hi = max(eigvals)

#     for _ in range(100):
#         mid = 0.5 * (lo + hi)
#         val = xp.sum(xp.maximum(eigvals - mid, 0))

#         if val > R:
#             lo = mid
#         else:
#             hi = mid

#         if abs(val - R) < tol:
#             break

#     return hi


# # -----------------------------
# # Main function: approximate SWS
# # -----------------------------
# def approx_SWS_trace(
#     op,
#     n,
#     R,
#     k_init=20,
#     k_max=None,
#     tol=1e-6,
#     verbose=True,
#     backend='auto',
# ):
#     """
#     Returns:
#         U  : (n x r) orthonormal basis of face
#         lam: threshold
#         eigvals: computed eigenvalues
#     """

#     k = k_init
#     if k_max is None:
#         k_max = n

#     prev_dim = None

#     while True:
#         if verbose:
#             print(f"[SWS] Trying k = {k}")

#         # Step 1: compute eigenpairs
#         eigvals, eigvecs = top_k_eigenpairs(op, n, k, backend=backend)

#         # Step 2: find lambda
#         lam = find_lambda(eigvals, R)

#         # Ensure we captured ALL active eigenvalues
#         needs_expansion = eigvals[-1] > lam + tol
#         at_capacity = (k >= n or k >= k_max)

#         if needs_expansion and not at_capacity:
#             k = min(2 * k, k_max, n)
#             continue
#         # Otherwise: accept approximation
#         elif needs_expansion and at_capacity:
#             if verbose:
#                 print("[SWS] Warning: spectrum likely truncated, accepting approximate solution.")

#         # Step 3: active eigenvalues
#         active = eigvals > lam + tol

#         r = int(xp.sum(active))

#         if verbose:
#             print(f"       lambda = {lam:.6f}, active rank = {r}")

#         # If no active eigenvalues → increase k
#         if r == 0:
#             k = min(min(2 * k, k_max), n)
#             if k == k_max or k == n:
#                 raise RuntimeError("Failed: no active eigenvalues")
#             continue

#         # Step 4: form subspace
#         U_active = eigvecs[:, active]
#         eigvals_active = eigvals[active]

#         # Step 5: trace consistency check
#         trace_val = xp.sum(xp.maximum(eigvals - lam, 0))

#         if verbose:
#             print(f"       trace approx = {trace_val:.6f} (target {R})")

#         if abs(trace_val - R) < tol:
#             if verbose:
#                 print("[SWS] Converged.")
#             break

#         # Step 6: expand subspace
#         if k >= k_max or k >= n:
#             if verbose:
#                 print("[SWS] Reached k_max or n, stopping.")
#             break

#         if prev_dim == r:
#             # no progress → expand aggressively
#             k = min(min(2 * k, k_max), n)
#         else:
#             k = min(min(k + max(10, r), k_max), n)

#         prev_dim = r

#     return U_active, lam, eigvals_active


# # -----------------------------
# # Optional: reconstruct solution
# # -----------------------------
# def reconstruct_X(U, eigvals, lam):
#     """
#     Build X* = U diag(max(eigvals - lam, 0)) U^T
#     """
#     active_vals = xp.maximum(eigvals - lam, 0)
#     # active = active_vals > 0

#     # U_active = U[:, active]
#     # vals = active_vals[active]

#     return U @ xp.diag(active_vals) @ U.T


# if __name__ == "__main__":

#     for n, R in [(1000, 1000), (2000, 2000), (5000, 5000)]:
#         print(f"\n===== Testing n = R = {n} =====")
#         # Generate random symmetric matrix
#         G = xp.random.randn(n, n)
#         G = (G + G.T) / 2

#         def matvec(v):
#             return G @ v

#         # Choose backend: 'auto', 'scipy', or 'cupyx'
#         op = make_linear_operator(matvec, n, backend='auto')
    
#         # SWS result
#         U, lam, eigvals = approx_SWS_trace(op, n, R, verbose=True, backend='auto')

#         # Ground truth
#         w, V = xp.linalg.eigh(G)
#         idx = xp.argsort(-w)
#         w = w[idx]
#         V = V[:, idx]

#         lam_true = find_lambda(w, R)
#         active = w > lam_true
#         U_true = V[:, active]

#         # Check subspace alignment
#         P = U @ U.T
#         P_true = U_true @ U_true.T
#         err = xp.linalg.norm(P - P_true)
#         print('Subspace alignment error:', err)

#         # Reconstruct X
#         X = reconstruct_X(U, eigvals, lam)
#         S = X - G + lam * xp.eye(n)
#         residual = xp.linalg.norm(X @ S)
#         print("Complementarity residual:", residual)

#         # Check trace constraint
#         print('Trace constraint error:', abs(xp.trace(X) - R))
    
# # import numpy as xp
# # from numpy.linalg import eigh, norm

# # try:
# #     from scipy.sparse.linalg import eigsh  # for large problems
# #     HAS_SCIPY = True
# # except ImportError:
# #     HAS_SCIPY = False


# # # -----------------------------
# # # Helper: projected gradient for dual
# # # -----------------------------
# # def solve_dual_projected_gradient(A_star, A, b, Yk, max_iters=50, step=1e-2):
# #     """
# #     Approximately solve:
# #         min ½ ||A*(λ) - Yk||_F^2 + b^T λ
# #         s.t. S = Yk - A*(λ) ⪰ 0

# #     using projected gradient on λ
# #     """
# #     m = len(b)
# #     lam = xp.zeros(m)

# #     for _ in range(max_iters):
# #         Astar_lam = A_star(lam)
# #         grad = A(Astar_lam - Yk) + b  # gradient

# #         lam = lam - step * grad

# #     # final slack
# #     S = Yk - A_star(lam)

# #     # project S onto PSD cone (important!)
# #     eigvals, eigvecs = eigh((S + S.T) / 2)
# #     eigvals_clipped = xp.maximum(eigvals, 0)
# #     S_psd = eigvecs @ xp.diag(eigvals_clipped) @ eigvecs.T

# #     return lam, S_psd


# # # -----------------------------
# # # Helper: get small eigenvectors
# # # -----------------------------
# # def small_eigspace(S, eps=1e-6, max_rank=None):
# #     """
# #     Returns eigenvectors corresponding to small eigenvalues of S
# #     """
# #     n = S.shape[0]

# #     if HAS_SCIPY and n > 200:
# #         # compute smallest eigenvalues only
# #         k = min(n - 1, max_rank or 20)
# #         vals, vecs = eigsh(S, k=k, which='SM')
# #     else:
# #         vals, vecs = eigh(S)

# #     # threshold
# #     thresh = eps * max(1.0, xp.max(vals))
# #     idx = xp.where(vals <= thresh)[0]

# #     if len(idx) == 0:
# #         return None

# #     return vecs[:, idx]


# # # -----------------------------
# # # Main algorithm: Inexact SWS
# # # -----------------------------
# # def inexact_SWS_projection(
# #     Y,
# #     A,
# #     A_star,
# #     b,
# #     eps_face=1e-6,
# #     eps_dual_iters=30,
# #     max_iter=5,
# #     verbose=True,
# # ):
# #     """
# #     Returns:
# #         U: basis of approximate minimal face
# #     """

# #     n = Y.shape[0]
# #     U = xp.eye(n)

# #     for k in range(max_iter):
# #         if verbose:
# #             print(f"\n[SWS] Iteration {k}, dim = {U.shape[1]}")

# #         # Restrict Y
# #         Yk = U.T @ Y @ U

# #         # Define reduced operators
# #         def A_k(Z):
# #             return A(U @ Z @ U.T)

# #         def A_k_star(lam):
# #             return U.T @ A_star(lam) @ U

# #         # Solve approximate dual
# #         lam, S_k = solve_dual_projected_gradient(
# #             A_k_star,
# #             A_k,
# #             b,
# #             Yk,
# #             max_iters=eps_dual_iters,
# #         )

# #         # Symmetrize
# #         S_k = (S_k + S_k.T) / 2

# #         # Eigenvalue decomposition
# #         Q_small = small_eigspace(S_k, eps=eps_face)

# #         if Q_small is None:
# #             if verbose:
# #                 print("[SWS] No small eigenvalues → stop")
# #             break

# #         # Update subspace
# #         U_new = U @ Q_small

# #         if verbose:
# #             print(f"[SWS] Reduced dim → {U_new.shape[1]}")

# #         # Check if no reduction
# #         if U_new.shape[1] == U.shape[1]:
# #             if verbose:
# #                 print("[SWS] No further reduction → stop")
# #             break

# #         # Orthonormalize (important!)
# #         U, _ = xp.linalg.qr(U_new)

# #     return U


# # if __name__ == "__main__":

# #     def A_trace(X):
# #         return xp.array([xp.trace(X)])

# #     def A_star_trace(lam):
# #         return lam[0] * xp.eye(n)


# #     # Problem size
# #     n = 1000

# #     # Random symmetric matrix
# #     xp.random.seed(0)
# #     Y = xp.random.randn(n, n)
# #     Y = (Y + Y.T) / 2

# #     b = xp.array([1.0])

# #     # Run inexact SWS
# #     U = inexact_SWS_projection(
# #         Y,
# #         A_trace,
# #         A_star_trace,
# #         b,
# #         max_iter=10,
# #     )

# #     print("\nFinal subspace dimension:", U.shape[1]