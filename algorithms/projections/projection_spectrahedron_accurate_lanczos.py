"""
Accurate Lanczos projection onto the spectrahedron.

Uses the common Lanczos utilities from lanczos_utils.py.

Projection:

    minimize_X  1/2 ||X - G||_F^2
    subject to  X >= 0
                tr(X) = diameter

No Frank-Wolfe iterations are performed.
"""

import time

from algorithms.projections.lanczos_utils import (
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


def accurate_lanczos_projection(
    G,
    start_value=None,
    epsilon_func=None,
    outer_iter_num=0,
    maxit=None,
    line_search=False,
    diameter=1.0,
    return_logging=False,
    verbose=False,
    preprocess_input=False,
    generate_spectral_warm_start=False,
    timing=False,
    sync_gpu_for_timing=False,
    boundary_certificate=True,
    boundary_eta=1e-2,

    # Accurate-Lanczos options
    lanczos_tol= 1e-8,
    boundary_tol= 1e-4,
    k_init=20,
    k_step=20,
    k_cap=None,
    # residual_tol=1e-9,
    active_tol=1e-12,
):
    """
    Accurate Lanczos projection onto the spectrahedron.

    This is a drop-in replacement for the existing projH call.

    Parameters
    ----------
    G : matrix or LinearOperator
        Symmetric projection input.

    start_value : tuple(U_prev, alpha_prev), optional
        Previous factorized iterate.

        The dominant previous eigenvector

            U_prev[:, 0]

        is recycled as the Lanczos starting vector.

    epsilon_func, line_search, preprocess_input,
    enable_rank_compression, enable_atom_reuse,
    generate_spectral_warm_start :
        Accepted for compatibility with the existing PDHG interface.
        They are not used.

    diameter : float
        Trace constraint.

    maxit : int
        Maximum number of Lanczos iterations passed to eigsh.

    lanczos_tol : float
        Accurate Lanczos tolerance.

    k_init : int
        Initial number of eigenpairs.

    k_step : int
        Increase in k when waterfilling is incomplete.

    k_cap : int, optional
        Maximum number of eigenpairs.

    residual_tol : float
        Threshold used only for logging.

    active_tol : float
        Threshold for the returned active coefficients.

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
    (U, alpha), logging
    """

    measure_timing = bool(timing)
    sync_for_timing = bool(sync_gpu_for_timing)

    start_time = time.perf_counter() if measure_timing else None

    # ==================================================================
    # Backend
    # ==================================================================

    backend = detect_backend(
        G,
        start_value,
    )

    xp = get_xp(backend)
    eigsh_fn = get_eigsh(backend)

    n = G.shape[0]

    if G.shape[0] != G.shape[1]:
        raise ValueError(
            f"G must be square, got {G.shape}"
        )

    if n < 2:
        raise ValueError(
            "Lanczos projection requires n >= 2."
        )

    # ==================================================================
    # k limits
    # ==================================================================

    k_cap_effective = min(
        n - 1,
        1000 if k_cap is None else k_cap,
    )

    if k_cap_effective < 1:
        raise ValueError(
            "k_cap must be at least 1."
        )

    if start_value:
        U_init, alpha_init = start_value
        k_init = max(U_init.shape[1], 20)

    k = min(
        k_init,
        k_cap_effective,
    )

    # ==================================================================
    # Recycled starting vector
    # ==================================================================

    v0 = get_initial_v0(
        start_value,
        xp,
    )

    # ==================================================================
    # Counted operator
    # ==================================================================

    matvec_counter = [0]

    counted_G = make_counted_operator(
        G=G,
        backend=backend,
        dtype=G.dtype,
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
    # final_residuals = None

    final_boundary_ratio = None
    waterfill_complete = False

    num_lanczos_rounds = 0
    total_lanczos_time = 0.0

    # ==================================================================
    # Adaptive Lanczos + waterfilling
    # ==================================================================

    while True:

        num_lanczos_rounds += 1

        if verbose:
            print(
                f"[AL] backend={backend} "
                f"round={num_lanczos_rounds} "
                f"k={k} "
                f"tol={lanczos_tol:.1e}"
            )

        # --------------------------------------------------------------
        # Lanczos
        # --------------------------------------------------------------

        lanczos_start = time.perf_counter() if measure_timing else None

        eigsh_kwargs = dict(
            k=k,
            which="LA",
            tol=lanczos_tol,
            # maxiter=maxit,
            # ncv=min(
            #     n,
            #     max(2 * k + 20, 40),
            # ),
        )

        # Only provide v0 when one exists.
        if v0 is not None:
            eigsh_kwargs["v0"] = v0

        vals, vecs = eigsh_fn(
            counted_G,
            **eigsh_kwargs,
        )

        if measure_timing:
            if backend == "cupy" and sync_for_timing:
                xp.cuda.get_current_stream().synchronize()

            total_lanczos_time += (
                time.perf_counter()
                - lanczos_start
            )

        # --------------------------------------------------------------
        # Sort eigenpairs
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
        #     G,
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
            diameter,
            xp,
            active_tol=active_tol,
        )

        # --------------------------------------------------------------
        # Boundary
        # --------------------------------------------------------------

        boundary_ratio = waterfill_boundary_ratio(
            alpha,
            xp,
        )

        waterfill_complete = (
            boundary_ratio <= boundary_tol
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
        final_boundary_ratio = boundary_ratio

        rank = int(
            xp.sum(active)
        )

        if verbose:
            if measure_timing:
                print(
                    f"[AL] rank={rank} "
                    f"theta={float(theta):.6e} "
                    f"alpha_k={float(alpha[-1]):.3e} "
                    f"rel_boundary={boundary_ratio:.3e} "
                    # f"max_res={max_residual:.3e} "
                    f"matvecs={matvec_counter[0]}"
                )
            else:
                print(
                    f"[AL] rank={rank} "
                    f"theta={float(theta):.6e} "
                    f"alpha_k={float(alpha[-1]):.3e} "
                    f"rel_boundary={boundary_ratio:.3e} "
                    # f"max_res={max_residual:.3e} "
                    f"matvecs={matvec_counter[0]}"
                )

        # --------------------------------------------------------------
        # Stop: waterfilling complete
        # --------------------------------------------------------------

        if waterfill_complete:
            break

        # --------------------------------------------------------------
        # Stop: k cap
        # --------------------------------------------------------------

        if k >= k_cap_effective:
            if verbose:
                print(
                    "[AL] k_cap reached before "
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

    mask = final_alpha > active_tol

    U = final_vecs[:, mask]
    alpha = final_alpha[mask]

    rank = int(
        xp.sum(mask)
    )

    if rank < len(final_vals):
        spectral_gap = float(
            final_vals[rank - 1] - final_vals[rank]
        )
    else:
        spectral_gap = xp.inf

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
        diameter,
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


    boundary_certificate_ok = None
    boundary_cert = None
    boundary_cert_matvecs = 0

    if boundary_certificate:
        before = matvec_counter[0]

        boundary_certificate_ok, boundary_cert = (
            boundary_residual_certificate(
                G=counted_G,
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

    logging = {
        "backend": backend,
        "method": "accurate_lanczos",

        "outer_iter_num": outer_iter_num,

        "lanczos_tol": lanczos_tol,
        "maxit": maxit,

        "k": len(final_vals),
        "k_cap": k_cap_effective,
        "k_init": k_init,
        "k_step": k_step,

        "rank": rank,
        "theta": float(final_theta),

        "trace_error": tr_error,

        # "max_eig_residual": final_max_residual,
        # "mean_eig_residual": final_mean_residual,
        # "residual_tol": residual_tol,
        "eigengap": spectral_gap,

        # "lanczos_residual_ok": (
        #     final_max_residual <= residual_tol
        # ),

        "waterfill_complete": waterfill_complete,
        "alpha_k": float(final_alpha[-1]),
        "relative_boundary": final_boundary_ratio,

        "matvec_count": algorithm_matvec_count,
        "lanczos_rounds": num_lanczos_rounds,
        "lanczos_time": total_lanczos_time if measure_timing else None,
        "total_time": total_time,

        "fw_iterations": 0,
        "line_search_iterations": 0,
        "boundary_certificate_matvecs": boundary_cert_matvecs,
        "boundary_certificate_ok": boundary_certificate_ok,
        "boundary_certificate": boundary_cert,
    }

    if verbose:
        print()
        print("[AL] ========================================")
        print("[AL] Accurate Lanczos projection complete")
        print("[AL] ========================================")
        print(f"[AL] backend       = {backend}")
        print(f"[AL] k            = {len(final_vals)}")
        print(f"[AL] rank         = {rank}")
        print(f"[AL] theta        = {float(final_theta):.6e}")
        # print(
        #     f"[AL] max residual = "
        #     f"{final_max_residual:.3e}"
        # )
        # print(
        #     f"[AL] mean residual= "
        #     f"{final_mean_residual:.3e}"
        # )
        print(
            f"[AL] trace error  = "
            f"{tr_error:.3e}"
        )
        print(
            f"[AL] matvecs      = "
            f"{matvec_counter[0]}"
        )
        if measure_timing:
            print(
                f"[AL] Lanczos time = "
                f"{total_lanczos_time:.3f}s"
            )
            print(
                f"[AL] total time   = "
                f"{total_time:.3f}s"
            )
        print("[AL] FW iterations = 0")
        print("[AL] ========================================")

    return (
        (U, alpha),
        logging,
    )