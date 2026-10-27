import time
import math
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.sparse import csr_matrix
import cupyx.scipy.sparse.linalg as cusplinalg
from cupyx.scipy.sparse.linalg import LinearOperator
from algorithms.projections.spectral_warm_start import spectral_warm_start 
import cProfile
import pstats

_REG_EPS = 1e-12

# =========================================================
# Spectrahedron Frank-Wolfe 
# =========================================================
def spectrahedron_fw_projection(
    G,
    start_value,
    epsilon_func,
    outer_iter_num,
    maxit,
    line_search=True,
    diameter: float = 1.0,
    return_logging: bool = False,
    verbose: bool = False,
    preprocess_input: bool = False,
    generate_spectral_warm_start: bool = True,
    timing: bool = False,
    sync_gpu_for_timing: bool = False,
):
    measure_timing = bool(timing)
    sync_for_timing = bool(sync_gpu_for_timing)

    # start_value: (U_init, alpha_init), G: LinearOperator
    U_init, alpha_init = start_value
    n = U_init.shape[0]
    D, traceG = estimate_diag_and_trace(G, n)

    if generate_spectral_warm_start:
        k_init = max(U_init.shape[1], 20)

        U_init, alpha_init, logging_ws = spectral_warm_start(
                                            A=G,
                                            n=n,
                                            R=diameter,
                                            U_prev=U_init,
                                            alpha_prev=alpha_init,
                                            outer_iter_num=outer_iter_num,
                                            k_init=k_init,
                                            k_step=20,
                                            k_cap=None,
                                            sws_tol=1e-2,
                                            boundary_tol=1e-4,
                                            maxiter=None,
                                            backend="auto",
                                            verbose=False,
                                            timing=measure_timing,
                                            sync_gpu_for_timing=sync_for_timing,
                                            boundary_certificate=True,
                                            boundary_eta=1e-2,
                                        )

    alpha_init /= diameter  # Rescale alpha to account for diameter
    spec_norm = 1.0
    Dinv = None
    Dinv_sq = None
    if preprocess_input:
        traceG = 0 # disable trace shift for now
        Dinv = 1 / D
        Dinv_sq = Dinv * Dinv
        orig_mv = G.matvec
        orig_rmv = G.rmatvec

        def G_centered_mv(x):
            return orig_mv(x) - (traceG/n) * x

        def G_centered_rmv(x):
            return orig_rmv(x) - (traceG/n) * x

        G_shifted = LinearOperator((n, n), matvec=G_centered_mv, rmatvec=G_centered_rmv) 

        G_cond = make_preconditioned_operator(G_shifted, D, n)
        G_cond_mv = G_cond.matvec
        G_cond_rmv = G_cond.rmatvec

        spec_norm = estimate_spectral_norm(G_shifted, n)

        def G_scaled_mv(x):
            return G_cond_mv(x) / spec_norm

        def G_scaled_rmv(x):
            return G_cond_rmv(x) / spec_norm

        G = LinearOperator((n, n), matvec=G_scaled_mv, rmatvec=G_scaled_rmv)

    # Scale U
    if preprocess_input and Dinv is not None:
        U_init = D[:, None] * U_init
        scaled_start_value = (D[:, None] * start_value[0],  start_value[1] / diameter)
    else:
        scaled_start_value = (start_value[0],  start_value[1] / diameter)

    # Cache GU 
    if preprocess_input and Dinv is not None:
        GU_init = cp.column_stack([G.matvec(Dinv_sq * U_init[:, i]) for i in range(U_init.shape[1])])
    else:
        GU_init = cp.column_stack([G.matvec(U_init[:, i]) for i in range(U_init.shape[1])])

    # Initialize from U_init, alpha_init
    state = State(G=G, U=U_init, alpha=alpha_init, n=n, diameter=diameter / spec_norm, GU=GU_init, trace_G=traceG, spec_norm=spec_norm, Dinv=Dinv)
    v_plus = None
        
    # Logging
    if return_logging:
        logging = {"iters": [], "times": [], "f_vals": [], "LMO_errors": [], "steps": [],
                    "rank": [], "rank_ws": [], "eig_residual_max": [], "eig_residual_mean": [], 
                    "num_matvecs": [], "projection_time": [], "theta": []}
    else:
        logging = {}

    cum_time = 0.0
    total_start_time = time.perf_counter() if measure_timing else None

    # Termination for stagnating gap
    gap_history = []
    gap_stagnate_tol = 1e-8
    gap_stagnate_window = 5

    # Main Loop
    for it in range(maxit):

        t0 = time.perf_counter() if measure_timing else None

        # FW direction 
        old_v_plus = v_plus
        v_plus = compute_v_plus(state, old_v_plus)

        # Compute FW gap
        LMO_error = state.compute_fw_gap(v_plus)
        gap_history.append(float(LMO_error))
        if len(gap_history) > gap_stagnate_window and it > min(100, maxit// 2):
            gap_history.pop(0)
            gap_changes = [abs(gap_history[i] - gap_history[i-1]) / (abs(gap_history[i-1]) + 1e-20) for i in range(1, len(gap_history))]
            if all(change < gap_stagnate_tol for change in gap_changes):
                # if verbose:
                print(f"Terminating early at iter {it} due to stagnating FW gap.")
                break
        
        # Compute v_minus and lambda_t for drop step
        v_minus, c_minus = compute_v_minus_rank_r(state)
        lambda_t = 1.0 / max(state.quad_pinv(v_minus), _REG_EPS)

        # Check if drop step is feasible and sufficient decrease
        drop_taken = False
        if lambda_t < 1.0 and state.U.shape[1] > 1:
            # Find the column most aligned with v_minus
            projections = [cp.abs(cp.dot(state.U[:, j], v_minus)) for j in range(state.U.shape[1])]
            drop_idx = int(cp.argmax(cp.array(projections)))
            U_new = cp.delete(state.U, drop_idx, axis=1)
            GU_new = cp.delete(state.GU, drop_idx, axis=1)
            alpha_to_drop = state.alpha[drop_idx]
            alpha_new = cp.delete(state.alpha, drop_idx)
            alpha_new = alpha_new / (1 - alpha_to_drop)

            tmp = State(G=G, U=U_new, alpha=alpha_new, n=n, diameter=diameter / spec_norm, GU=GU_new, trace_G=traceG, spec_norm=spec_norm, Dinv=Dinv)
            if tmp.objective() <= state.objective():
                state = tmp
                drop_taken = True
                step_name = "drop"
                fval = state.objective()
    
        # If drop step not taken, evaluate FW, AFW, PFW
        if not drop_taken:
            # FW update
            den = state.compute_nDtsq(v_plus)
            if den <= _REG_EPS:
                eta_fw = 0.0
            else:
                eta_fw = LMO_error / den
            eta_fw = np.clip(eta_fw, 0, 1.0)
            plus_idx = find_existing_atom(state.U, v_plus, tol=1e-4, Dinv_sq=Dinv_sq)
            U_fw, alpha_fw, GU_fw = fw_update(U=state.U, alpha=state.alpha, v_plus=v_plus, eta=eta_fw, plus_idx=plus_idx, G=G, GU=state.GU, Dinv_sq=Dinv_sq)
            state_fw = State(G=G, U=U_fw, alpha=alpha_fw, n=n, diameter=diameter / spec_norm, GU=GU_fw, trace_G=traceG, spec_norm=spec_norm, Dinv=Dinv)
            f_fw = state_fw.objective()

            # AFW (rank-r)
            f_afw = cp.inf
            state_afw = None
            if state.U.shape[1] > 0:
                eta_afw = LMO_error / (
                    LMO_error + state.compute_nDtsq(v_minus) + _REG_EPS
                )
                U_afw, alpha_afw, GU_afw = afw_multi_update(state.U, state.alpha, c_minus, eta_afw, lambda_t=lambda_t, GU=state.GU, drop_tol=_REG_EPS)
                state_afw = State(G=G, U=U_afw, alpha=alpha_afw, n=n, diameter=diameter / spec_norm, GU=GU_afw, trace_G=traceG, spec_norm=spec_norm, Dinv=Dinv)
                f_afw = state_afw.objective()

            # PFW (rank-r)
            f_pfw = cp.inf
            state_pfw = None
            u_minus, c_minus = sample_u_minus(state)

            gamma_t = 1.0 / max(
                state.quad_pinv(u_minus.reshape(-1,1)),
                _REG_EPS
            )
            u_plus = compute_u_plus(state, u_minus, gamma_t)

            plus_idx = find_existing_atom(state.U, u_plus, tol=1e-4, Dinv_sq=Dinv_sq)
            U_pfw, alpha_pfw, GU_pfw = pfw_multi_update(
                state.U, state.alpha, c_minus, u_plus, gamma_t, plus_idx=plus_idx, G=G, GU=state.GU, Dinv_sq=Dinv_sq)
            state_pfw = State(G=G, U=U_pfw, alpha=alpha_pfw, n=n, diameter=diameter / spec_norm, GU=GU_pfw, trace_G=traceG, spec_norm=spec_norm, Dinv=Dinv)
            f_pfw = state_pfw.objective()

            candidates = [
                ("PFW", f_pfw, state_pfw, U_pfw, alpha_pfw, GU_pfw),
                ("FW", f_fw, state_fw, U_fw, alpha_fw, GU_fw),
                ("AFW", f_afw, state_afw, U_afw, alpha_afw, GU_afw)
                
            ]
            step_name, fval, state, U_new, alpha_new, GU_new = min(candidates, key=lambda x: x[1])

        
        # compute time and logs
        elapsed = None
        if measure_timing:
            if sync_for_timing:
                cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter() - t0
            cum_time += elapsed
        # if verbose:
        #     print(f"Iter {it}: f_val={fval:.12e}, gap={LMO_error:.6e}, step={step_name}, rank={state.U.shape[1]}")

        # Save logs
        if return_logging:
            logging["iters"].append(it)
            if measure_timing:
                logging["times"].append(cum_time)
            logging["f_vals"].append(fval)
            logging["LMO_errors"].append(LMO_error)
            logging["steps"].append(step_name)
            logging["rank"].append(state.U.shape[1])
            if measure_timing:
                logging["projection_time"].append(cum_time)
        
        eps = epsilon_func((state.U, state.alpha), scaled_start_value, outer_iter_num)

        if LMO_error <= eps:
            break

    # Scale the eigenvectors by preconditioner to recover the original problem solution
    if preprocess_input:
        U = Dinv[:, None] * state.U
    else:
        U = state.U


    # Print last LMO error, last projection tolerance epsilon, and number of projection iterations
    last_LMO_error = LMO_error # logging["LMO_errors"][-1] if logging["LMO_errors"] else None
    last_eps = eps #  epsilon_func((state.U, state.alpha), scaled_start_value, outer_iter_num)
    num_iters = it + 1
    if verbose:
        print(f"[spectrahedron_fw_projection] Last LMO error: {last_LMO_error}")
        print(f"[spectrahedron_fw_projection] Last projection tolerance epsilon: {last_eps}")
        print(f"[spectrahedron_fw_projection] Number of projection iterations: {num_iters}")

        # Print rank of last iterate
        print(f"[spectrahedron_fw_projection] Rank of last iterate: {U.shape[1]}")

    if measure_timing:
        if sync_for_timing:
            cp.cuda.get_current_stream().synchronize()
        logging["total_time"] = time.perf_counter() - total_start_time

    if return_logging:
        logging["last_LMO_error"] = last_LMO_error
        logging["last_eps"] = last_eps
        logging["num_iters"] = num_iters

        if generate_spectral_warm_start:
            logging["rank_ws"] = logging_ws["rank_ws"]
            # logging["eig_residual_max"] = logging_ws["max_eig_residual"]
            # logging["eig_residual_mean"] = logging_ws["mean_eig_residual"]
            logging["theta"] = logging_ws["theta"]

    return (U, state.alpha * diameter), logging


# =========================================================
# Vanilla Frank-Wolfe (same signature)
# =========================================================
def vanilla_fw_projection(
    G,
    start_value,
    epsilon_func,
    outer_iter_num,
    maxit,
    line_search=True,
    diameter: float = 1.0,
    return_logging: bool = False,
    verbose: bool = False,
    preprocess_input: bool = False,
    generate_spectral_warm_start: bool = False,
    timing: bool = False,
    sync_gpu_for_timing: bool = False,
):
    measure_timing = bool(timing)
    sync_for_timing = bool(sync_gpu_for_timing)
    """
    Plain Frank-Wolfe for the projection objective. Uses exact leading eigenvector of -grad.
    Matches signature of spectrahedron_fw_projection.
    """
    # start_value: (U_init, alpha_init), G: LinearOperator
    U_init, alpha_init = start_value
    n = U_init.shape[0]
    D, traceG = estimate_diag_and_trace(G, n)

    if generate_spectral_warm_start:
        k_init = max(U_init.shape[1], 20)
        U_init, alpha_init, logging_ws = spectral_warm_start(
                                            A=G,
                                            n=n,
                                            R=diameter,
                                            U_prev=U_init,
                                            alpha_prev=alpha_init,
                                            outer_iter_num=outer_iter_num,
                                            k_init=k_init,
                                            k_step=20,
                                            k_cap=None,
                                            sws_tol=1e-2,
                                            boundary_tol=1e-4,
                                            maxiter=None,
                                            backend="auto",
                                            verbose=False,
                                            timing=measure_timing,
                                            sync_gpu_for_timing=sync_for_timing,
                                        )

    alpha_init /= diameter  # Rescale alpha to account for diameter
    spec_norm = 1.0
    Dinv = None
    Dinv_sq = None

    if preprocess_input:
        traceG = 0 # disable trace shift for now
        Dinv = 1 / D
        Dinv_sq = Dinv * Dinv
        orig_mv = G.matvec
        orig_rmv = G.rmatvec

        def G_centered_mv(x):
            return orig_mv(x) - (traceG/n) * x

        def G_centered_rmv(x):
            return orig_rmv(x) - (traceG/n) * x

        G_shifted = LinearOperator((n, n), matvec=G_centered_mv, rmatvec=G_centered_rmv) 

        G_cond = make_preconditioned_operator(G_shifted, D, n)
        G_cond_mv = G_cond.matvec
        G_cond_rmv = G_cond.rmatvec

        spec_norm = estimate_spectral_norm(G_shifted, n)
        # print('spec_norm:', spec_norm)

        def G_scaled_mv(x):
            return G_cond_mv(x) / spec_norm

        def G_scaled_rmv(x):
            return G_cond_rmv(x) / spec_norm

        G = LinearOperator((n, n), matvec=G_scaled_mv, rmatvec=G_scaled_rmv)

    # Scale U
    if preprocess_input and Dinv is not None:
        U_init = D[:, None] * U_init
        scaled_start_value = (D[:, None] * start_value[0],  start_value[1] / diameter)
    else:
        scaled_start_value = (start_value[0],  start_value[1] / diameter)

    # Cache GU 
    if preprocess_input and Dinv is not None:
        GU_init = cp.column_stack([G.matvec(Dinv_sq * U_init[:, i]) for i in range(U_init.shape[1])])
    else:
        GU_init = cp.column_stack([G.matvec(U_init[:, i]) for i in range(U_init.shape[1])])

    # Initialize from U_init, alpha_init
    state = State(G=G, U=U_init, alpha=alpha_init, n=n, diameter=diameter / spec_norm, GU=GU_init, trace_G=traceG, spec_norm=spec_norm, Dinv=Dinv)
    v_plus = None
        
    # Logging
    if return_logging:
        logging = {"iters": [], "times": [], "f_vals": [], "LMO_errors": [],
                    "rank": [], "rank_ws": [], "eig_residual_max": [], "eig_residual_mean": [], 
                    "num_matvecs": [], "projection_time": [], "theta": []}
    else:
        logging = {}

    cum_time = 0.0
    total_start_time = time.perf_counter() if measure_timing else None

    # Termination for stagnating gap
    gap_history = []
    gap_stagnate_tol = 1e-8
    gap_stagnate_window = 5

    for it in range(maxit):
        t0 = time.perf_counter() if measure_timing else None

        # FW direction 
        old_v_plus = v_plus
        v_plus = compute_v_plus(state, old_v_plus)

        # Print first few entries of v_plus for debugging
        # if verbose:
        #     print(f"  v_plus (first 5 entries): {v_plus[:5].get()}")

        # Compute FW gap
        LMO_error = state.compute_fw_gap(v_plus)
        gap_history.append(float(LMO_error))
        if len(gap_history) > gap_stagnate_window:
            gap_history.pop(0)
            gap_changes = [abs(gap_history[i] - gap_history[i-1]) / (abs(gap_history[i-1]) + 1e-20) for i in range(1, len(gap_history))]
            if all(change < gap_stagnate_tol for change in gap_changes):
                if verbose:
                    print(f"Terminating early at iter {it} due to stagnating FW gap.")
                break

        # Compute eta_fw
        den = state.compute_nDtsq(v_plus)
        if line_search:
            if den <= _REG_EPS:
                eta_fw = 0.0
            else:
                eta_fw = LMO_error / den
        else:
            eta_fw = 2 / (it + 1)

        eta_fw = np.clip(eta_fw, 0, 1.0)
        plus_idx = find_existing_atom(state.U, v_plus, tol=1e-4, Dinv_sq=Dinv_sq)
        U_fw, alpha_fw, GU_fw = fw_update(U=state.U, alpha=state.alpha, v_plus=v_plus, eta=eta_fw, plus_idx=plus_idx, G=G, GU=state.GU, Dinv_sq=Dinv_sq)
        state.rebuild(U_fw, alpha_fw, GU_fw)

        # compute time and logs
        elapsed = None
        if measure_timing:
            if sync_for_timing:
                cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter() - t0
            cum_time += elapsed

        # compute objective for logging
        fcur = state.objective() 
        
        if verbose and measure_timing:
            print(f'Iter {it}: f_val={fcur:.6e}, LMO_Error={LMO_error:.6e}, time={elapsed:.4f} s, rank={state.U.shape[1]}')

        # Save logs
        if return_logging:
            logging["iters"].append(it)
            if measure_timing:
                logging["times"].append(cum_time)
            logging["f_vals"].append(fcur)
            logging["LMO_errors"].append(LMO_error)
            logging["rank"].append(state.U.shape[1])
            if measure_timing:
                logging["projection_time"].append(cum_time)
        
        eps = epsilon_func((state.U, state.alpha), scaled_start_value, outer_iter_num)

        if LMO_error <= eps:
            break

    # Scale the eigenvectors by preconditioner to recover the original problem solution
    if preprocess_input:
        U = Dinv[:, None] * state.U
    else:
        U = state.U


    # Print last LMO error, last projection tolerance epsilon, and number of projection iterations
    last_LMO_error = logging["LMO_errors"][-1] if logging["LMO_errors"] else None
    last_eps = epsilon_func((state.U, state.alpha), scaled_start_value, outer_iter_num)
    num_iters = it + 1
    print(f"[vanilla_fw_projection] Last LMO error: {last_LMO_error}")
    print(f"[vanilla_fw_projection] Last projection tolerance epsilon: {last_eps}")
    print(f"[vanilla_fw_projection] Number of projection iterations: {num_iters}")


    # Print rank of last iterate
    print(f"[vanilla_fw_projection] Rank of last iterate: {U.shape[1]}")

    if measure_timing:
        if sync_for_timing:
            cp.cuda.get_current_stream().synchronize()
        logging["total_time"] = time.perf_counter() - total_start_time

    if measure_timing:
        if sync_for_timing:
            cp.cuda.get_current_stream().synchronize()
        logging["total_time"] = time.perf_counter() - total_start_time

    if return_logging:
        logging["last_LMO_error"] = last_LMO_error
        logging["last_eps"] = last_eps
        logging["num_iters"] = num_iters

        if generate_spectral_warm_start:
            logging["rank_ws"] = logging_ws["rank_ws"]
            # logging["eig_residual_max"] = logging_ws["max_eig_residual"]
            # logging["eig_residual_mean"] = logging_ws["mean_eig_residual"]
            logging["theta"] = logging_ws["theta"]

    return (U, state.alpha * diameter), logging



# =========================================================
# Low-rank state 
# X = D * sum alpha_i u_i u_i^T
# with sum(alpha)=1
# =========================================================
class State:

    def __init__(self, G, U, alpha, n, GU, trace_G=None, diameter=1.0, spec_norm=1.0, Dinv=None):

        self.G = G
        self.n = n
        self.diameter = diameter   # <-- scaling D
        self.spec_norm = spec_norm
        self.Dinv = None
        self.D = None
        if Dinv is not None:
            self.D = 1 / Dinv
            self.Dinv = Dinv
            self.Dinv_sq = self.Dinv * self.Dinv
            self.G_original = make_preconditioned_operator(G, self.Dinv, n)
        else:
            self.Dinv = None
        self.trace_G = trace_G
        if self.trace_G is not None:        
            self.trace_shift = trace_G / n

        self.rebuild(U, alpha, GU)

    # -----------------------------------------------------
    # rebuild state
    # -----------------------------------------------------
    def rebuild(self, U, alpha, GU):

        # normalize atoms
        if self.Dinv is not None:
            norms = cp.sqrt(cp.sum((self.Dinv_sq[:, None] * U) * U, axis=0)) + 1e-20
        else:
            norms = cp.linalg.norm(U, axis=0) + 1e-20
        U = U / norms
        GU = GU / norms
        alpha = alpha * norms**2

        # enforce trace
        alpha *= 1.0 / cp.sum(alpha)

        self.U = U
        self.alpha = alpha
        self.GU = GU

        if U.shape[1] == 0:
            self.sqrt_alpha = cp.zeros((0,))
            self.GU = cp.zeros((self.n, 0))
            return
        
        self.sqrt_alpha = cp.sqrt(alpha)


    # -----------------------------------------------------
    # Y x = V(V^T x)
    # -----------------------------------------------------
    def D_inv2_H_Dinv2_mv(self, x):
        x_tilde = self.Dinv_sq[:, None] * x
        return self.Dinv_sq[:, None] * (self.H_mv(x_tilde))
    
    def H_mv(self, x):
        V = self.U * self.sqrt_alpha
        if V.shape[1] == 0:
            return cp.zeros_like(x)
        return V @ (V.T @ x)

    # -----------------------------------------------------
    # gradient matvec
    # grad f(X) = X - G = R*Y - G
    # -----------------------------------------------------
    def grad_mv(self, x):
        if self.Dinv is not None:
            return self.diameter * self.D_inv2_H_Dinv2_mv(x) - self.Dinv_sq[:, None] * self.G.matvec(self.Dinv_sq[:, None] * x)
        else:
            return self.diameter * self.H_mv(x) - self.G.matvec(x)

    # -----------------------------------------------------
    # projector onto span(U)
    # -----------------------------------------------------
    def projector_mv(self, x):
        if self.U.shape[1] == 0:
            return cp.zeros_like(x)

        if self.Dinv is not None:
            Gm = self.DinvU.T @ self.DinvU
        else:
            Gm = self.U.T @ self.U
        Ginv = cp.linalg.inv(Gm + 1e-12 * cp.eye(Gm.shape[0]))

        if self.Dinv is not None:
            DinvU = self.Dinv[:, None] * self.U
            return DinvU @ (Ginv @ (DinvU.T @ x))
        else:
            return self.U @ (Ginv @ (self.U.T @ x))

    # -----------------------------------------------------
    # quadratic form with pseudoinverse
    # -----------------------------------------------------
    def quad_pinv(self, x):

        if self.U.shape[1] == 0:
            return 0.0
        
        if self.Dinv is not None:
            DinvU = self.Dinv[:, None] * self.U

        if self.Dinv is not None: 
            Gm = DinvU.T @ DinvU
        else:
            Gm = self.U.T @ self.U
        Ginv = cp.linalg.inv(Gm + 1e-12 * cp.eye(Gm.shape[0]))

        if self.Dinv is not None:
            y = Ginv @ (DinvU.T @ x)
        else:
            y = Ginv @ (self.U.T @ x)
        return float((y.T @ y).item())

    # -----------------------------------------------------
    # objective
    #
    # f(X)=0.5(||X||² - 2<tr(G,X)>)
    # X = D Y
    #
    # ||X||² = R² ||Y||²
    # <G,X> = D <G,Y>
    # -----------------------------------------------------
    def objective(self):

        if self.Dinv is not None:

            # trace_DinvY2 = cp.sum(self.GmDinv**2)
            DinvU =self.Dinv[:, None] * self.U
            DinvV = DinvU * self.sqrt_alpha
            M = DinvV.T @ DinvV
            trace_DinvY2 = cp.sum(M * M)
            del M

            if self.U.shape[1] == 0:
                trace_Dinv2YDinv2G = 0.0
            else:
                trace_Dinv2YDinv2G = cp.sum(
                    self.alpha * cp.sum(DinvU * (self.Dinv[:, None] * self.GU), axis=0)
                )

            R = self.diameter

            # objective in transformed problem
            obj = 0.5 * self.spec_norm ** 2 * float(R**2 * trace_DinvY2 - 2.0 * R * trace_Dinv2YDinv2G)

            # restore trace component
            tau = self.trace_shift

            correction = -2 * tau * R

            return obj + correction
        else:
        
            # trace_Y2 = cp.sum(self.Gm**2)
            V = self.U * self.sqrt_alpha
            M = V.T @ V
            trace_Y2 = cp.sum(M * M)
            del M

            if self.U.shape[1] == 0:
                trace_GY = 0.0
            else:
                trace_GY = cp.sum(
                    self.alpha * cp.sum(self.U * self.GU, axis=0)
                )

            R = self.diameter

            # objective in transformed problem
            obj = 0.5 * self.spec_norm**2 * float(R**2 * trace_Y2 - 2.0 * R * trace_GY)

            # restore trace component
            if self.trace_G is not None:
                tau = self.trace_shift
                correction = - 2 * tau * R

                return obj + correction
            else:
                return obj
            

    # -----------------------------------------------------
    # FW gap
    #
    # gap = <X-G, X - R ss^T> with X = R Y
    # 
    # --------------------------------------
    def compute_fw_gap(self, s):

        R = self.diameter

        if self.Dinv is not None:

            Dinv_s = self.Dinv[:, None] * s

            # trace_DinvY2 = cp.sum(self.GmDinv**2)
            DinvU =self.Dinv[:, None] * self.U
            DinvV = DinvU * self.sqrt_alpha
            M = DinvV.T @ DinvV
            trace_DinvY2 = cp.sum(M * M)
            del M

            if self.U.shape[1] == 0:
                trace_Dinv2YDinv2G = 0.0
                HS = 0.0
            else:
                trace_Dinv2YDinv2G = cp.sum(
                    self.alpha * cp.sum(DinvU * (self.Dinv[:, None] * self.GU), axis=0)
                )
                # trace_Dinv2YDinv2G = cp.sum(
                #     self.alpha * cp.sum(self.DinvU * self.DinvGDinvU, axis=0)
                # )

                Vs = DinvV.T @ Dinv_s
                HS = cp.sum(Vs**2)      # = s^T Y s

            try:
                GS = Dinv_s.T @ self.G_original.matvec(Dinv_s)
            except Exception:
                GS = Dinv_s.T @ self.G_original.matvec(Dinv_s.ravel())

            gap = R**2 * trace_DinvY2 - R * trace_Dinv2YDinv2G - R**2 * HS + R * GS
        else:

            # trace_Y2 = cp.sum(self.Gm ** 2)
            V = self.U * self.sqrt_alpha
            M = V.T @ V
            trace_Y2 = cp.sum(M * M)
            del M    

            if self.U.shape[1] == 0:
                trace_GY = 0.0
                HS = 0.0
            else:
                trace_GY = cp.sum(
                        self.alpha * cp.sum(self.U * self.GU, axis=0)
                    )

                Vs = V.T @ s
                HS = cp.sum(Vs**2)      # = s^T Y s
            try:
                GS = s.T @ self.G.matvec(s)
            except Exception:
                GS = s.T @ self.G.matvec(s.ravel())

            gap = R**2 * trace_Y2 - R * trace_GY - R**2 * HS + R * GS

        return float(gap.item())

    # -----------------------------------------------------
    # denominator for FW line-search
    #
    # ||X - R ss^T||²
    # = R² ||Y - ss^T||²
    # -----------------------------------------------------
    def compute_nDtsq(self, s):

        R = self.diameter

        if self.Dinv is not None:

            Dinv_s = self.Dinv[:, None] * s

            # trace_DinvY2 = cp.sum(self.GmDinv**2)
            DinvV =self.Dinv[:, None] * self.U * self.sqrt_alpha
            M = DinvV.T @ DinvV
            trace_DinvY2 = cp.sum(M * M)
            del M

            Vs = DinvV.T @ Dinv_s
            HS = cp.sum(Vs**2)

            norm_s = float((Dinv_s.T @ Dinv_s).item())
            return R**2 * (trace_DinvY2 - 2.0 * HS + norm_s**2)
        else:
            # trace_Y2 = cp.sum(self.Gm ** 2)
            V = self.U * self.sqrt_alpha
            M = V.T @ V
            trace_Y2 = cp.sum(M * M)
            del M    

            Vs = V.T @ s
            HS = cp.sum(Vs**2)

            return float(
                R**2 * (trace_Y2 + 1.0 - 2.0 * HS)
            )

# =========================================================
# Leading eigenvector of M
# =========================================================
def leading_eigvec_minus(
    M,
    maxiter=500,
    tol=1e-8,
    dtype=cp.float32,
    reorth=True,
    random_state=None,
    old_v_plus=None
):
    """
    Compute the eigenvector corresponding to the smallest eigenvalue of M.
    Supports:
      - CuPy dense arrays
      - cupyx sparse matrices
      - LinearOperator-style matvecs (Lanczos only)
    """
    n = M.shape[0]

    # ---- Small dense matrix path ----
    if n <= 5 and hasattr(M, "toarray"):
        A = M.toarray()
        vals, vecs = cp.linalg.eigh(A)
        v = vecs[:, 0]
        return v / (cp.linalg.norm(v) + 1e-20)

    # ---- Lanczos path (works for all types) ----
    if old_v_plus is not None:
        v0 = old_v_plus.ravel()
        v0_copy = v0.copy()
    else:
        v0 = None
        v0_copy = None
    try:       
        eigval, eigvec = cusplinalg.eigsh(M, k = 1, which='LA', tol=tol,v0=v0, maxiter=maxiter)
        v = eigvec
        if cp.isnan(v).any():
            raise ValueError("NaN encountered in eigenvector from Lanczos Method.")

    except Exception as e:
        eigval, eigvec = power_method(M, n=n, max_iter=maxiter, tol=tol, old_v_plus=v0_copy)
        v = eigvec
        if cp.isnan(v).any():
            raise ValueError("NaN encountered in eigenvector from Power Method.")
    return v / (cp.linalg.norm(v) + 1e-20)

# =========================================================
# Power method for leading eigenvector (used as fallback if Lanczos fails)
# =========================================================
def power_method(M, n, max_iter=500, tol=1e-8, old_v_plus=None):
    def matvec(x):
        try:
            return M.dot(x) if hasattr(M, "dot") else M @ x
        except Exception:
            return M.dot(x.ravel()) if hasattr(M, "dot") else M @ x.ravel()
    
    if old_v_plus is None:
        y = cp.random.randn(n,1)
    else:
        y = old_v_plus.copy()

    # y = cp.random.randn(n)
    y /= (cp.linalg.norm(y) + 1e-20)

    eigval_old = 0.0

    for _ in range(max_iter):
        v = matvec(y)
        nrm = cp.linalg.norm(v)
        if nrm == 0:
            raise ValueError("Zero vector encountered.")

        y = v / (nrm + 1e-20)
        # eigval = cp.dot(y, matvec(y))
        eigval = cp.dot(y.ravel(), matvec(y.ravel()))

        if cp.abs(eigval - eigval_old) < tol:
            break

        eigval_old = eigval

    return eigval, y

# =========================================================
# Find rough spectral norm of a LinearOperator using power method
# =========================================================
def estimate_spectral_norm(G, n, max_iter=20, tol=1e-4):

    # largest eigenvalue
    lambda_max, _ = power_method(G, n, max_iter=max_iter, tol=tol)

    # smallest eigenvalue via -G
    class NegOperator:
        def __init__(self, G):
            self.G = G
            self.shape = G.shape
        def dot(self, x):
            return -self.G.dot(x)

    lambda_min_neg, _ = power_method(NegOperator(G), n, max_iter=max_iter, tol=tol)

    lambda_min = -lambda_min_neg

    return float(max(abs(lambda_max), abs(lambda_min)))

# ========================================================
# v_plus
# ========================================================
def compute_v_plus(state, old_v_plus=None):
    n = state.n

    # Scale gradient by preconditioner if needed
    if state.D is not None:
        def scaled_grad_mv(x):
            x = x.reshape(-1,1)
            y = state.grad_mv(state.D[:, None] * x)
            return state.D[:, None] * y
    else:
        scaled_grad_mv = state.grad_mv


    # FW atom: leading eigenvector of -grad
    A = cusplinalg.LinearOperator((n, n), matvec=scaled_grad_mv)
    v_plus = leading_eigvec_minus(-A, old_v_plus=old_v_plus)
    if state.D is not None:
        v_plus = state.D[:, None] * v_plus.reshape(-1, 1)
    return v_plus.reshape(-1, 1)   # force column vector

# =========================================================
# FW update
# =========================================================

def fw_update(U, alpha, v_plus, eta, G, GU, plus_idx=None, drop_tol=1e-14, Dinv_sq=None):
    """
    Frank-Wolfe update for low-rank state: (U, alpha)
    """
    alpha_new = (1 - eta) * alpha
    U_new = U.copy()
    if plus_idx is None:
        # Add new atom
        U_new = cp.hstack([U_new, v_plus.reshape(-1, 1)])
        try:
            if Dinv_sq is not None:
                GU_new  = cp.hstack([GU, G.matvec(Dinv_sq[:, None] * v_plus).reshape(-1, 1)])
            else:
                GU_new = cp.hstack([GU, G.matvec(v_plus).reshape(-1, 1)])
        except Exception:
            # Fallback if G.matvec fails
            if Dinv_sq is not None:
                GU_new  = cp.hstack([GU, G.matvec(Dinv_sq[:, None] * v_plus.ravel()).reshape(-1, 1)])
            else:
                GU_new = cp.hstack([GU, G.matvec(v_plus.ravel()).reshape(-1, 1)])
        alpha_new = cp.concatenate([alpha_new, cp.array([eta])])
    else:
        # Existing atom receives FW mass
        alpha_new[plus_idx] += eta
        GU_new = GU.copy()
    # Remove atoms with very small alpha
    keep = alpha_new > drop_tol
    U_new = U_new[:, keep]
    GU_new = GU_new[:, keep]
    alpha_new = alpha_new[keep]
    # Renormalize trace if needed
    trace = cp.sum(alpha_new)
    if trace > 0:
        alpha_new = alpha_new / trace
    return U_new, alpha_new, GU_new
    

# =========================================================
# Compute gradient norm (not used anymore - replaced by compute_v_minus_rank_r)
# =========================================================
def compute_grad_norm(state):
    A = cusplinalg.LinearOperator(
        (n, n),
        matvec=state.grad_mv
    )

    val, _ = cusplinalg.eigsh(A, k=1, which='LM')

    return float(cp.abs(val[0]))

# =========================================================
# v_minus (not used anymore - replaced by compute_v_minus_rank_r)
# =========================================================
def compute_v_minus(state, zeta=0.1):
    grad_norm = compute_grad_norm()

    def B_mv(x):
        u = state.projector_mv(x)
        g = state.grad_mv(u)
        p = state.projector_mv(g)
        return p + (1 + zeta) * grad_norm * u

    A = cusplinalg.LinearOperator((n, n), matvec=B_mv)
    return leading_eigvec_minus(A).reshape(-1, 1)   # force column vector

# =========================================================
# u_minus (for PFW)
# =========================================================
def sample_u_minus(state):
    """
    Random direction inside Im(X_t).
    """
    r = state.U.shape[1]

    if r == 0:
        return cp.zeros((n,1)), None

    # random coefficients
    c = cp.random.randn(r)

    # normalize in subspace metric
    V = state.U * state.sqrt_alpha
    B = V.T @ V # state.Gm
    norm = cp.sqrt(c.T @ B @ c)

    c /= (norm + 1e-20)

    u = V @ c

    return u.reshape(-1,1), c

# =========================================================
# u_plus (for PFW)
# =========================================================
def compute_u_plus(state, u_minus, gamma):
    n = state.n
    def M_mv(x):
        x = x.reshape(-1,1)
        return gamma * (u_minus @ (u_minus.T @ x)) - state.grad_mv(x)

    A = cusplinalg.LinearOperator((n, n), matvec=M_mv)
    return leading_eigvec_minus(A)

# ========================================================= 
# Compute reduced matrices for solving for v_minus
# =========================================================
def build_reduced_matrices(state):
    """
    Build rank-r matrices:
        A = V^T grad V
        B = V^T V
    """
    V = state.U * state.sqrt_alpha
    r = V.shape[1]
    if state.Dinv is not None:
        DinvV = state.Dinv[:, None] * state.U * state.sqrt_alpha
        # DinvV = state.DinvV

    if r == 0:
        return None, None, None

    GV = state.GU * state.sqrt_alpha

    if state.Dinv is not None:
        DinvV = state.Dinv[:, None] * V
        B = DinvV.T @ DinvV # state.GmDinv                # V^T D^{-2} V
    else:
        B = V.T @ V # state.Gm                    # V^T V

    if state.Dinv is not None:
        A = B @ B - DinvV.T @ (state.Dinv[:, None] * GV)     #  V^T D^{-2} V - V^T D^{-2} G D^{-2} V with preconditioning
    else:
        A = B @ B - V.T @ GV            # V^T (X-G) V

    return A, B, GV

# =========================================================
# v_minus (for drop and AFW)
# =========================================================
def compute_v_minus_rank_r(state):

    V = state.U * state.sqrt_alpha

    A, B, _ = build_reduced_matrices(state)

    M = cp.linalg.solve(B, A)
    vals, vecs = cp.linalg.eigh(M)

    c = vecs[:, -1]      # largest eigenvalue

    v_minus = V @ c
    v_minus = v_minus.reshape(-1, 1)   # force column vector
    v_minus /= cp.linalg.norm(v_minus) + 1e-20

    if state.Dinv is not None:
        v_minus = state.D[:, None] * v_minus

    return v_minus, c

# =========================================================
# AFW update (multi-atom)
# =========================================================
def afw_multi_update(U, alpha, c_minus, eta, lambda_t, GU, drop_tol=1e-12):
    """
    Away-step Frank-Wolfe update for low-rank state: (U, alpha)
    """
    w = alpha * (c_minus ** 2)
    w /= cp.sum(w) + 1e-20
    eta_weights = float(cp.min(alpha / (w + _REG_EPS)))
    eta_max = min(lambda_t, eta_weights)
    eta = min(eta, eta_max - _REG_EPS)
    alpha_new = (alpha - eta * w) / (1 - eta)
    # Remove atoms with very small alpha
    keep = alpha_new > drop_tol
    U_new = U[:, keep]
    GU_new = GU[:, keep]
    alpha_new = alpha_new[keep]
    # Renormalize trace if needed
    trace = cp.sum(alpha_new)
    if trace > 0:
        alpha_new = alpha_new / trace
    return U_new, alpha_new, GU_new

# =========================================================
# PFW update (multi-atom)
# =========================================================
def pfw_multi_update(
    U,
    alpha,
    c_minus,
    u_plus,
    gamma,
    G,
    GU,
    plus_idx=None,
    drop_tol=1e-12,
    Dinv_sq = None
):
    """
    Pairwise Frank-Wolfe update for low-rank state: (U, alpha)
    """
    w = c_minus ** 2
    w /= (cp.sum(w) + _REG_EPS)
    gamma_max = cp.min(alpha / (w + _REG_EPS))
    gamma = min(gamma, float(gamma_max))
    alpha_new = alpha - gamma * w
    U_new = U.copy()
    GU_new = GU.copy()
    if plus_idx is None:
        # Add new atom
        U_new = cp.hstack([U_new, u_plus.reshape(-1, 1)])
        try:
            if Dinv_sq is not None:
                GU_new = cp.hstack([GU_new, G.matvec(Dinv_sq[:, None] * u_plus.reshape(-1, 1))])
            else:
                GU_new = cp.hstack([GU_new, G.matvec(u_plus.reshape(-1, 1))])
        except Exception:
            # Fallback if G.matvec fails
            if Dinv_sq is not None:
                GU_new = cp.hstack([GU_new, G.matvec(Dinv_sq[:, None] * u_plus.ravel()).reshape(-1, 1)])
            else:
                GU_new = cp.hstack([GU_new, G.matvec(u_plus.ravel()).reshape(-1, 1)])
        alpha_new = cp.concatenate([alpha_new, cp.array([gamma])])
    else:
        # Add mass to existing atom
        alpha_new[plus_idx] += gamma
    # Remove atoms with very small alpha
    keep = alpha_new > drop_tol
    U_new = U_new[:, keep]
    alpha_new = alpha_new[keep]
    GU_new = GU_new[:, keep]
    # Renormalize trace if needed
    trace = cp.sum(alpha_new)
    if trace > 0:
        alpha_new = alpha_new / trace
    return U_new, alpha_new, GU_new

# ========================================================
# Column rescaling to maintain trace constraint during multi-atom updates
# ========================================================
def rescale_columns(V, alphas_new, drop_tol=1e-12):
    """
    Rescale columns so ||V_i||^2 = alphas_new[i]
    """

    alphas_old = cp.sum(V**2, axis=0)

    # keep = alphas_new > drop_tol

    # V_new = V[:, keep]
    V_new = V.copy()
    # alphas_old = alphas_old[keep]
    # alphas_new = alphas_new[keep]

    scales = cp.sqrt(
        alphas_new / (alphas_old + 1e-20)
    )

    V_new *= scales

    return V_new

# ========================================================
# Check if s is already in the span of V (up to tol)
# ========================================================
def find_existing_atom(U, s, tol=1e-3, Dinv_sq=None): 
    if U.shape[1] == 0: 
        return None 
    
    # Compute correlations with existing atoms
    if Dinv_sq is not None:
        # D^{-2}-weighted correlation
        corr = cp.abs((Dinv_sq[:, None] * U).T @ s).ravel()
    else:
        corr = cp.abs(U.T @ s).ravel()
    idx = int(cp.argmax(corr)) 
    
    if corr[idx] > 1 - tol: 
        return idx 
    
    return None


# ========================================================
# Check if V V^T is in the spectrahedron (PSD, trace=1)
# ========================================================
def check_spectrahedron(U, alpha, name="", tol=1e-8):
    """
    Check if X = U diag(alpha) U^T is in the spectrahedron (PSD, trace=1, nonnegative weights)
    """
    trace_X = float(cp.sum(alpha))
    min_alpha = float(cp.min(alpha)) if len(alpha) > 0 else 0.0
    ok_trace = abs(trace_X - 1.0) < tol
    ok_alpha = min_alpha >= -tol
    if not (ok_trace and ok_alpha):
        print(f"[{name}] Spectrahedron violation:")
        print(f"   trace(X) = {trace_X:.6e}")
        print(f"   min alpha = {min_alpha:.6e}")
    return ok_trace and ok_alpha

def estimate_diag_and_trace(G, n, k=20, seed=0):

    cp.random.seed(seed)

    diag = cp.zeros(n)
    trace = 0.0

    for _ in range(k):
        z = cp.random.choice(cp.array([-1.0, 1.0]), size=n)

        Gz = G.matvec(z) if hasattr(G, "matvec") else G @ z

        diag += z * Gz
        trace += z @ Gz

    diag /= k
    trace /= k

    D = 1 /cp.sqrt(cp.abs(diag) + 1e-12)

    return D, float(trace)

def make_preconditioned_operator(G, D, n):

    D = D.reshape(-1,1)

    def matvec(x):
        x = x.reshape(-1, 1)
        return D * G.matvec(D * x)
    
    def rmatvec(x):
        x = x.reshape(-1, 1)
        return D * G.rmatvec(D * x)

    return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec)


def reuse_atoms(U, alpha, k_reuse, strategy="topk", Dinv_sq=None):
    """
    Select k_reuse atoms from previous solution.

    strategy:
        - "topk": largest alpha
        - "uniform": random subset
        - "hybrid": half topk, half random
    """

    r = U.shape[1]

    if k_reuse == 0:
        n = U.shape[0]
        r0 = min(1, n)
        Z = cp.random.randn(n, r0)
        Q, _ = cp.linalg.qr(Z)
        U0 = Q
        alpha0 = (cp.ones(r0) / r0) * n
        return U0, alpha0

    if r <= k_reuse:
        return U, alpha / cp.sum(alpha)

    if strategy == "topk":
        idx = cp.argsort(alpha)[::-1][:k_reuse]

    elif strategy == "uniform":
        idx = cp.random.choice(r, size=k_reuse, replace=False)

    elif strategy == "hybrid":
        k1 = k_reuse // 2
        idx_top = cp.argsort(alpha)[::-1][:k1]
        idx_rand = cp.random.choice(r, size=k_reuse - k1, replace=False)
        idx = cp.unique(cp.concatenate([idx_top, idx_rand]))[:k_reuse]

    else:
        raise ValueError("Unknown reuse strategy")

    U_new = U[:, idx]
    alpha_new = alpha[idx]

    alpha_new /= cp.sum(alpha_new)

    return U_new, alpha_new

def choose_k_reuse(outer_iter, max_rank=50):
    """
    Adaptive reuse size depending on outer iteration.
    """

    if outer_iter < 100:
        return 0   # restart
    elif outer_iter < 1000:
        return int(0.025 * max_rank)
    elif outer_iter < 5000:
        return int(0.05 * max_rank)
    elif outer_iter < 10000:
        return int(0.1 * max_rank)
    elif outer_iter < 20000:
        return int(0.2 * max_rank)
    elif outer_iter < 50000:
        return int(0.5 * max_rank)
    elif outer_iter < 100000:
        return int(0.75 * max_rank)
    else:
        return max_rank

def compress_rank_fw_aware(U, alpha, GU,
                           k_keep=20, 
                           rank_max=50,
                           weight_power=1.0,
                           grad_power=1.0, Dinv_sq=None):
    """
    FW-aware compression:
    - keep important atoms (by alpha + gradient alignment)
    - compress the rest spectrally
    """

    r = U.shape[1]
    if r <= rank_max:
        return U, alpha

    # --------------------------------------------------
    # 1. Compute importance score
    # --------------------------------------------------
    # weight importance
    w_score = alpha ** weight_power

    # gradient alignment: <u_i, G u_i>
    if Dinv_sq is not None:
        g_score = cp.abs(cp.sum((Dinv_sq[:, None] * U) * GU, axis=0)) ** grad_power
    else:
        g_score = cp.abs(cp.sum(U * GU, axis=0)) ** grad_power

    # combine scores (normalize to avoid scale issues)
    w_score /= (cp.sum(w_score) + 1e-20)
    g_score /= (cp.sum(g_score) + 1e-20)

    score = w_score + g_score

    # --------------------------------------------------
    # 2. Select important atoms to KEEP EXACTLY
    # --------------------------------------------------
    k_keep = min(k_keep, r)
    idx_keep = cp.argsort(score)[::-1][:k_keep]

    mask = cp.zeros(r, dtype=bool)
    mask[idx_keep] = True

    U_keep = U[:, mask]
    alpha_keep = alpha[mask]
    GU_keep = GU[:, mask]

    return U_keep, alpha_keep


def truncate_top_k(U, alpha, GU, k):
    if U.shape[1] <= k:
        return U, alpha, GU

    idx = cp.argsort(alpha)[::-1][:k]
    U_new = U[:, idx]
    alpha_new = alpha[idx]
    GU_new = GU[:, idx]

    alpha_new /= cp.sum(alpha_new)
    return U_new, alpha_new, GU_new

def project_simplex(v):
    u = cp.sort(v)[::-1]
    cssv = cp.cumsum(u)
    rho = cp.nonzero(u * cp.arange(1, len(v)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return cp.maximum(v - theta, 0)


def fully_corrective_step(U, alpha, GU, Dinv_sq=None, n_steps=20, step_size=0.1):
    """
    Fully corrective Frank-Wolfe:
    optimize alpha over simplex with fixed atoms U
    """

    if U.shape[1] <= 1:
        return alpha

    # --- Build Q ---
    Gram = U.T @ U
    Q = Gram**2   # (u_i^T u_j)^2

    # --- Build c ---
    if Dinv_sq is not None:
        c = cp.sum((Dinv_sq[:, None] * U) * GU, axis=0)
    else:
        c = cp.sum(U * GU, axis=0)

    # --- PGD on simplex ---
    alpha_new = alpha.copy()

    for _ in range(n_steps):
        grad = Q @ alpha_new - c
        alpha_new = alpha_new - step_size * grad
        alpha_new = project_simplex(alpha_new)

    return alpha_new

def reuse_atoms_gradient_filter(U, alpha, G, k_max=5, Dinv_sq=None):
    """
    Keep only atoms aligned with current gradient.
    """

    if U.shape[1] == 0:
        return U, alpha

    # Compute GU
    if Dinv_sq is not None:
        GU = cp.column_stack([G.matvec(Dinv_sq * U[:, i]) for i in range(U.shape[1])])
    else:
        GU = cp.column_stack([G.matvec(U[:, i]) for i in range(U.shape[1])])

    # Alignment score
    if Dinv_sq is not None:
        scores = cp.sum((Dinv_sq[:, None] * U) * GU, axis=0)
    else:
        scores = cp.sum(U * GU, axis=0)

    # Keep only *positive alignment* atoms
    threshold = 1e-3 * cp.max(scores)
    keep = scores > threshold

    if cp.sum(keep) == 0:
        return None, None  # force restart

    U_keep = U[:, keep]
    alpha_keep = alpha[keep]

    # If too many, keep best ones
    if U_keep.shape[1] > k_max:
        idx = cp.argsort(scores[keep])[::-1][:k_max]
        U_keep = U_keep[:, idx]
        alpha_keep = alpha_keep[idx]

    alpha_keep /= cp.sum(alpha_keep)

    return U_keep, alpha_keep

def truncate_by_true_gap_adaptive(
    U, alpha, GU, R,
    LMO_error, epsilon,
    outer_iter,
    max_rank=None,
    Dinv_sq=None
):
    # --- u_i^T G u_i ---
    if Dinv_sq is not None:
        gG = cp.sum((Dinv_sq[:, None] * U) * GU, axis=0)
    else:
        gG = cp.sum(U * GU, axis=0)

    # --- u_i^T Y u_i ---
    Gram = U.T @ U
    gY = (Gram**2) @ alpha

    # ============================================
    # CORRECT gap-consistent scoring
    # ============================================
    scores = R * gG - R**2 * gY

    # shift to positive (important!)
    scores = scores - cp.min(scores)

    max_score = cp.max(scores)

    # --- adaptive tau ---
    tau = min(0.05, 0.5 * LMO_error / (epsilon + 1e-12))

    keep = scores >= tau * max_score

    # --- minimum rank safeguard ---
    min_rank = 5 if outer_iter < 200 else 10

    if cp.sum(keep) < min_rank:
        idx = cp.argsort(scores)[::-1][:min_rank]
        keep = idx

    U_new = U[:, keep]
    alpha_new = alpha[keep]
    GU_new = GU[:, keep]

    # optional cap
    if max_rank is not None and U_new.shape[1] > max_rank:
        idx = cp.argsort(scores[keep])[::-1][:max_rank]
        U_new = U_new[:, idx]
        alpha_new = alpha_new[idx]
        GU_new = GU_new[:, idx]

    alpha_new /= cp.sum(alpha_new)

    return U_new, alpha_new, GU_new

# ========================================================
# Main experiment
# ========================================================
if __name__ == "__main__":

    cp.random.seed(1)
    
    diameter = 2000

    n = 2000
    r = 10

    B = cp.random.randn(n, r)

    G_dense = B @ B.T
    G_dense = 0.5 * (G_dense + G_dense.T)

    print("G Rank:", cp.linalg.matrix_rank(G_dense))


    # LinearOperator required by FW
    def G_mv(x):
        return G_dense @ x

    G_op = cusplinalg.LinearOperator((n, n), matvec=G_mv, rmatvec=G_mv)

    # ------------------------------------------
    # Exact projection
    # ------------------------------------------
    t0 = time.time()
    w, V = cp.linalg.eigh(G_dense)

    w_sorted = cp.sort(w)[::-1]
    cssv = cp.cumsum(w_sorted)

    rho = cp.where(
        w_sorted - (cssv - diameter) / cp.arange(1,n+1) > 0
    )[0][-1] + 1

    theta = (cssv[rho-1] - diameter) / rho
    w_proj = cp.maximum(w - theta, 0)
    exact_time = time.time() - t0
    print(f"Exact eig-projection: {exact_time:.4f}s")
    H_star = (V * w_proj) @ V.T

    trace_H2 = cp.sum(w_proj**2)
    trace_HG = cp.sum(H_star * G_dense)
    f_star = 0.5 * float(trace_H2 - 2 * trace_HG)
    print('f_star: ', f_star)

    # ------------------------------------------
    # Low-rank initialization (U0, alpha0)
    # ------------------------------------------
    rank_init = 1
    r0 = min(rank_init, n)
    Z = cp.random.randn(n, r0)
    Q, _ = cp.linalg.qr(Z)
    U0 = Q
    alpha0 = (cp.ones(r0) / r0) * diameter
    start_value=(U0, alpha0)

    # epsilon function: we stop when objective is close to f_star (rel tolerance)
    def eps_func(H, start, outer_iter):
        return 1e-8

    max_iters = 1000               

    print("Running Spectrahedron-FW...")
    cp.random.seed(1)
    cp.cuda.Device().synchronize()
    t0_sf = time.time()
    #with cProfile.Profile() as pr:
    (U_sf, alpha_sf), log_sf = spectrahedron_fw_projection(
        G=G_op,
        start_value=start_value,
        epsilon_func=eps_func,
        diameter=diameter,
        outer_iter_num=0,
        maxit=max_iters,
        line_search=False,
        return_logging=True,
        verbose=True,
        preprocess_input=True,
        generate_spectral_warm_start=False,
    )
    cp.cuda.Stream.null.synchronize()
    t1_sf = time.time()

    # stats = pstats.Stats(pr)
    # stats.sort_stats("cumulative").print_stats(10)

    print("Running Vanilla FW...")
    cp.cuda.Device().synchronize()
    t0_vf = time.time()
    # with cProfile.Profile() as pr:
    cp.random.seed(1)
    (U_vw, alpha_vw), log_vw = vanilla_fw_projection(
        G=G_op,
        start_value=start_value,
        epsilon_func=eps_func,
        diameter=diameter,
        outer_iter_num=0,
        maxit=max_iters,
        line_search=False,
        return_logging=True,
        verbose=True,
        preprocess_input=True,
        generate_spectral_warm_start=False,
    )
    cp.cuda.Device().synchronize()
    t1_vf = time.time()
    print(f"Spectrahedron-FW time: {t1_sf - t0_sf:.3f}s")
    print(f"Vanilla-FW time: {t1_vf - t0_vf:.3f}s")
    # stats = pstats.Stats(pr)
    # stats.sort_stats("cumulative").print_stats(10)

    # prepare plot data (convert to numpy for matplotlib)
    it_sf = np.array(log_sf["iters"])
    time_sf = np.array(log_sf["times"])
    fvals_sf = np.array(log_sf["f_vals"])
    gaps_sf = np.array(log_sf["LMO_errors"])

    it_vw = np.array(log_vw["iters"])
    time_vw = np.array(log_vw["times"])
    fvals_vw = np.array(log_vw["f_vals"])
    gaps_vw = np.array(log_vw["LMO_errors"])

    # Convert objective gaps relative to f_star
    obj_res_sf = abs(fvals_sf - f_star) / abs(f_star) if f_star != 0 else abs(fvals_sf - f_star)
    obj_res_vw = abs(fvals_vw - f_star) / abs(f_star) if f_star != 0 else abs(fvals_vw - f_star)

    # Plot iteration-wise (loglog)
    plt.figure(figsize=(10, 5))
    plt.loglog(it_sf + 1, obj_res_sf, label="Spectrahedron-FW")
    plt.loglog(it_vw + 1, obj_res_vw, label="Vanilla-FW")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Residual")
    plt.title("Convergence vs Iteration (log-log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot time-wise (loglog)
    plt.figure(figsize=(10, 5))
    plt.loglog(time_sf + 1e-12, obj_res_sf, label="Spectrahedron-FW")
    plt.loglog(time_vw + 1e-12, obj_res_vw, label="Vanilla-FW")
    plt.xlabel("Cumulative time (s)")
    plt.ylabel("Objective Residual")
    plt.title("Convergence vs Time (log-log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot iteration-wise (loglog)
    plt.figure(figsize=(10, 5))
    plt.loglog(it_sf + 1, gaps_sf, label="Spectrahedron-FW")
    plt.loglog(it_vw + 1, gaps_vw, label="Vanilla-FW")
    plt.xlabel("Iteration")
    plt.ylabel("FW Gap")
    plt.title("Convergence vs Iteration (log-log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Plot time-wise (loglog)
    plt.figure(figsize=(10, 5))
    plt.loglog(time_sf + 1e-12, gaps_sf, label="Spectrahedron-FW")
    plt.loglog(time_vw + 1e-12, gaps_vw, label="Vanilla-FW")
    plt.xlabel("Cumulative time (s)")
    plt.ylabel("FW Gap")
    plt.title("Convergence vs Time (log-log)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Print summary
    print("\nSummary:")
    print(f"Exact projection: f_star = {f_star:.3e}, time = {exact_time:.3f}s")
    print(f"Spectrahedron-FW: iterations = {len(it_sf)}, final gap = {gaps_sf[-1]:.3e}, f_sfw = {fvals_sf[-1]:.3e}, final obj res = {obj_res_sf[-1]:.3e}")
    print(f"Vanilla-FW: iterations = {len(it_vw)}, final gap = {gaps_vw[-1]:.3e}, f_vfw = {fvals_vw[-1]:.3e}, final obj res = {obj_res_vw[-1]:.3e}  ")
