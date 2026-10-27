import re
import os
import glob
import cvxpy
import scipy
import pickle
import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from datautils.parsers.gset_parser import parse_gset
from algorithms.operator_splitting.exact_pdhg_factorized import exact_pdhg_factorized
from algorithms.projections.projection_spectrahedron import projection_spectrahedron
from algorithms.operator_splitting.inexact_pdhg_factorized import inexact_pdhg_factorized
from algorithms.frank_wolfe.spectrahedron_frank_wolfe_low_rank import spectrahedron_fw_projection
from algorithms.projections.projection_spectrahedron_accurate_lanczos import accurate_lanczos_projection

np.random.seed(10)

MAXCUT_OPTIMAL_OBJECTIVE_TABLE = {
    'G1': -12083.0,
    'G6': -2656.0,
    'G11': -629.0,
    'G14': -3191.0,
    'G18': -1166.0,
    'G22': -14136.0,
    'G27': -4141.0, 
    'G32': -1567.0,
    'G35': -8014.0,
    'G39': -2877.0,
    'G43': -7032.0,
    'G48': -6000.0,
    'G51': -4006.0,
    'G55': -11039.0,
    'G57': -3885.0,
    'G59': -7312.0,
    'G60': -15222.0,
    'G62': -5431.0,
    'G64': -10466.0,
    'G65': -6206.0,
    'G66': -7077.0,
    'G67': -7744.0,
    'G70': -9863.0,
    'G72': -7809.0,
    'G77': -11046.0,
    'G81': -15656.0,
}


def default_snapshot_dir(results_path, data_path, method_name):
    graph_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(results_path, 'snapshots', method_name)


def obj_maxcut(X_LR, rows, cols, weights, xp=np):
    """
    -0.25 * Tr(L_w X)

    X = U diag(alpha) U^T
    """

    U, alpha = X_LR

    diff = U[rows] - U[cols]

    return -0.25 * xp.sum(
        weights[:, None]
        * (diff ** 2)
        * alpha[None, :]
    )

def make_grad_operator(rows, cols, weights, xp=np):
    """
    Returns grad matvec:

        v -> (-1/4) L_w v

    Supports v with shape (n,) or (n,k).
    """

    scale = -0.25

    def grad_mv(v):
        out = xp.zeros_like(v, dtype=xp.float64)

        diff = v[rows] - v[cols]

        if diff.ndim == 1:
            weighted_diff = weights * diff
        else:
            weighted_diff = weights[:, None] * diff

        xp.add.at(out, rows,  weighted_diff)
        xp.add.at(out, cols, -weighted_diff)

        return scale * out

    return grad_mv

def cvx_solver_maxcut(n, rows, cols):
    """
    MaxCut SDP using edge formulation.

    Parameters
    ----------
    rows, cols : edge arrays from GSet parser
    n : number of nodes
    """

    # CVXPY variable
    X = cvxpy.Variable((n, n), symmetric=True)

    # Constraints
    constraints = [
        X >> 0,
        cvxpy.diag(X) == 1
    ]

    # ---- Edge-based objective ----
    # Tr(LX) = sum_{(i,j)} (X_ii + X_jj - 2 X_ij)
    edge_term = cvxpy.sum(X[rows, rows] +
                       X[cols, cols] -
                       2 * X[rows, cols])
    objective = cvxpy.Minimize(-0.25 * edge_term)

    prob = cvxpy.Problem(objective, constraints)
    prob.solve(verbose=True)

    print("Optimal value:", prob.value)

    return prob.value, X.value


# def cvx_solver_maxcut(A):
#     # Calculate the coefficients for the objective
#     n = A.shape[0]
#     if isinstance(A, cupy.ndarray):
#         A = A.get()
#     L = np.diag(np.matmul(A, np.ones(n))) - A

#     # Define and solve the MaxCut SDP CVXPY problem.
#     # Create a symmetric matrix variable.
#     X = cp.Variable((n, n), symmetric=True)
#     # The operator >> denotes matrix inequality.
#     constraints = [X >> 0]
#     constraints += [cp.diag(X) == 1]
#     prob = cp.Problem(cp.Minimize(-0.25 * cp.trace(L.T @ X)),
#                       constraints)
#     prob.solve(verbose=True)

#     # Print result.
#     print("The optimal value is", prob.value)
#     print("A solution X is")
#     print(X.value)

#     return prob.value, X.value

def fro_norm_sq_from_eigs(U, alpha, V, beta, xp):
    M = U.T @ V
    trAB = xp.sum((alpha[:, None] * beta[None, :]) * (M**2))

    norm2 = xp.sum(alpha**2) + xp.sum(beta**2) - 2 * trAB
    return xp.maximum(norm2, 0)

def pdhg_solver_maxcut_exact_projection(n, rows, cols, weights, x0, maxit, tau, verbose, freq, save_iterates, use_gpu=False,
                                       snapshot_iters=None, snapshot_dir=None, max_matvecs=None):
    """
    Unified exact PDHG solver for MaxCut SDP supporting both CPU and GPU.
    Set use_gpu=True for GPU arrays (cupy), otherwise uses numpy/scipy.
    """
    xp = cp if use_gpu else np
    # L = xp.diag(xp.matmul(A, xp.ones(n))) - A
    obj = lambda X_lr: obj_maxcut(X_lr, rows, cols, weights, xp=xp) # -0.25 * xp.vdot(L, (X_lr[0]**2) @ X_lr[1])
    grad_mv = make_grad_operator(rows, cols, weights, xp=xp)
    grad = lambda X_lr: grad_mv # -0.25 * L
    Aop = lambda X_lr: (X_lr[0]**2) @ X_lr[1]
    ATop = lambda Y: xp.diag(Y)
    y0 = Aop(x0)
    projH = projection_spectrahedron
    projK = lambda Y: xp.ones_like(y0)
    Lf = 0
    normAop = 1
    sigma = 0.5 * (1 - tau * Lf) / (tau * normAop ** 2)
    info = exact_pdhg_factorized(
        obj=obj,
        grad=grad,
        projH=projH,
        projK=projK,
        Aop=Aop,
        ATop=ATop,
        tau=tau,
        sigma=sigma,
        x0=x0,
        y0=y0,
        maxit=maxit,
        diameter=n,
        freq=freq,
        save_iterates=save_iterates,
        verbose=verbose,
        snapshot_iters=snapshot_iters,
        snapshot_dir=snapshot_dir,
        max_matvecs=max_matvecs
    )
    return info


def pdhg_solver_maxcut_sfw_inexact_projection(n, rows, cols, weights, x0, maxit, tau, rho, verbose, freq, save_iterates,
                                                 use_gpu=False, preprocess_proj_input=True,
                                                 use_accurate_inexact_pdhg=False, snapshot_iters=None,
                                                 snapshot_dir=None, max_matvecs=None):
    """
    Unified inexact (sfw) PDHG solver for MaxCut SDP supporting both CPU and GPU.
    Set use_gpu=True for GPU arrays
    ) (cupy), otherwise uses numpy/scipy.
    """
    xp = cp if use_gpu else np
    # L = xp.diag(xp.matmul(A, xp.ones(n))) - A
    obj = lambda X_lr: obj_maxcut(X_lr, rows, cols, weights, xp=xp) # -0.25 * xp.vdot(L, (X_lr[0]**2) @ X_lr[1])
    grad_mv = make_grad_operator(rows, cols, weights, xp=xp)
    grad = lambda X_lr: grad_mv # -0.25 * L
    gradxT = lambda X_lr: grad_mv
    Aop = lambda X_lr: (X_lr[0]**2) @ X_lr[1]
    ATop = lambda Y: xp.diag(Y)
    ATopyT = lambda Y: xp.diag(Y)
    y0 = Aop(x0)
    if use_accurate_inexact_pdhg:
        projH = accurate_lanczos_projection
    else:
        projH = spectrahedron_fw_projection
    projK = lambda Y: xp.ones_like(y0)
    Lf = 0
    normAop = 1
    sigma = 0.5 * (1 - tau * Lf) / (tau * normAop ** 2)
    def epsilon_func(x, xt, t):
        tol = rho * (1 - tau * Lf - tau * sigma * normAop ** 2) * fro_norm_sq_from_eigs(x[0], x[1], xt[0], xt[1], xp)
        tol = max(tol, 1e-6) # xp.linalg.norm(x - xt) ** 2    
        # print('proj tol:', tol)
        return tol
    proj_maxit = 25
    proj_linesearch = True
    diameter = n
    info = inexact_pdhg_factorized(
        obj=obj,
        grad=grad,
        gradxT=gradxT,
        projH=projH,
        projK=projK,
        Aop=Aop,
        ATop=ATop,
        ATopyT=ATopyT,
        tau=tau,
        sigma=sigma,
        x0=x0,
        y0=y0,
        maxit=maxit,
        epsilon_func=epsilon_func,
        proj_maxit=proj_maxit,
        proj_linesearch=proj_linesearch,
        diameter=diameter,
        freq=freq,
        save_iterates=save_iterates,
        verbose=verbose,
        preprocess_proj_input=preprocess_proj_input,
        use_gpu=use_gpu,
        snapshot_iters=snapshot_iters,
        snapshot_dir=snapshot_dir,
        max_matvecs=max_matvecs
    )
    return info


def pdhg_solver_maxcut_baseline_error_controls_inexact_projection(n, rows, cols, weights, x0, maxit, tau, rho, epsilon, error_tolerance_type,
                                                                  verbose, freq, save_iterates, use_gpu=False,
                                                                  preprocess_proj_input=True, use_accurate_inexact_pdhg=False,
                                                                  snapshot_iters=None, snapshot_dir=None,
                                                                  max_matvecs=None):
    xp = cp if use_gpu else np
    obj = lambda X_lr: obj_maxcut(X_lr, rows, cols, weights, xp=xp) 
    grad_mv = make_grad_operator(rows, cols, weights, xp=xp)
    grad = lambda X_lr: grad_mv 
    gradxT = lambda X_lr: grad_mv
    Aop = lambda X_lr: (X_lr[0]**2) @ X_lr[1]
    ATop = lambda Y: xp.diag(Y)
    ATopyT = lambda Y: xp.diag(Y)
    y0 = Aop(x0)
    if use_accurate_inexact_pdhg:
        projH = accurate_lanczos_projection
    else:
        projH = spectrahedron_fw_projection
    projK = lambda Y: xp.ones_like(y0)
    Lf = 0
    normAop = 1
    sigma = 0.5 * (1 - tau * Lf) / (tau * normAop ** 2)

    if error_tolerance_type == 't_squared_decay':
        def epsilon_func(x, xt, t):
            return max(epsilon / (t + 1) ** 2, 1e-6)
    elif error_tolerance_type == 't_decay':
        def epsilon_func(x, xt, t):
            return max(epsilon / (t + 1), 1e-6)
    elif error_tolerance_type == 'relative_error_control_t_decay':
        def epsilon_func(x, xt, t):
            return max(rho * fro_norm_sq_from_eigs(x[0], x[1], xt[0], xt[1], xp) / (t + 1), 1e-6)
    elif error_tolerance_type == 'relative_error_control_t_squared_decay':
        def epsilon_func(x, xt, t):
            return max(rho * fro_norm_sq_from_eigs(x[0], x[1], xt[0], xt[1], xp) / (t + 1) ** 2, 1e-6)
    elif error_tolerance_type == 'fixed_tol':
        def epsilon_func(x, xt, t):
            return epsilon
    else:
        raise ValueError('Invalid epsilon decay type')

    proj_maxit = 25
    proj_linesearch = True
    diameter = n
    info = inexact_pdhg_factorized(
        obj=obj,
        grad=grad,
        gradxT=gradxT,
        projH=projH,
        projK=projK,
        Aop=Aop,
        ATop=ATop,
        ATopyT=ATopyT,
        tau=tau,
        sigma=sigma,
        x0=x0,
        y0=y0,
        maxit=maxit,
        epsilon_func=epsilon_func,
        proj_maxit=proj_maxit,
        proj_linesearch=proj_linesearch,
        diameter=diameter,
        freq=freq,
        save_iterates=save_iterates,
        verbose=verbose,
        preprocess_proj_input=preprocess_proj_input,
        use_gpu=use_gpu,
        snapshot_iters=snapshot_iters,
        snapshot_dir=snapshot_dir,
        max_matvecs=max_matvecs
    )
    return info


def tune_pdhg(n, rows, cols, weights, x0, maxit, tau_vals, results_path, verbose, freq, save_iterates, use_gpu=False, use_cvx=False, data_path=None):
    # Plotting parameters
    clrs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:cyan', 'tab:olive',
            'tab:pink']
    markers = ['o', '+', '*', 'x', 's', 'D', 'p', 'h', 'v']

    # Calculate the Lipschitz constant and norm of Ad
    Lf = 0
    normAop = 1

    tau_vals = [float(tau) for tau in tau_vals]

    # Check how many tau values provided
    if len(tau_vals) > 9:
        raise ValueError('Number of tau values has to be less than 10')

    epoch_data_dir = os.path.join(results_path, os.path.basename(results_path) + '_epoch_data')
    os.makedirs(epoch_data_dir, exist_ok=True)

    info_list = []
    legend_labels = []

    # Run / load only the tau values explicitly requested by the caller.
    for tau in tau_vals:
        sigma = 0.5 * (1 - tau * Lf) / (tau * normAop ** 2)
        file_prefix = 'exact_pdhg_maxcut_sdp_tau_'
        filename = file_prefix + str(tau) + '_sigma_' + str(sigma) + '.pkl'
        filepath = os.path.join(epoch_data_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, 'rb') as h:
                info = pickle.load(h)
        else:
            if n > 2000:
                info = pdhg_solver_maxcut_sfw_inexact_projection(
                    n, rows, cols, weights, x0, maxit, tau, None, verbose, freq, save_iterates,
                    use_gpu=use_gpu, use_accurate_inexact_pdhg=True, snapshot_dir=None
                )
            else:
                info = pdhg_solver_maxcut_exact_projection(
                    n, rows, cols, weights, x0, maxit, tau, verbose, freq, save_iterates, use_gpu, snapshot_dir=None
                )
            with open(filepath, 'wb') as h:
                pickle.dump(info, h)

        info_list.append(info)
        legend_labels.append('$\\tau$ = ' + str(tau) + ' $\\sigma$ = ' + str(round(sigma, 4)))

    ground_truth_obj = None
    cvx_pickle_path = os.path.join(epoch_data_dir, 'cvx_soln_maxcut_sdp.pkl')
    if use_cvx or os.path.exists(cvx_pickle_path):
        try:
            with open(cvx_pickle_path, 'rb') as h:
                cvx_soln = pickle.load(h)
            ground_truth_obj = float(cvx_soln['obj'])
        except Exception:
            ground_truth_obj = None
    if ground_truth_obj is None:
        graph_name = os.path.splitext(os.path.basename(data_path))[0] if data_path is not None else os.path.basename(results_path)
        graph_name = graph_name.replace('GSet_', '')
        ground_truth_obj = MAXCUT_OPTIMAL_OBJECTIVE_TABLE.get(graph_name)
    if ground_truth_obj is None:
        try:
            ground_truth_obj = float(info_list[0]['obj'][-1])
        except Exception:
            raise FileNotFoundError('No CVX pickle and no benchmark objective found for MaxCut tuning.')

    # Plot the objective residual for the requested tuning runs only
    plt.figure(7)
    for idx, info in enumerate(info_list):
        num_epochs = len(info['obj']) * freq
        epochs = range(1, num_epochs + 1, freq)
        plt.loglog(epochs, abs(info['obj'] - ground_truth_obj) / max(1.0, abs(ground_truth_obj)), linewidth=3,
                marker=markers[idx], color=clrs[idx])
    plt.xlabel('Epochs')
    plt.ylabel('Objective Residual')
    plt.title('Obj. Residual for MaxCut SDP with PDHG')
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, os.path.basename(results_path) + '_plots',
                             'PDHG_tuning_objective_maxcut.pdf'), format='pdf')

    # Plot the feasibility error for the requested tuning runs only
    plt.figure(8)
    for idx, info in enumerate(info_list):
        num_epochs = len(info['obj']) * freq
        epochs = range(1, num_epochs + 1, freq)
        plt.loglog(epochs, info['feas'], linewidth=3, marker=markers[idx], color=clrs[idx])
    plt.xlabel('Epochs')
    plt.ylabel('Feasibility Error')
    plt.title('Feas. Error for MaxCut SDP with PDHG')
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, os.path.basename(results_path) + '_plots',
                             'PDHG_tuning_feasibility_error_maxcut.pdf'), format='pdf')


def tune_rho_inexact_pdhg(n, rows, cols, weights, x0, maxit, tau, rho_vals, results_path, verbose, freq,
                         save_iterates, use_gpu=False, preprocess_proj_input=True,
                         use_accurate_inexact_pdhg=False, use_cvx=False, data_path=None):
    clrs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:cyan', 'tab:olive',
            'tab:pink']
    markers = ['o', '+', '*', 'x', 's', 'D', 'p', 'h', 'v']

    rho_vals = [float(rho) for rho in rho_vals]
    if len(rho_vals) > 9:
        raise ValueError('Number of rho values has to be less than 10')

    epoch_data_dir = os.path.join(results_path, os.path.basename(results_path) + '_epoch_data')
    os.makedirs(epoch_data_dir, exist_ok=True)

    info_list = []
    legend_labels = []

    for rho in rho_vals:
        filename = f'inexact_pdhg_maxcut_sdp_tau_{tau}_rho_{rho}.pkl'
        filepath = os.path.join(epoch_data_dir, filename)

        if os.path.exists(filepath):
            with open(filepath, 'rb') as h:
                info = pickle.load(h)
        else:
            info = pdhg_solver_maxcut_sfw_inexact_projection(
                n=n, rows=rows, cols=cols, weights=weights, x0=x0, maxit=maxit, tau=tau, rho=rho,
                verbose=verbose, freq=freq, save_iterates=save_iterates,
                use_gpu=use_gpu, preprocess_proj_input=preprocess_proj_input,
                use_accurate_inexact_pdhg=use_accurate_inexact_pdhg,
                snapshot_iters=None, snapshot_dir=None
            )
            with open(filepath, 'wb') as h:
                pickle.dump(info, h)

        info_list.append(info)
        legend_labels.append('$\\rho$ = ' + str(rho))

    ground_truth_obj = None
    cvx_pickle_path = os.path.join(epoch_data_dir, 'cvx_soln_maxcut_sdp.pkl')
    if use_cvx or os.path.exists(cvx_pickle_path):
        try:
            with open(cvx_pickle_path, 'rb') as h:
                cvx_soln = pickle.load(h)
            ground_truth_obj = float(cvx_soln['obj'])
        except Exception:
            ground_truth_obj = None
    if ground_truth_obj is None:
        graph_name = os.path.splitext(os.path.basename(data_path))[0] if data_path is not None else os.path.basename(results_path)
        graph_name = graph_name.replace('GSet_', '')
        ground_truth_obj = MAXCUT_OPTIMAL_OBJECTIVE_TABLE.get(graph_name)
    if ground_truth_obj is None:
        try:
            ground_truth_obj = float(info_list[0]['obj'][-1])
        except Exception:
            raise FileNotFoundError('No CVX pickle and no benchmark objective found for rho tuning.')

    plt.figure(9)
    for idx, info in enumerate(info_list):
        num_epochs = len(info['obj']) * freq
        epochs = range(1, num_epochs + 1, freq)
        plt.loglog(epochs,
                   abs(info['obj'] - ground_truth_obj) / max(1.0, abs(ground_truth_obj)),
                   linewidth=3, marker=markers[idx], color=clrs[idx])
    plt.xlabel('Epochs')
    plt.ylabel('Objective Residual')
    plt.title(f'Obj. Residual for MaxCut SDP with fixed $\\tau$ = {tau}')
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, os.path.basename(results_path) + '_plots',
                             f'PDHG_rho_tuning_objective_maxcut_tau_{tau}.pdf'), format='pdf')

    plt.figure(10)
    for idx, info in enumerate(info_list):
        num_epochs = len(info['obj']) * freq
        epochs = range(1, num_epochs + 1, freq)
        plt.loglog(epochs, info['feas'] / np.sqrt(n), linewidth=3, marker=markers[idx], color=clrs[idx])
    plt.xlabel('Epochs')
    plt.ylabel('Feasibility Error')
    plt.title(f'Feas. Error for MaxCut SDP with fixed $\\tau$ = {tau}')
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, os.path.basename(results_path) + '_plots',
                             f'PDHG_rho_tuning_feasibility_error_maxcut_tau_{tau}.pdf'), format='pdf')

    plt.figure(11)
    for idx, info in enumerate(info_list):
        mv, obj, _ = matvec_curve(info)
        plt.loglog(mv,
                abs(obj - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                linewidth=3, marker=markers[idx], color=clrs[idx])
    plt.xlabel('Total Number of Matvecs')
    plt.ylabel('Objective Residual')
    plt.title(f'Obj. Residual vs Total Matvecs for MaxCut SDP with fixed $\\tau$ = {tau}')
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, os.path.basename(results_path) + '_plots',
                             f'PDHG_rho_tuning_objective_maxcut_tau_{tau}_matvecs.pdf'), format='pdf')

    plt.figure(12)
    for idx, info in enumerate(info_list):
        mv, _, feas = matvec_curve(info)
        plt.loglog(mv, feas / np.sqrt(n), linewidth=3, marker=markers[idx], color=clrs[idx])
    plt.xlabel('Total Number of Matvecs')
    plt.ylabel('Feasibility Error')
    plt.title(f'Feas. Error for MaxCut SDP with fixed $\\tau$ = {tau}')
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, os.path.basename(results_path) + '_plots',
                             f'PDHG_rho_tuning_feasibility_error_maxcut_tau_{tau}_matvecs.pdf'), format='pdf')

    plt.figure(13)
    for idx, info in enumerate(info_list):
        num_epochs = len(info['obj']) * freq
        epochs = range(1, num_epochs + 1, freq)
        num_inner = extract_logging_metric(
            info['logging_proj'],
            'num_iters',
            prepend_zero=True
        )
        plt.semilogx(
            epochs,
            num_inner.astype(int),
            linewidth=3,
            marker=markers[idx],
            color=clrs[idx]
        )
    plt.xlabel('Epochs')
    plt.ylabel('Number of SFW Iterations')
    plt.title(f'Number of SFW Iterations for MaxCut SDP with fixed $\\tau$ = {tau}')
    plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, os.path.basename(results_path) + '_plots',
                             f'PDHG_rho_tuning_num_inner_iter_error_maxcut_tau_{tau}_matvecs.pdf'), format='pdf')

    # Print the total number of inner iterations
    for idx, info in enumerate(info_list):
        num_inner = extract_logging_metric(
            info['logging_proj'],
            'num_iters',
            prepend_zero=True
        )
        total_inner = np.sum(num_inner)
        print(f'Total number of inner iterations for rho={rho_vals[idx]}: {total_inner}')

    # Print the total number of matvecs
    for idx, info in enumerate(info_list):
        mv, _, _ = matvec_curve(info)
        total_mv = mv[-1]
        print(f'Total number of matvecs for rho={rho_vals[idx]}: {total_mv}')

    # Print the total number of inner iterations after 2500 iterations
    for idx, info in enumerate(info_list):
        num_inner = extract_logging_metric(
            info['logging_proj'],
            'num_iters',
            prepend_zero=True
        )
        if len(num_inner) > 2500:
            total_inner_2500 = np.sum(num_inner[:2500])
            print(f'Total number of inner iterations after 2500 iterations for rho={rho_vals[idx]}: {total_inner_2500}')
        else:
            print(f'Not enough iterations to report inner iterations after 2500 iterations for rho={rho_vals[idx]}')

    # Print the total number of matvecs after 2500 iterations   
    for idx, info in enumerate(info_list):
        mv, _, _ = matvec_curve(info)
        if len(mv) > 2499:
            total_mv_2500 = mv[2499]
            print(f'Total number of matvecs after 2500 iterations for rho={rho_vals[idx]}: {total_mv_2500}')
        else:
            print(f'Not enough iterations to report matvecs after 2500 iterations for rho={rho_vals[idx]}')
    
    return info_list


def main():

    parser = argparse.ArgumentParser(
        prog='Inexact PDHG for MaxCut',
        description='Solves the MaxCut problem with Inexact PDHG')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the MaxCut data problem file with GSet format.')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Directory path where results should be stored which will be create if needed.')
    parser.add_argument('--run_cvx', action='store_true', default=False,
                        help='Whether or not to run CVX solution.')
    parser.add_argument('--run_exact_pdhg', action='store_true', default=False,
                        help='Whether or not to run the Exact PDHG algorithm.')
    parser.add_argument('--run_sfw_inexact_pdhg', action='store_true', default=False,
                        help='Whether or not to run our Inexact PDHG + SFW algorithm.')
    parser.add_argument('--run_accurate_inexact_pdhg', action='store_true', default=False,
                        help='Whether or not to run the accurate inexact PDHG algorithm.')
    parser.add_argument('--run_baseline_error_control_inexact_pdhg', action='store_true', default=False,
                        help='Whether or not to run Inexact PDHG algorithm with fixed or decaying error tolerance as per fixed rule.')
    parser.add_argument('--run_plotting', action='store_true', default=False,
                        help='Whether or not to plot the results for different types of PDHG.')
    parser.add_argument('--tau', type=float, required=False, default=0.1,
                        help='Hyperparameter tau for PDHG. Sigma will be calculated from this if it is not specified.')
    parser.add_argument('--rho', type=float, required=False, default=0.1,
                        help='Hyperparameter rho for our inexact error scheme (0, 1). ')
    parser.add_argument('--maxit', type=int, required=True,
                        help='Maximum number of iterations for PDHG.')
    parser.add_argument('--max_matvecs', type=int, required=False, default=None,
                        help='Optional hard cap on total projection matvecs before stopping PDHG.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Whether or not to print metrics for PDHG at iteration level.')
    parser.add_argument('--error_tolerance', type=float, required=False, default=1e-3,
                        help='Error tolerance for inexact PDHG with fixed error tolerance.' +
                             'If --error_tolerance_type is set then this will decay at specified rate.')
    parser.add_argument('--error_tolerance_type', nargs='+', type=str, required=False,
                        choices=['t_squared_decay', 't_decay', 'relative_error_control_t_decay',
                                 'relative_error_control_t_squared_decay', 'fixed_tol'], default='fixed',
                        help='Error tolerance type for inexact PHDG.')
    parser.add_argument('--tune_pdhg', action='store_true', default=False,
                        help='Whether or not to tune the Exact PDHG algorithm.')
    parser.add_argument('--tau_vals_tuning', nargs='+', type=float, required=False,
                        help='The tau values to tune with (up to 9). The value of sigma is set automatically.')
    parser.add_argument('--tune_rho_inexact_pdhg', action='store_true', default=False,
                        help='Whether or not to tune rho for the inexact PDHG algorithm at a fixed tau.')
    parser.add_argument('--rho_vals_tuning', nargs='+', type=float, required=False,
                        help='The rho values to tune for the inexact PDHG algorithm at a fixed tau (up to 9).')
    parser.add_argument('--save_iterates', action='store_true', default=False,
                        help='Whether or not to save iterates.')
    parser.add_argument('--freq', type=int, required=False, default=1,
                        help='How often to save the per-iteration data of PDHG.')
    parser.add_argument('--use_gpu', action='store_true', default=False,
                        help='Whether or not to use nonsparse GPU implementation for exact PDHG and sparse GPU implementationfor inexact PDHG  (--sparse is then ignored)')
    parser.add_argument('--use_cvx', action='store_true', default=False,
                        help='Whether or not to use CVX for solving MaxCut.')    
    parser.add_argument('--preprocess_proj_input', action='store_true', default=False,
                        help='Whether or not to shift and scale before FW-based projection.')
    parser.add_argument('--snapshot_iters', nargs='+', type=int, required=False, default=None,
                        help='Iteration indices to save projection snapshots for. Defaults to a log-spaced set when snapshot_dir is provided.')
    parser.add_argument('--snapshot_dir', type=str, required=False, default=None,
                        help='Directory to save PR-computation snapshots. Defaults to results/{graph_name}/snapshots/{method_name}/.')
    args = parser.parse_args()

    # Universal plotting parameters
    plt.rcParams.update({'font.size': 12})
    # Plotting parameters
    clrs = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:cyan', 'tab:olive',
            'tab:pink']
    markers = ['o', '+', '*', 'x', 's', 'D', 'p', 'h', 'v']

    # Create results directory and subdirectories if needed
    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(
        os.path.join(args.results_path, os.path.basename(args.results_path) + '_plots'),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data'),
        exist_ok=True,
    )

    # Read in the GSet file
    n, rows, cols, weights = parse_gset(args.data_path)
    graph_name = os.path.splitext(os.path.basename(args.data_path))[0]

    # Call the CVX solver for MaxCut
    if args.run_cvx:
        cvx_opt_val, sol = cvx_solver_maxcut(n=n, rows=rows, cols=cols)
        cvx_soln = {'obj': cvx_opt_val, 'sol': sol}
        with open(os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                               'cvx_soln_maxcut_sdp.pkl'), 'wb') as h:
            pickle.dump(cvx_soln, h)

    # Generate a starting point
    rank_init = 1
    r0 = min(rank_init, n)
    xp = cp if args.use_gpu else np
    Z = xp.random.randn(n, r0)
    Q, _ = xp.linalg.qr(Z)
    U0 = Q
    alpha0 = (xp.ones(r0) / r0) * n
    x0 = (U0, alpha0)
    weights = xp.asarray(weights)

    # Call exact PDHG
    if args.run_exact_pdhg:
        exact_snapshot_dir = args.snapshot_dir if args.snapshot_dir is not None else default_snapshot_dir(
            args.results_path, args.data_path, 'exact')
        os.makedirs(exact_snapshot_dir, exist_ok=True)
        info_exact = pdhg_solver_maxcut_exact_projection(
            n=n, rows=rows, cols=cols, weights=weights, x0=x0, maxit=args.maxit, tau=args.tau,
            verbose=args.verbose, freq=args.freq, save_iterates=args.save_iterates,
            use_gpu=args.use_gpu, snapshot_iters=args.snapshot_iters, snapshot_dir=exact_snapshot_dir,
            max_matvecs=args.max_matvecs)
        info_exact['time'] = np.vstack(([1e-12], info_exact['time']))
        with open(os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                               'exact_pdhg_maxcut_sdp.pkl'), 'wb') as g:
            pickle.dump(info_exact, g)

    # Call inexact PDHG with our error tolerance scheme
    if args.run_sfw_inexact_pdhg:
        method_name = 'inexact-sfw'
        sfw_snapshot_dir = args.snapshot_dir if args.snapshot_dir is not None else default_snapshot_dir(
            args.results_path, args.data_path, method_name)
        os.makedirs(sfw_snapshot_dir, exist_ok=True)
        info_sfw_inexact = pdhg_solver_maxcut_sfw_inexact_projection(
            n=n, rows=rows, cols=cols, weights=weights, x0=x0, maxit=args.maxit, tau=args.tau, rho=args.rho,
            verbose=args.verbose, freq=args.freq, save_iterates=args.save_iterates,
            use_gpu=args.use_gpu, preprocess_proj_input=args.preprocess_proj_input,
            use_accurate_inexact_pdhg=False,
            snapshot_iters=args.snapshot_iters, snapshot_dir=sfw_snapshot_dir,
            max_matvecs=args.max_matvecs)
        info_sfw_inexact['time'] = np.vstack(([1e-12], info_sfw_inexact['time']))
        with open(os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                               'sfw_inexact_pdhg_maxcut_sdp.pkl'), 'wb') as f:
            pickle.dump(info_sfw_inexact, f)

    if args.run_accurate_inexact_pdhg:
        method_name = 'inexact-accurate'
        accurate_snapshot_dir = args.snapshot_dir if args.snapshot_dir is not None else default_snapshot_dir(
            args.results_path, args.data_path, method_name)
        os.makedirs(accurate_snapshot_dir, exist_ok=True)
        info_accurate_inexact = pdhg_solver_maxcut_sfw_inexact_projection(
            n=n, rows=rows, cols=cols, weights=weights, x0=x0, maxit=args.maxit, tau=args.tau, rho=args.rho,
            verbose=args.verbose, freq=args.freq, save_iterates=args.save_iterates,
            use_gpu=args.use_gpu, preprocess_proj_input=args.preprocess_proj_input,
            use_accurate_inexact_pdhg=True,
            snapshot_iters=args.snapshot_iters, snapshot_dir=accurate_snapshot_dir,
            max_matvecs=args.max_matvecs)
        info_accurate_inexact['time'] = np.vstack(([1e-12], info_accurate_inexact['time']))
        with open(os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                               'accurate_inexact_pdhg_maxcut_sdp.pkl'), 'wb') as f:
            pickle.dump(info_accurate_inexact, f)

    # Call inexact PDHG with a baseline error control scheme
    if args.run_baseline_error_control_inexact_pdhg:
        for error_tolerance_type in args.error_tolerance_type:
            method_name = 'inexact-accurate' if args.run_accurate_inexact_pdhg else 'inexact-sfw'
            baseline_snapshot_dir = args.snapshot_dir if args.snapshot_dir is not None else default_snapshot_dir(
                args.results_path, args.data_path, method_name)
            os.makedirs(baseline_snapshot_dir, exist_ok=True)
            info_baseline_inexact = pdhg_solver_maxcut_baseline_error_controls_inexact_projection(
                n=n, rows=rows, cols=cols, weights=weights, x0=x0, maxit=args.maxit, tau=args.tau, epsilon=args.error_tolerance,
                error_tolerance_type=error_tolerance_type, verbose=args.verbose, freq=args.freq, rho=args.rho,
                save_iterates=args.save_iterates, use_gpu=args.use_gpu, preprocess_proj_input=args.preprocess_proj_input,
                use_accurate_inexact_pdhg=args.run_accurate_inexact_pdhg,
                snapshot_iters=args.snapshot_iters, snapshot_dir=baseline_snapshot_dir,
                max_matvecs=args.max_matvecs)
            info_baseline_inexact['time'] = np.vstack(([1e-12], info_baseline_inexact['time']))
            with open(os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                                error_tolerance_type + '_baseline_inexact_pdhg_maxcut_sdp.pkl'), 'wb') as k:
                pickle.dump(info_baseline_inexact, k)

    # Run the tuning of PDHG if desired
    if args.tune_pdhg:
        tune_pdhg(n=n, rows=rows, cols=cols, weights=weights, x0=x0, maxit=args.maxit, tau_vals=args.tau_vals_tuning, results_path=args.results_path,
                  verbose=args.verbose, freq=args.freq, save_iterates=args.save_iterates,
                  use_gpu=args.use_gpu, use_cvx=args.use_cvx, data_path=args.data_path)

    if args.tune_rho_inexact_pdhg:
        if args.rho_vals_tuning is None:
            raise ValueError('Please provide --rho_vals_tuning when using --tune_rho_inexact_pdhg.')
        tune_rho_inexact_pdhg(
            n=n, rows=rows, cols=cols, weights=weights, x0=x0, maxit=args.maxit, tau=args.tau,
            rho_vals=args.rho_vals_tuning, results_path=args.results_path, verbose=args.verbose,
            freq=args.freq, save_iterates=args.save_iterates, use_gpu=args.use_gpu,
            preprocess_proj_input=args.preprocess_proj_input,
            use_accurate_inexact_pdhg=args.run_accurate_inexact_pdhg,
            use_cvx=args.use_cvx, data_path=args.data_path
        )

    # Plot the objective residual and feasibility error for MaxCut for exact and inexact PDHG
    if args.run_plotting:
        ground_truth_obj = None
        cvx_pickle_path = os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                                      'cvx_soln_maxcut_sdp.pkl')
        if args.use_cvx or os.path.exists(cvx_pickle_path):
            try:
                with open(cvx_pickle_path, 'rb') as h:
                    cvx_soln = pickle.load(h)
                ground_truth_obj = float(cvx_soln['obj'])
            except Exception:
                ground_truth_obj = None
        if ground_truth_obj is None:
            graph_name = os.path.splitext(os.path.basename(args.data_path))[0]
            ground_truth_obj = MAXCUT_OPTIMAL_OBJECTIVE_TABLE.get(graph_name)
        if ground_truth_obj is None:
            try:
                with open(os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                                       'exact_pdhg_maxcut_sdp.pkl'), 'rb') as g:
                    info_exact = pickle.load(g)
                ground_truth_obj = float(info_exact['obj'][-1])
            except Exception:
                raise FileNotFoundError('No CVX pickle and no table entry found for ' + args.data_path)
        exact_pickle_path = os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                                         'exact_pdhg_maxcut_sdp.pkl')
        if os.path.exists(exact_pickle_path):
            with open(exact_pickle_path, 'rb') as g:
                info_exact = pickle.load(g)
        else:
            info_exact = None

        sfw_pickle_path = os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                                           'sfw_inexact_pdhg_maxcut_sdp.pkl')
        if os.path.exists(sfw_pickle_path):
            with open(sfw_pickle_path, 'rb') as f:
                info_sfw_inexact = pickle.load(f)
        else:
            info_sfw_inexact = None

        accurate_pickle_path = os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data',
                                           'accurate_inexact_pdhg_maxcut_sdp.pkl')
        if os.path.exists(accurate_pickle_path):
            with open(accurate_pickle_path, 'rb') as f:
                info_accurate_inexact = pickle.load(f)
        else:
            info_accurate_inexact = None

        num_epochs_exact = len(info_exact['obj']) * args.freq if info_exact is not None else None
        num_epochs_sfw_inexact = len(info_sfw_inexact['obj']) * args.freq if info_sfw_inexact is not None else None
        num_epochs_accurate_inexact = len(info_accurate_inexact['obj']) * args.freq if info_accurate_inexact is not None else None

        epochs_exact = range(1, num_epochs_exact + 1, args.freq) if info_exact is not None else None
        epochs_sfw_inexact = range(1, num_epochs_sfw_inexact + 1, args.freq) if info_sfw_inexact is not None else None
        epochs_accurate_inexact = range(1, num_epochs_accurate_inexact + 1, args.freq) if info_accurate_inexact is not None else None

        # Define mapping from internal identifiers to human-readable labels for the baselines
        # np.set_printoptions(precision=2, formatter={'float_kind': '{:e}'.format})
        label_mapping = {
            't_squared_decay': '1/T² Decay',
            't_decay': '1/T Decay',
            'relative_error_control_t_decay': 'Relative Error Control (1/T Decay)',
            'relative_error_control_t_squared_decay': 'Relative Error Control (1/T² Decay)',
            'fixed_tol': f"Fixed Tolerance {args.error_tolerance:.2e}"
        }

        # Construct directory path
        epoch_data_dir = os.path.join(args.results_path, os.path.basename(args.results_path) + '_epoch_data')

        # Pattern to match relevant pickle files
        pattern = os.path.join(epoch_data_dir, '*_baseline_inexact_pdhg_maxcut_sdp.pkl')

        # Storage lists
        baseline_info_dicts = []
        baseline_labels = []

        # If there are other baseline error control approaches saved,
        # extract the data for each one for plotting
        for file_path in glob.glob(pattern):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                baseline_info_dicts.append(data)

            # Extract the * part (method name)
            filename = os.path.basename(file_path)
            key = filename.split('_baseline_inexact_pdhg_maxcut_sdp.pkl')[0]

            # Store human-readable label
            label = label_mapping.get(key, key)  # Default to key if not found
            baseline_labels.append(label)

        plot_legend = []
        if info_exact is not None:
            plot_legend.append('Exact PDHG')
        if info_sfw_inexact is not None:
            plot_legend.append('Inexact PDHG - Warm Start + SFW (Ours)')
        if info_accurate_inexact is not None:
            plot_legend.append('Accurate Inexact PDHG')
        plot_legend += baseline_labels

        # Plot the objective residual vs epochs
        plt.figure(1)
        if info_exact is not None:
            plt.loglog(epochs_exact,
                    abs(info_exact['obj'] - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[0], color=clrs[0])
        if info_sfw_inexact is not None:
            plt.loglog(epochs_sfw_inexact,
                    abs(info_sfw_inexact['obj'] - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[1], color=clrs[1])
        if info_accurate_inexact is not None:
            plt.loglog(epochs_accurate_inexact,
                    abs(info_accurate_inexact['obj'] - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[2], color=clrs[2])
        for i, info_dict in enumerate(baseline_info_dicts):
            num_epochs = len(info_dict['obj']) * args.freq
            epochs = range(1, num_epochs + 1, args.freq)
            plt.loglog(epochs,
                    abs(info_dict['obj'] - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[3 + i], color=clrs[3 + i])
        if plot_legend:
            plt.xlabel('Epochs')
            plt.ylabel('Objective Residual')
            plt.title('Objective Residual for MaxCut SDP')
            plt.legend(plot_legend)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_path, os.path.basename(args.results_path) + '_plots',
                                     'objective_maxcut_sdp_epochs.pdf'), format='pdf')

        # Plot the feasibility error vs epochs
        plt.figure(2)
        if info_exact is not None:
            plt.loglog(epochs_exact, info_exact['feas'] / np.sqrt(n),
                       linewidth=3, marker=markers[0], color=clrs[0])
        if info_sfw_inexact is not None:
            plt.loglog(epochs_sfw_inexact, info_sfw_inexact['feas'] / np.sqrt(n),
                       linewidth=3, marker=markers[1], color=clrs[1])
        if info_accurate_inexact is not None:
            plt.loglog(epochs_accurate_inexact, info_accurate_inexact['feas'] / np.sqrt(n),
                       linewidth=3, marker=markers[2], color=clrs[2])
        for i, info_dict in enumerate(baseline_info_dicts):
            num_epochs = len(info_dict['obj']) * args.freq
            epochs = range(1, num_epochs + 1, args.freq)
            plt.loglog(epochs, info_dict['feas'] / np.sqrt(n),
                       linewidth=3, marker=markers[3 + i], color=clrs[3 + i])
        if plot_legend:
            plt.xlabel('Epochs')
            plt.ylabel('Feasibility Error')
            plt.title('Feasibility Error for MaxCut SDP')
            plt.legend(plot_legend)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_path, os.path.basename(args.results_path) + '_plots',
                                     'feasibility_error_maxcut_sdp_epochs.pdf'), format='pdf')


        # Plot the objective residual vs total matvec count
        plt.figure(14)
        if info_exact is not None:
            exact_mv, exact_obj, _ = matvec_curve(info_exact)
            plt.loglog(exact_mv,
                    abs(exact_obj - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[0], color=clrs[0])
        if info_sfw_inexact is not None:
            sfw_mv, sfw_obj, _ = matvec_curve(info_sfw_inexact)
            plt.loglog(sfw_mv,
                    abs(sfw_obj - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[1], color=clrs[1])
        if info_accurate_inexact is not None:
            accurate_mv, accurate_obj, _ = matvec_curve(info_accurate_inexact)
            plt.loglog(accurate_mv,
                    abs(accurate_obj - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[2], color=clrs[2])
        for i, info_dict in enumerate(baseline_info_dicts):
            mv, obj_vals, _ = matvec_curve(info_dict)
            plt.loglog(mv,
                    abs(obj_vals - ground_truth_obj) / max(1, abs(ground_truth_obj)),
                    linewidth=3, marker=markers[3 + i], color=clrs[3 + i])
        if plot_legend:
            plt.xlabel('Total Number of Matvecs')
            plt.ylabel('Objective Residual')
            plt.title('Objective Residual vs Total Matvecs for MaxCut SDP')
            plt.legend(plot_legend)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_path, os.path.basename(args.results_path) + '_plots',
                                     'objective_residual_maxcut_sdp_matvecs.pdf'), format='pdf')

        # Plot the feasibility error vs total matvec count
        plt.figure(15)
        if info_exact is not None:
            exact_mv, _, exact_feas = matvec_curve(info_exact)
            plt.loglog(exact_mv, exact_feas / np.sqrt(n), linewidth=3, marker=markers[0], color=clrs[0])
        if info_sfw_inexact is not None:
            sfw_mv, _, sfw_feas = matvec_curve(info_sfw_inexact)
            plt.loglog(sfw_mv, sfw_feas / np.sqrt(n), linewidth=3, marker=markers[1], color=clrs[1])
        if info_accurate_inexact is not None:
            accurate_mv, _, accurate_feas = matvec_curve(info_accurate_inexact)
            plt.loglog(accurate_mv, accurate_feas / np.sqrt(n), linewidth=3, marker=markers[2], color=clrs[2])
        for i, info_dict in enumerate(baseline_info_dicts):
            mv, _, feas_vals = matvec_curve(info_dict)
            plt.loglog(mv, feas_vals / np.sqrt(n), linewidth=3, marker=markers[3 + i], color=clrs[3 + i])
        if plot_legend:
            plt.xlabel('Total Number of Matvecs')
            plt.ylabel('Feasibility Error')
            plt.title('Feasibility Error vs Total Matvecs for MaxCut SDP')
            plt.legend(plot_legend)
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_path, os.path.basename(args.results_path) + '_plots',
                                     'feasibility_error_maxcut_sdp_matvecs.pdf'), format='pdf')

        # Plot the average iterate-to-iterate distance
        if args.save_iterates and info_exact is not None:
            plt.figure(5)
            if args.use_sparse_arrays:
                norm_diffs = [scipy.sparse.linalg.norm(x - y) for (x, y) in
                              zip(info_sfw_inexact['x'], info_exact['x'])]
            else:
                norm_diffs = [np.linalg.norm(x - y) for (x, y) in zip(info_sfw_inexact['x'], info_exact['x'])]
            avg_norm_diffs = np.cumsum(norm_diffs) / np.arange(1, len(norm_diffs) + 1)
            num_iterates = (len(norm_diffs) - 1) * args.freq
            iterates = range(1, num_iterates + 2, args.freq)
            plt.loglog(iterates, avg_norm_diffs, 'b')
            plt.xlabel('Epochs')
            plt.ylabel(r'$\frac{1}{t+1}\sum_{\tau=0}^t\|x_t - x_t^\prime\|^2$')
            plt.title('Mean Norm Difference Between Exact and Inexact Iterates')
            plt.savefig(os.path.join(args.results_path, os.path.basename(args.results_path) + '_plots',
                                     'norm_difference_iterates_maxcut_sdp.pdf'), format='pdf')

        if info_sfw_inexact is not None:
            # ============================================================
            # Number of SFW Iterations
            # ============================================================

            plt.figure(6)

            sfw_num_inner = extract_logging_metric(
                info_sfw_inexact['logging_proj'],
                'num_iters',
                prepend_zero=True
            )

            plt.semilogx(
                epochs_sfw_inexact,
                sfw_num_inner.astype(int),
                linewidth=3,
                marker=markers[1],
                color=clrs[1]
            )

            for i, info_dict in enumerate(baseline_info_dicts):

                num_epochs = len(info_dict['obj']) * args.freq
                epochs = range(1, num_epochs + 1, args.freq)

                num_inner = extract_logging_metric(
                    info_dict['logging_proj'],
                    'num_inner_iters',
                    prepend_zero=True
                )

                plt.semilogx(
                    epochs,
                    num_inner.astype(int),
                    linewidth=3,
                    marker=markers[2 + i],
                    color=clrs[2 + i]
                )

            plt.xlabel('Epochs')
            plt.ylabel('Number of SFW Iterations')
            plt.title('Number of SFW Iterations for PDHG')
            plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            plt.savefig(
                os.path.join(
                    args.results_path,
                    os.path.basename(args.results_path) + '_plots',
                    'num_inner_iter_maxcut_sdp_epochs.pdf'
                ),
                format='pdf'
            )


            # ============================================================
            # Rank of primal iterates
            # ============================================================

            plt.figure(7)

            rank_x = np.concatenate(([0], info_sfw_inexact['rank_x'].flatten()))

            plt.semilogx(
                epochs_sfw_inexact,
                rank_x.astype(int),
                linewidth=3,
                marker=markers[1],
                color=clrs[1]
            )

            for i, info_dict in enumerate(baseline_info_dicts):

                num_epochs = len(info_dict['obj']) * args.freq
                epochs = range(1, num_epochs + 1, args.freq)

                rank_x = np.concatenate(([0], info_dict['rank_x'].flatten()))

                plt.semilogx(
                    epochs,
                    rank_x.astype(int),
                    linewidth=3,
                    marker=markers[2 + i],
                    color=clrs[2 + i]
                )

            plt.xlabel('Epochs')
            plt.ylabel('Rank of Primal Iterate')
            plt.title('Rank of Primal Iterates for PDHG')
            plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            plt.savefig(
                os.path.join(
                    args.results_path,
                    os.path.basename(args.results_path) + '_plots',
                    'rank_x_maxcut_sdp_epochs.pdf'
                ),
                format='pdf'
            )


            # ============================================================
            # Rank of Warm Start
            # ============================================================

            plt.figure(8)

            rank_ws = extract_logging_metric(
                info_sfw_inexact['logging_proj'],
                'rank_ws',
                prepend_zero=True
            )

            plt.semilogx(
                epochs_sfw_inexact,
                rank_ws.astype(int),
                linewidth=3,
                marker=markers[1],
                color=clrs[1]
            )

            for i, info_dict in enumerate(baseline_info_dicts):

                num_epochs = len(info_dict['obj']) * args.freq
                epochs = range(1, num_epochs + 1, args.freq)

                rank_ws = extract_logging_metric(
                    info_dict['logging_proj'],
                    'rank_ws',
                    prepend_zero=True
                )

                plt.semilogx(
                    epochs,
                    rank_ws.astype(int),
                    linewidth=3,
                    marker=markers[2 + i],
                    color=clrs[2 + i]
                )

            plt.xlabel('Epochs')
            plt.ylabel('Rank of Warm Start')
            plt.title('Rank of Warm Start for SFW Projection')

            plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            plt.savefig(
                os.path.join(
                    args.results_path,
                    os.path.basename(args.results_path) + '_plots',
                    'rank_ws_maxcut_sdp_epochs.pdf'
                ),
                format='pdf'
            )


            # ============================================================
            # Mean eigenvalue residual
            # ============================================================

            # plt.figure(9)

            # mean_residual = extract_logging_metric(
            #     info_sfw_inexact['logging_proj'],
            #     'eig_residual_mean',
            #     prepend_zero=True
            # )

            # plt.semilogx(
            #     epochs_sfw_inexact,
            #     mean_residual,
            #     linewidth=3,
            #     marker=markers[1],
            #     color=clrs[1]
            # )

            # for i, info_dict in enumerate(baseline_info_dicts):

            #     num_epochs = len(info_dict['obj']) * args.freq
            #     epochs = range(1, num_epochs + 1, args.freq)

            #     mean_residual = extract_logging_metric(
            #         info_dict['logging_proj'],
            #         'eig_residual_mean',
            #         prepend_zero=True
            #     )

            #     plt.semilogx(
            #         epochs,
            #         mean_residual,
            #         linewidth=3,
            #         marker=markers[2 + i],
            #         color=clrs[2 + i]
            #     )

            # plt.xlabel('Epochs')
            # plt.ylabel('Mean Eigenvalue Residual')

            # plt.title('Mean Eigenvalue Residual for Warm Start')

            # plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            # plt.savefig(
            #     os.path.join(
            #         args.results_path,
            #         os.path.basename(args.results_path) + '_plots',
            #         'mean_eig_residual_maxcut_sdp_epochs.pdf'
            #     ),
            #     format='pdf'
            # )


            # ============================================================
            # Max eigenvalue residual
            # ============================================================

            # plt.figure(10)

            # max_residual = extract_logging_metric(
            #     info_sfw_inexact['logging_proj'],
            #     'eig_residual_max',
            #     prepend_zero=True
            # )

            # plt.semilogx(
            #     epochs_sfw_inexact,
            #     max_residual,
            #     linewidth=3,
            #     marker=markers[1],
            #     color=clrs[1]
            # )

            # for i, info_dict in enumerate(baseline_info_dicts):

            #     num_epochs = len(info_dict['obj']) * args.freq
            #     epochs = range(1, num_epochs + 1, args.freq)

            #     max_residual = extract_logging_metric(
            #         info_dict['logging_proj'],
            #         'eig_residual_max',
            #         prepend_zero=True
            #     )

            #     plt.semilogx(
            #         epochs,
            #         max_residual,
            #         linewidth=3,
            #         marker=markers[2 + i],
            #         color=clrs[2 + i]
            #     )

            # plt.xlabel('Epochs')
            # plt.ylabel('Max Eigenvalue Residual')

            # plt.title('Max Eigenvalue Residual for Warm Start')

            # plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            # plt.savefig(
            #     os.path.join(
            #         args.results_path,
            #         os.path.basename(args.results_path) + '_plots',
            #         'max_eig_residual_maxcut_sdp_epochs.pdf'
            #     ),
            #     format='pdf'
            # )


            # ============================================================
            # Theta
            # ============================================================

            plt.figure(11)

            theta = extract_logging_metric(
                info_sfw_inexact['logging_proj'],
                'theta',
                prepend_zero=True
            )

            plt.semilogx(
                epochs_sfw_inexact,
                theta,
                linewidth=3,
                marker=markers[1],
                color=clrs[1]
            )

            for i, info_dict in enumerate(baseline_info_dicts):

                num_epochs = len(info_dict['obj']) * args.freq
                epochs = range(1, num_epochs + 1, args.freq)

                theta = extract_logging_metric(
                    info_dict['logging_proj'],
                    'theta',
                    prepend_zero=True
                )

                plt.semilogx(
                    epochs,
                    theta,
                    linewidth=3,
                    marker=markers[2 + i],
                    color=clrs[2 + i]
                )

            plt.xlabel('Epochs')
            plt.ylabel('Threshold Theta')

            plt.title('Threshold for Simplex Projection for Warm Start')

            plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            plt.savefig(
                os.path.join(
                    args.results_path,
                    os.path.basename(args.results_path) + '_plots',
                    'theta_maxcut_sdp_epochs.pdf'
                ),
                format='pdf'
            )

            # ============================================================
            # Error Tolerance for Projection Subproblem
            # ============================================================

            plt.figure(12)

            err_tol = extract_logging_metric(
                info_sfw_inexact['logging_proj'],
                'last_eps',
                prepend_zero=True
            )

            plt.loglog(
                epochs_sfw_inexact,
                err_tol,
                linewidth=3,
                marker=markers[1],
                color=clrs[1]
            )

            for i, info_dict in enumerate(baseline_info_dicts):

                num_epochs = len(info_dict['obj']) * args.freq
                epochs = range(1, num_epochs + 1, args.freq)

                err_tol = extract_logging_metric(
                    info_dict['logging_proj'],
                    'last_eps',
                    prepend_zero=True
                )

                plt.loglog(
                    epochs,
                    theta,
                    linewidth=3,
                    marker=markers[2 + i],
                    color=clrs[2 + i]
                )

            plt.xlabel('Epochs')
            plt.ylabel('Error Tolerance')

            plt.title('Error Tolerance for Projection Subproblem')

            plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            plt.savefig(
                os.path.join(
                    args.results_path,
                    os.path.basename(args.results_path) + '_plots',
                    'err_tol_maxcut_sdp_epochs.pdf'
                ),
                format='pdf'
            )

            # ============================================================
            # Last LMO Error
            # ============================================================

            plt.figure(13)

            last_LMO_error = extract_logging_metric(
                info_sfw_inexact['logging_proj'],
                'last_LMO_error',
                prepend_zero=True
            )

            plt.loglog(
                epochs_sfw_inexact,
                last_LMO_error,
                linewidth=3,
                marker=markers[1],
                color=clrs[1]
            )

            for i, info_dict in enumerate(baseline_info_dicts):

                num_epochs = len(info_dict['obj']) * args.freq
                epochs = range(1, num_epochs + 1, args.freq)

                last_LMO_error = extract_logging_metric(
                    info_dict['logging_proj'],
                    'last_LMO_error',
                    prepend_zero=True
                )

                plt.loglog(
                    epochs,
                    last_LMO_error,
                    linewidth=3,
                    marker=markers[2 + i],
                    color=clrs[2 + i]
                )

            plt.xlabel('Epochs')
            plt.ylabel('Last LMO Error')

            plt.title('Last LMO Error for Projection Subproblem')

            plt.legend(['Inexact PDHG - Warm Start + SFW (Ours)'] + baseline_labels)

            plt.savefig(
                os.path.join(
                    args.results_path,
                    os.path.basename(args.results_path) + '_plots',
                    'last_lmo_err_maxcut_sdp_epochs.pdf'
                ),
                format='pdf'
            )

def matvec_curve(info_dict):
    if info_dict is None:
        return None, None, None
    matvecs = np.asarray(info_dict['matvec_count']).ravel().astype(float)
    matvecs = np.where(np.isfinite(matvecs), matvecs, 0.0)
    total_matvecs = np.cumsum(matvecs)
    obj_vals = np.asarray(info_dict['obj']).ravel().astype(float)
    feas_vals = np.asarray(info_dict['feas']).ravel().astype(float)
    if obj_vals.size > total_matvecs.size:
        obj_vals = obj_vals[1:1 + total_matvecs.size]
    if feas_vals.size > total_matvecs.size:
        feas_vals = feas_vals[1:1 + total_matvecs.size]
    return total_matvecs, obj_vals, feas_vals

def extract_logging_metric(logging_proj, key, prepend_zero=False):
    vals = np.array([cp.asnumpy(d[key]) if isinstance(d[key], cp.ndarray) else d[key] for d in logging_proj])

    if prepend_zero:
        if vals.ndim == 1:
            vals = np.concatenate(([0], vals))
        else:
            vals = np.vstack(([0], vals))

    return vals


if __name__ == "__main__":
    main()
