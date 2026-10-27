import os
import scipy.sparse.linalg as splinalg
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cusplinalg
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

import time  
import numpy as np
from collections.abc import Callable

matvec_count = 0

def _is_cupy_array(x):
    """Check if x is a cupy array."""
    return HAS_CUPY and cp is not None and isinstance(x, cp.ndarray)


def _to_numpy(x):
    """Convert cupy array to numpy if needed."""
    if _is_cupy_array(x):
        return cp.asnumpy(x)
    return x

def logarithmic_iters(maxit, count=12):
    return set(
        np.unique(
            np.round(
                np.geomspace(1, maxit, count)
            ).astype(int)
        )
    )

def inexact_pdhg_factorized(obj: Callable[[tuple[np.ndarray, np.ndarray]], float],
         grad: Callable[[tuple[np.ndarray, np.ndarray]], np.ndarray],
         gradxT: Callable[[tuple[np.ndarray, np.ndarray]], np.ndarray],
         projH: Callable[..., np.ndarray],
         projK: Callable[[np.ndarray], np.ndarray],
         Aop: Callable[[np.ndarray], np.ndarray],
         ATop: Callable[[np.ndarray], np.ndarray],
         ATopyT: Callable[[np.ndarray], np.ndarray],
         tau: float, sigma: float,
         x0: tuple[np.ndarray, np.ndarray], y0: np.ndarray,
         maxit: int, epsilon_func: Callable[[np.ndarray, np.ndarray, int], float], proj_maxit: int,
         proj_linesearch: bool, diameter: float,
         verbose: bool, freq: int, save_iterates: bool, warm_start: dict = None, preprocess_proj_input: bool = False, use_gpu : bool = True, 
         snapshot_iters=None, snapshot_dir=None, max_matvecs: int | None = None):
    """
    The inexact_pdhg_factorized function runs the inexact adaptive Primal-Dual Hybrid Gradient (PDHG) with adaptive error tolerances for projection.
    It allows the user to specify the desired operators for this operator splitting scheme. It solves the following optimization problem:
    At present, only projection onto set K is allowed to be inexact while projection onto set H is exact. This function
    allows the factorization b of the primal variable x (ie. x = bb^T) to be stored with x in a tuple (b, x) in the first
    iteration and in the subsequent iteration, it will store the active set of various Frank Wolfe variants.

    min f(x)
    s.t. x in H
         A(x) in K

    where A is an operator, H and K are conic sets.

    @type obj: Callable[[numpy.ndarray], float]
    @param obj: the objective function f(x) to minimize
    @type grad: Callable[[numpy.ndarray], float]
    @param grad: the gradient operator of objective function f
    @type gradT: Callable[[numpy.ndarray], float]
    @param gradxT: the transpose of the gradient operator evaluated at x of objective function f
    @type Lf: float
    @param Lf: estimate of Lipschitz constant for objective function f
    @type projH: Callable[..., np.ndarray]
    @param projH: the inexact projection function over primal variable x onto set H. This can be a function like spectrahedron_fw_projection from algorithms/frank_wolfe/spectrahedron_frank_wolfe_low_rank.py, which expects arguments:
        - G: a LinearOperator or matrix representing the gradient step (see below)
        - start_value: initial value for projection (usually previous iterate)
        - epsilon_func: error tolerance function
        - outer_iter_num: iteration number (or 0)
        - maxit: maximum number of projection iterations
        - line_search: whether to use line search
        - diameter: diameter of the set H
        - return_logging: whether to return logging info
        - verbose: print debug info (if supported)
      When using spectrahedron_fw_projection, the G argument should be constructed as a LinearOperator representing the matvec x[1] - alpha * grad(x) - beta * ATop(y).
    @type projK: Callable[[numpy.ndarray], np.ndarray]
    @param projK: the projection function over dual variable y onto set K.
    @type Aop: Callable[[numpy.ndarray], np.ndarray]
    @param Aop: the A operator that takes the primal variable as input
    @type ATop: Callable[[numpy.ndarray], np.ndarray]
    @param ATop: the transpose of the A operator that takes the dual variable as input
    @type ATopyT: Callable[[numpy.ndarray], np.ndarray]
    @param ATopT: takes y and returns the transpose of the ATop(y) operator 
    @type alpha: float
    @param alpha: the alpha hyperparameter for PDHG
    @type beta: float
    @param beta: the beta hyperparameter for PDHG
    @type x0: (numpy.ndarray, np.ndarray)
    @param x0: the initial value of primal variable x
    @type y0: numpy.ndarray
    @param y0: the initial value of dual variable y, same dimension as A(x)
    @type maxit: int
    @param maxit: the maximum number of iterations to run PDHG
    @type epsilon_func: Callable([numpy.ndarray, numpy.ndarray], float)
    @param epsilon_func: an error tolerance function that takes the current FW iterate and previous PDHG iterate as inputs
    @type proj_maxit: int
    @param proj_maxit: the maximum number of iterations to run algorithm for inexact projection onto set K
    @type proj_linesearch: bool
    @param proj_linesearch: whether or not to do linesearch for step size used for inexact projection onto set K
    @type diameter: float
    @param diameter: the diameter of the projection (diameter of the set H)
    @type: bool
    @param verbose: a boolean flag to indicate whether to print obj value and feas. error every iteration
    @type warm_start: dict
    @param warm_start: a dictionary with same format as the return argument of this function (see below) to warm start
                       the PDHG algorithm.
    @type preprocess_proj_input: bool
    @param preprocess_proj_input: whether or not to shift/scale the projection input
    @type use_gpu: bool
    @param use_gpu: whether or not to use the GPU
    @type snapshot_iters: List
    @param snapshot_iters: which iterations to save as snapshots for projection replay study
    @type snapshot_dir: str
    @param snapshot_dir: where the iteration snapshots are saved
    @rtype dict
    @return: a dictionary containing the following keys:
        x: an numpy.ndarray that holds the solution to the problem
        obj: an numpy.ndarray that holds the objective value for each iteration of PDHG
        feas: an numpy.ndarray that holds the feasibility error for each iteration of PDHG
        time: an numpy.ndarray that holds the time for each iteration of PDHG
        maxit: an int that indicates the maximum number of iterations to run PDHG
        tau: the tau hyperparameter for PDHG
        sigma: the sigma hyperparameter for PDHG.
    """
    xp = cp if (HAS_CUPY and cp is not None and isinstance(x0[0], cp.ndarray)) else np

    global matvec_count
    matvec_count = 0

    if snapshot_iters is None and snapshot_dir is not None:
        snapshot_iters = logarithmic_iters(maxit, count=12)
    elif snapshot_iters is not None:
        snapshot_iters = set(snapshot_iters)

    # Initialization
    if not warm_start:
        x = x0
        Ax = Aop(x0)
        y = y0
        # Create a dictionary to hold performance metrics
        if save_iterates:
            info = {'obj': np.full((int(maxit/freq) + 1, 1), np.nan), 'feas': np.full((int(maxit/freq) + 1, 1), np.nan),
                    'time': np.full((int(maxit/freq), 1), np.nan), 'x': [],
                    'num_inner_iter': np.full((int(maxit/freq), 1), np.nan), 'matvec_count': np.full((int(maxit/freq), 1), np.nan),
                    'rank_x': np.full((int(maxit/freq), 1), np.nan),
                    'y': [], 'z': []}
        else:
            info = {'obj': np.full((int(maxit/freq) + 1, 1), np.nan), 'feas': np.full((int(maxit/freq) + 1, 1), np.nan),
                    'time': np.full((int(maxit/freq), 1), np.nan),
                    'num_inner_iter': np.full((int(maxit/freq), 1), np.nan), 'matvec_count': np.full((int(maxit/freq), 1), np.nan),
                    'rank_x': np.full((int(maxit/freq), 1), np.nan), 'logging_proj': []}
        loop_range = range(maxit)
    else:
        if 'x' in warm_start.keys():
            if isinstance(warm_start['x'], list):
                x = warm_start['x'][-1]
                y = warm_start['y'][-1]
            else:
                x = warm_start['x']
                y = warm_start['y']
            Ax = Aop(x)

            old_maxit = warm_start['maxit']

            # Create a dictionary to hold performance metrics
            if save_iterates:
                info = {'obj': np.full((maxit % freq, 1), np.nan), 'feas': np.full((maxit % freq, 1), np.nan),
                        'time': np.full((maxit % freq, 1), np.nan), 'x': warm_start['x'],
                        'num_inner_iter': np.full((int(maxit / freq), 1), np.nan), 'matvec_count': np.full((int(maxit / freq), 1), np.nan),
                        'y': warm_start['y'], 'z': warm_start['z'], 'rank_x': np.full((int(maxit / freq), 1), np.nan)}
            else:
                info = {'obj': np.full((maxit % freq, 1), np.nan), 'feas': np.full((maxit % freq, 1), np.nan),
                        'time': np.full((maxit % freq, 1), np.nan),
                        'num_inner_iter': np.full((int(maxit / freq), 1), np.nan), 'matvec_count': np.full((int(maxit / freq), 1), np.nan), 'rank_x': np.full((int(maxit / freq), 1), np.nan)}
            info['obj'][0:old_maxit, :] = warm_start['obj']
            info['feas'][0:old_maxit, :] = warm_start['feas']
            info['time'][0:old_maxit, :] = warm_start['time']
            info['num_inner_iter'][0:old_maxit, :] = warm_start['num_inner_iter']
            info['rank_x'][0:old_maxit, :] = warm_start['rank_x']
            info['matvec_count'][0:old_maxit, :] = warm_start['matvec_count']
            if 'logging_proj' in warm_start.keys():
                info['logging_proj'][0:old_maxit, :] = warm_start['logging_proj']
            loop_range = range(old_maxit, maxit)
        else:
            raise ValueError('Cannot warm start from provided data as no data exists for primal and dual iterates.')

    # Main loop for PDHG
    index = 0
    obj_value = obj(x)
    feas_error = xp.linalg.norm(Ax - projK(Ax))
    if _is_cupy_array(obj_value):
        obj_value = _to_numpy(obj_value)
        feas_error = _to_numpy(feas_error)
    info['obj'][0] = obj_value
    info['feas'][0] = feas_error

    backend = "cupy" if HAS_CUPY and xp is cp else "numpy"
    if backend == "cupy":
        cp.cuda.get_current_stream().synchronize()
    t0 = time.perf_counter()

    for k in loop_range:
        if max_matvecs is not None and matvec_count >= max_matvecs:
            break

        # Record the primal and dual iterate before PDHG loop
        x_proj_start = (
            x[0].copy(),
            x[1].copy()
        )
        y_proj_start = y.copy()
        previous_rank = x[0].shape[1]

        Axold = Ax
        def matvec(v):
            if xp is cp and not isinstance(v, cp.ndarray):
                v = cp.asarray(v)

            v = v.reshape(-1, 1)

            term1 = x[0] @ (x[1].reshape(-1,1) * (x[0].T @ v))
            term2 = -tau * grad(x)(v).reshape(-1, 1)
            term3 = -tau * (ATop(y) @ v).reshape(-1, 1)
            out = term1 + term2 + term3
            global matvec_count
            matvec_count += 1
            return out.reshape(-1, 1)
        
        def rmatvec(v):
            return matvec(v)  # For symmetric G, rmatvec is the same as matvec
        
        def matmat(X):
            return xp.column_stack([matvec(X[:, i]) for i in range(X.shape[1])])

        if HAS_CUPY and use_gpu:
            n = x[0].shape[0]
            G = cusplinalg.LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, matmat=matmat, dtype=x[0].dtype)
        else:
            n = x[0].shape[0]
            G = splinalg.LinearOperator((n, n), matvec=matvec, rmatvec=matvec, matmat=matmat, dtype=x[0].dtype)

        # Call projH with G and other arguments
        projection_matvec_start = matvec_count

        xFW, logging_proj = projH(
            G=G,
            start_value=x,
            epsilon_func=epsilon_func,
            outer_iter_num=k,
            maxit=proj_maxit,
            line_search=proj_linesearch,
            diameter=diameter,
            return_logging=True,  # return_logging is True since we want to collect logging information about projection
            verbose=False,  
            preprocess_input=preprocess_proj_input,
            generate_spectral_warm_start=True,  
        )
        projection_matvecs = matvec_count - projection_matvec_start

        x = xFW
        Ax = Aop(x)
        z = y + 2 * Ax - Axold
        y = z - projK(z)

        # Save the snapshot
        rank_change_threshold = max(10, int(0.02 * n))
        current_rank = x[0].shape[1]

        rank_changed = (
            abs(current_rank - previous_rank)
            >= rank_change_threshold
        )

        save_rank_change_snapshot = False
        if rank_changed:
            save_rank_change_snapshot = True

        if snapshot_dir is not None and (k in snapshot_iters or save_rank_change_snapshot):
            U_snap = x_proj_start[0]
            alpha_snap = x_proj_start[1]
            y_snap = y_proj_start

            if xp is cp:
                U_snap = cp.asnumpy(U_snap)
                alpha_snap = cp.asnumpy(alpha_snap)
                y_snap = cp.asnumpy(y_snap)

            snapshot_path = os.path.join(
                snapshot_dir,
                f"snapshot_iter_{k:06d}.npz"
            )

            np.savez_compressed(
                snapshot_path,
                # Input to projection
                iter=k,
                start_U=U_snap,
                start_alpha=alpha_snap,
                y_start=y_snap,

                # Parameters
                tau=tau,
                sigma=sigma,
                diameter=diameter,

                # Information obtained after projection
                previous_rank=previous_rank,
                projected_rank=current_rank,
                rank_changed=rank_changed,
            )

        obj_value = obj(x)
        feas_error = xp.linalg.norm((Ax - projK(Ax)).reshape(-1, 1))
        if _is_cupy_array(obj_value):
            obj_value = _to_numpy(obj_value)
            feas_error = _to_numpy(feas_error)
        if verbose:
            print('[PDHG] Iteration: %i, Obj: %f, Feas: %f' % (k, obj_value, feas_error))
        if k % freq == 0:
            info['obj'][index + 1] = obj_value
            info['feas'][index + 1] = feas_error
            info['matvec_count'][index] = projection_matvecs
            info['rank_x'][index] = x[0].shape[1] 
            info['logging_proj'].append(logging_proj)
            if save_iterates:
                info['x'].append(x)
                info['y'].append(y)
                info['z'].append(z)
            index += 1
    info['maxit'] = maxit 
    info['tau'] = tau
    info['sigma'] = sigma
    info['max_matvecs'] = max_matvecs
    info['matvec_budget_reached'] = bool(max_matvecs is not None and matvec_count >= max_matvecs)
    if not save_iterates:
        info['x'] = x
        info['y'] = y
    return info