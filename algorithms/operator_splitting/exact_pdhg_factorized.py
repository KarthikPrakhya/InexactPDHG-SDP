import os
import numpy as np
import scipy.sparse.linalg as splinalg
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cusplinalg
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    
import time  
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

def exact_pdhg_factorized(obj: Callable[[tuple[np.ndarray, np.ndarray]], float],
         grad: Callable[[tuple], np.ndarray],
         projH: Callable[[tuple], tuple],
         projK: Callable[[np.ndarray], np.ndarray],
         Aop: Callable[[tuple], np.ndarray],
         ATop: Callable[[np.ndarray], tuple],
         tau: float, sigma: float,
         x0: tuple, y0: np.ndarray,
         maxit: int, diameter: float,
         verbose: bool, freq: int, save_iterates: bool, warm_start: dict = None, use_gpu : bool = True, snapshot_iters=None, snapshot_dir=None, max_matvecs: int | None = None):
    """
    The pdhg function runs the exact Primal-Dual Hybrid Gradient (PDHG). It allows the user to specify the desired
    operators for this operator splitting scheme. It solves the following optimization problem:

    min f(x)
    s.t. x in H
         A(x) in K

    where A is an operator, H and K are conic sets. Exact PDHG for low-rank (factorized) representation. Uses exact projection for projH.
    Arguments mirror inexact_pdhg_factorized, but projH should be an exact projection function.
    x = (U, alpha) where U: eigenvectors, alpha: eigenvalues.

    @type obj: Callable[[tuple], float]
    @param obj: the objective function f(x) to minimize
    @type grad: Callable[[tuple], np.ndarray]
    @param grad: the gradient of objective function f
    @type projH: Callable[[numpy.ndarray], np.ndarray]
    @param projH: the projection function over primal variable W onto set H
    @type projK: Callable[[np.ndarray], np.ndarray]
    @param projK: the projection function over dual variable y onto set K.
    @type Aop: Callable[[numpy.ndarray], np.ndarray]
    @param Aop: the A operator that takes the primal variable as input
    @type ATop: Callable[[numpy.ndarray], np.ndarray]
    @param ATop: the transpose of the A operator that takes the dual variable as input
    @type tau: float
    @param tau: the tau hyperparameter for PDHG
    @type sigma: float
    @param sigma: the sigma hyperparameter for PDHG
    @type x0: numpy.ndarray
    @param x0: the initial value of primal variable x
    @type y0: numpy.ndarray
    @param y0: the initial value of dual variable y, same dimension as A(x)
    @type maxit: int
    @param maxit: the maximum number of iterations to run PDHG
    @type diameter: float
    @param diameter: the diameter of the primal variable set H, used for adaptive step siz
    @type verbose: bool
    @param verbose: a boolean flag to indicate whether to print obj value and feas. error every iteration
    @type freq: int
    @param freq: frequency of saving PDHG performance metrics
    @type save_iterates: bool
    @param save_iterates: whether or not to do save_iterates
    @type warm_start: dict
    @param warm_start: a dictionary with the same format as the return argument of this function (see below) to warm start
                       the PDHG algorithm.
    @param use_gpu: whether or not to use the GPU
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
                    'y': [], 'z': [], 'matvec_count': np.full((int(maxit/freq), 1), np.nan), 'rank_x': np.full((int(maxit/freq), 1), np.nan)}
        else:
            info = {'obj': np.full((int(maxit/freq) + 1, 1), np.nan), 'feas': np.full((int(maxit/freq) + 1, 1), np.nan),
                    'time': np.full((int(maxit/freq), 1), np.nan), 'matvec_count': np.full((int(maxit/freq), 1), np.nan), 'rank_x': np.full((int(maxit/freq), 1), np.nan)}
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
                info = {'obj': np.full((int(maxit/freq) + 1, 1), np.nan), 'feas': np.full((int(maxit/freq) + 1, 1), np.nan),
                        'time': np.full((int(maxit/freq), 1), np.nan), 'x': warm_start['x'],
                        'y': warm_start['y'], 'z': warm_start['z'], 'matvec_count': np.full((int(maxit/freq), 1), np.nan), 'rank_x': np.full((int(maxit/freq), 1), np.nan)}
            else:
                info = {'obj': np.full((int(maxit/freq) + 1, 1), np.nan), 'feas': np.full((int(maxit/freq) + 1, 1), np.nan),
                        'time': np.full((int(maxit/freq), 1), np.nan), 'matvec_count': np.full((int(maxit/freq), 1), np.nan), 'rank_x': np.full((int(maxit/freq), 1), np.nan)}
            info['obj'][0:old_maxit, :] = warm_start['obj']
            info['feas'][0:old_maxit, :] = warm_start['feas']
            info['time'][0:old_maxit, :] = warm_start['time']
            info['matvec_count'][0:old_maxit, :] = warm_start['matvec_count']
            info['rank_x'][0:old_maxit, :] = warm_start['rank_x']
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
        # Construct matvec for G as in inexact_pdhg_factorized
        def matvec(v):
            if xp is cp and not isinstance(v, cp.ndarray):
                v = cp.asarray(v)
            global matvec_count
            matvec_count += 1
            # v = v.reshape(-1, 1)  # reshape v to match the shape of x[1]
            return x[0] @ (x[1] * (x[0].T @ v)) - tau * grad(x)(v) - tau * ATop(y) @ v  
        def rmatvec(v):
            return matvec(v)  # For symmetric G, rmatvec is the same as matvec
        
        if HAS_CUPY and isinstance(x0[0], cp.ndarray):
            n = x[0].shape[0]
            G = cusplinalg.LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec)
        else:
            n = x[0].shape[0]
            G = splinalg.LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec)
        # Exact projection: projH should accept G and x, but do exact projection
        projection_matvec_start = matvec_count
        xFW = projH(G, diameter=diameter)
        projection_matvecs = matvec_count - projection_matvec_start

        x = xFW
        Ax = Aop(x)
        z = y + sigma*(2 * Ax - Axold)
        y = z - sigma*projK(z/sigma)

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
            print('Iteration: %i, Obj: %f, Feas: %f' % (k, obj_value, feas_error))
        if k % freq == 0:
            info['obj'][index + 1] = obj_value
            info['feas'][index + 1] = feas_error
            info['matvec_count'][index] = projection_matvecs
            info['rank_x'][index] = x[0].shape[1] 
            if save_iterates:
                info['x'].append(x)
                info['y'].append(y)
                info['z'].append(z)
            index += 1
    if backend == "cupy":
        cp.cuda.get_current_stream().synchronize()
    info['total_time'] = time.perf_counter() - t0
    info['maxit'] = maxit
    info['tau'] = tau
    info['sigma'] = sigma
    info['max_matvecs'] = max_matvecs
    info['matvec_budget_reached'] = bool(max_matvecs is not None and matvec_count >= max_matvecs)
    if not save_iterates:
        info['x'] = x
        info['y'] = y
    return info
