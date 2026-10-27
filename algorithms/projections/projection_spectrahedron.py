import numpy as np
import scipy.sparse.linalg as splinalg
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cusplinalg
    HAS_CUPY = True
except ImportError:    
    HAS_CUPY = False


def projection_spectrahedron(G, diameter=1.0):
    """
    Projects onto the (optionally scaled) spectrahedron {H >= 0, trace(H) = diameter}.
    Accepts G as a dense matrix, LinearOperator, or matvec function.
    diameter: float, the trace constraint (default 1.0)
    Returns: (U, alpha) low-rank representation of projected matrix
    """
    xp = cp if HAS_CUPY and isinstance(G, cp.ndarray) or isinstance(G, cusplinalg.LinearOperator) else np
    xsplinalg = cusplinalg if HAS_CUPY and isinstance(G, cusplinalg.LinearOperator) else splinalg
    
    # Convert G to dense if it's a LinearOperator or matvec
    n = None
    if isinstance(G, xp.ndarray):
        G_dense = G
        n = G.shape[0]
    elif isinstance(G, xsplinalg.LinearOperator):
        n = G.shape[0]
        # Apply G to identity to get dense
        G_dense = xp.column_stack([G(xp.eye(n)[:, i]) for i in range(n)])
    else:
        raise ValueError("G must be a dense matrix or a LinearOperator.")

    # Use cupy if available and input is cupy, else numpy
    if hasattr(cp, 'ndarray') and isinstance(G_dense, cp.ndarray):
        xp = cp
    else:
        xp = np
        G_dense = xp.asarray(G_dense)

    w, V = xp.linalg.eigh(G_dense)
    w_sorted = xp.sort(w)[::-1]
    cssv = xp.cumsum(w_sorted)
    rho_idx = int(xp.max(xp.where(w_sorted - (cssv - diameter) / (xp.arange(1, n+1)) > 0)[0]) + 1)
    theta = (cssv[rho_idx - 1] - diameter) / rho_idx
    w_star = xp.maximum(w - theta, 0)
    # Return low-rank representation (eigenvectors, eigenvalues)
    return (V, w_star)
