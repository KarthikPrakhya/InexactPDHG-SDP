import numpy as np

def parse_gset(filename):
    """
    Parse a GSet graph.

    Returns
    -------
    n : int
        Number of vertices.
    rows : ndarray, shape (m,)
        Zero-based source vertices.
    cols : ndarray, shape (m,)
        Zero-based destination vertices.
    weights : ndarray, shape (m,)
        Edge weights, typically +/-1.
    """

    with open(filename, encoding="utf8") as f:
        n, m = map(int, next(f).split())

        rows = np.empty(m, dtype=np.int32)
        cols = np.empty(m, dtype=np.int32)
        weights = np.empty(m, dtype=np.float64)

        for k, line in enumerate(f):
            i, j, w = map(int, line.split())

            rows[k] = i - 1
            cols[k] = j - 1
            weights[k] = w

    return n, rows, cols, weights