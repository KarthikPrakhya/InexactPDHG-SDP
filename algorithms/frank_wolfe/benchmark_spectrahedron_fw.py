import time
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator
from algorithms.frank_wolfe.spectrahedron_frank_wolfe_low_rank import spectrahedron_fw_projection, vanilla_fw_projection
import matplotlib.pyplot as plt

# --------------------------------------------------
# Closed-form PSD projection
# --------------------------------------------------
def closed_form_psd_projection(G):
    """
    Exact projection onto {X >= 0, trace(X)=1}
    """
    w, V = cp.linalg.eigh(G)
    n = w.size

    w_sorted = cp.sort(w)[::-1]
    cssv = cp.cumsum(w_sorted)
    rho = cp.max(
        cp.where(
            w_sorted - (cssv - 1) / (cp.arange(1, n+1)) > 0
        )[0]
    ) + 1

    theta = (cssv[rho-1] - 1) / rho
    w_proj = cp.maximum(w - theta, 0)

    H_star = (V * w_proj) @ V.T

    trace_H2 = cp.sum(w_proj**2)
    trace_HG = cp.sum(H_star * G)
    f_star = 0.5 * float(trace_H2 - 2 * trace_HG)
    print(f"Closed-form projection: f*={f_star:.4f}")

    return H_star, f_star


# --------------------------------------------------
# Objective using low-rank factorization X = U diag(alpha) U^T
# --------------------------------------------------
def f_obj_lowrank(V, G):
    """
    f(X) = 0.5 ||X-G||_F^2
    with X = V V^T
    """
    Gm = V.T @ V
    trace_H2 = cp.sum(Gm**2)

    GV = cp.column_stack([G @ V[:, i] for i in range(V.shape[1])])
    trace_HG = cp.sum(V * GV)

    return 0.5 * float(trace_H2 - 2 * trace_HG)


def f_obj_factorized(U, alpha, G):
    """Objective for X = U diag(alpha) U^T with trace-normalized low-rank factor."""
    if U.shape[1] == 0:
        return 0.5 * float(cp.sum(G * G))

    V = U * cp.sqrt(cp.maximum(alpha, 0.0))[None, :]
    return f_obj_lowrank(V, G)


# --------------------------------------------------
# Dummy epsilon
# --------------------------------------------------
def eps_func(H, start_value, outer_iter):
    return 1e-8


# --------------------------------------------------
# Benchmark
# --------------------------------------------------
def benchmark_spectrahedron_vs_exact(
    sizes = [500, 1000, 2000, 4000, 5000, 7000, 10000],
    rank_init = 1,
    fw_max_iters = 25,
    seed=1
):

    results = []
    plot_data = {}


    for n in sizes:
        cp.random.seed(seed)
        print(f"\n=== Testing n={n} ===")

        # ------------------------------------------
        # Random symmetric G
        # ------------------------------------------
        A = cp.random.randn(n, n)
        G_dense = 0.5 * (A + A.T)

        # LinearOperator required by FW
        def G_mv(x):
            return G_dense @ x

        G_op = LinearOperator((n, n), matvec=G_mv, rmatvec=G_mv)

        # ------------------------------------------
        # Exact projection
        # ------------------------------------------
        t0 = time.time()
        H_star, f_star = closed_form_psd_projection(G_dense)
        exact_time = time.time() - t0
        print(f"Exact eig-projection: {exact_time:.4f}s")

        # ------------------------------------------
        # Low-rank initialization (U0, alpha0)
        # ------------------------------------------
        r0 = min(rank_init, n)
        Z = cp.random.randn(n, r0)
        Q, _ = cp.linalg.qr(Z)
        U0 = Q
        alpha0 = cp.ones(r0) / r0

        # ------------------------------------------
        # Spectrahedron FW
        # ------------------------------------------
        t1 = time.time()
        (U_fw, alpha_fw), info = spectrahedron_fw_projection(
            G=G_op,
            start_value=(U0, alpha0),
            epsilon_func=eps_func,
            outer_iter_num=0,
            maxit=fw_max_iters,
            line_search=False,
            diameter=1.0,
            return_logging=True,
            verbose=True,
            preprocess_input=False,
            generate_spectral_warm_start=False,
        )

        fw_time = time.time() - t1

        f_fw = f_obj_factorized(U_fw, alpha_fw, G_dense)
        err_fw = abs(f_fw - f_star) / abs(f_star) if f_star != 0 else abs(f_fw - f_star)
        final_lmo_fw = float(info['LMO_errors'][-1]) if info.get('LMO_errors') else float('nan')
        sfw_iters = int(info.get('num_iters', len(info.get('iters', []))))
        print(f"FW time: {fw_time:.4f}s (iters={sfw_iters}) | rel error={err_fw:.2e} | final LMO error={final_lmo_fw:.2e}")

        # ------------------------------------------
        # Vanilla FW
        # ------------------------------------------
        t2 = time.time()
        (U_vfw, alpha_vfw), info_vfw = vanilla_fw_projection(
            G=G_op,
            start_value=(U0, alpha0),
            epsilon_func=eps_func,
            outer_iter_num=0,
            maxit=fw_max_iters,
            line_search=False,
            diameter=1.0,
            return_logging=True,
            verbose=True,
            preprocess_input=False,
            generate_spectral_warm_start=False,
        )

        vfw_time = time.time() - t2

        f_vfw = f_obj_factorized(U_vfw, alpha_vfw, G_dense)
        err_vfw = abs(f_vfw - f_star) / abs(f_star) if f_star != 0 else abs(f_vfw - f_star)
        final_lmo_vfw = float(info_vfw['LMO_errors'][-1]) if info_vfw.get('LMO_errors') else float('nan')
        vfw_iters = int(info_vfw.get('num_iters', len(info_vfw.get('iters', []))))
        print(f"Vanilla FW time: {vfw_time:.4f}s (iters={vfw_iters}) | rel error={err_vfw:.2e} | final LMO error={final_lmo_vfw:.2e}")


        # Store logs for plotting if n >= 2000
        if n >= 2000:
            plot_data[n] = {
                'fw': {
                    'times': cp.asnumpy(cp.array(info['times'])),
                    'lmo': cp.asnumpy(cp.array(info['LMO_errors'])),
                },
                'vfw': {
                    'times': cp.asnumpy(cp.array(info_vfw['times'])),
                    'lmo': cp.asnumpy(cp.array(info_vfw['LMO_errors'])),
                }
            }

        results.append({
            "n": n,
            "exact_time": exact_time,
            "sfw_time": fw_time,
            "vfw_time": vfw_time,
            "sfw_speedup": exact_time / fw_time,
            "vfw_speedup": exact_time / vfw_time,
            "sfw_obj_error": err_fw,
            "vfw_obj_error": err_vfw,
            "sfw_iters": sfw_iters,
            "vfw_iters": vfw_iters,
            "sfw_lmo": final_lmo_fw,
            "vfw_lmo": final_lmo_vfw
        })

    return results, plot_data


# --------------------------------------------------
# Runner
# --------------------------------------------------
if __name__ == "__main__":
    print("Benchmark: exact vs Spectrahedron-FW vs Vanilla-FW (max 25 iterations)")
    results, plot_data = benchmark_spectrahedron_vs_exact()
    print("\n=== Summary (max 25 iterations for FW algorithms) ===")
    print("n    | exact_time | sfw_time | vfw_time | sfw_speedup | vfw_speedup | sfw_rel_error | vfw_rel_error | sfw_iters | vfw_iters | sfw_LMO_err | vfw_LMO_err")
    print("-----|------------|---------|----------|------------|-------------|--------------|---------------|----------|-----------|------------|-------------")
    for r in results:
        print(f"{r['n']:4d} | {r['exact_time']:.4f}s | {r['sfw_time']:.4f}s | {r['vfw_time']:.4f}s | "
              f"{r['sfw_speedup']:.2f}x | {r['vfw_speedup']:.2f}x | {r['sfw_obj_error']:.2e} | {r['vfw_obj_error']:.2e} | "
              f"{r['sfw_iters']:2d} | {r['vfw_iters']:2d} | {r['sfw_lmo']:.2e} | {r['vfw_lmo']:.2e}")

    # Plotting for n >= 2000
    for n in sorted(plot_data.keys()):
        fw = plot_data[n]['fw']
        vfw = plot_data[n]['vfw']

        plt.figure(figsize=(7,5))
        plt.loglog(fw['times'], fw['lmo'], label='Spectrahedron FW', marker='o')
        plt.loglog(vfw['times'], vfw['lmo'], label='Vanilla FW', marker='x')
        plt.xlabel('Time (s)')
        plt.ylabel('LMO error')
        plt.title(f'LMO error vs Time (n={n})')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(f'lmo_vs_time_n{n}.png')
        plt.close()

        plt.figure(figsize=(7,5))
        plt.loglog(range(1, len(fw['lmo'])+1), fw['lmo'], label='Spectrahedron FW', marker='o')
        plt.loglog(range(1, len(vfw['lmo'])+1), vfw['lmo'], label='Vanilla FW', marker='x')
        plt.xlabel('Iteration')
        plt.ylabel('LMO error')
        plt.title(f'LMO error vs Iteration (n={n})')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(f'lmo_vs_iter_n{n}.png')
        plt.close()