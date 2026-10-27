# InexactPDHG-SDP

Research code for primal--dual hybrid gradient (PDHG) methods for convex
composite optimization, with an emphasis on trace-bounded semidefinite
programs (SDPs). The included MaxCut experiment uses the GSet benchmark
graphs and compares exact PDHG with inexact projection methods.

> **Status:** Research code. Commands and results depend on the
> installed CPU/GPU stack and the benchmark files available locally. See
> the code and paper for the precise algorithmic assumptions and
> experiment settings.

## Contents

-   `algorithms/operator_splitting/` --- exact and inexact PDHG/PDCS
    implementations.
-   `algorithms/projections/` --- PSD and spectrahedron projection
    routines, including Lanczos-based and GPU variants.
-   `algorithms/frank_wolfe/` --- low-rank Frank--Wolfe projection
    routines.
-   `datautils/parsers/gset_parser.py` --- parser for GSet graph
    instances.
-   `scripts/maxcut/run_inexact_pdhg_maxcut.py` --- MaxCut experiment
    entry point.
-   `scripts/download_benchmarks.sh` --- downloads the GSet instances
    G38, G55, G60 and G70.
-   `pyproject.toml` --- package metadata, dependencies, and CLI entry
    point.

## Installation

Python 3.11 or newer is recommended (the project metadata requires
Python `>=3.11`).

``` bash
git clone https://github.com/KarthikPrakhya/InexactPDHG-SDP.git
cd InexactPDHG-SDP

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The project imports CuPy in the MaxCut runner. For GPU experiments,
install the optional GPU dependencies appropriate to your CUDA
environment:

``` bash
python3 -m pip install -e '.[gpu]'
```

The GPU extra currently specifies `cupy-cuda13x` and `pylibraft-cu13`;
make sure your system/cluster supports the corresponding CUDA stack. If
you only need CPU execution, note that the current runner still imports
CuPy at module import time, so CuPy must be importable unless the code
is adjusted to make it optional.

Check the available command-line options with:

``` bash
run_inexact_pdhg_maxcut --help
```

If the console command is not available after installation, run the
script from the repository root:

``` bash
python3 scripts/maxcut/run_inexact_pdhg_maxcut.py --help
```

## Problem formulation

The general convex-composite saddle-point problem is

$$
\min_{x\in\mathcal{X}}\max_{y\in\mathcal{Y}}
\langle Ax,y\rangle + f(x)+g(x)-h^*(y),
$$

where $\mathcal{X}$ and $\mathcal{Y}$ are Hilbert spaces, $A:\mathcal{X}\to\mathcal{Y}$ is a bounded linear operator, $A^*$ is its adjoint, $f$ is convex and differentiable with an $L$-Lipschitz gradient, and $g,h$ are proper, lower-semicontinuous convex functions.

A common PDHG update is

$$
x_{t+1}\approx\mathrm{prox}_{\tau g}
\bigl(x_t-\tau\nabla f(x_t)-\tau A^*y_t\bigr),
$$

$$
y_{t+1}=\mathrm{prox}_{\sigma h^*}
\bigl(y_t+\sigma A(2x_{t+1}-x_t)\bigr).
$$

The inexact projection routines solve the relevant proximal subproblem
approximately. The MaxCut implementation uses a factorized
representation and a problem-specific operator setup; consult
`scripts/maxcut/run_inexact_pdhg_maxcut.py` and the algorithm modules
for implementation details. In the MaxCut runner, $\sigma$ is computed
from $\tau$ as

$$
\sigma=\frac{1}{2\tau}
$$

because that implementation sets $L_f=0$ and $\|A\|=1$. This is an
implementation-specific setting, not a universal step-size rule.

## Download benchmark data

From the repository root, run:

``` bash
bash scripts/download_benchmarks.sh
```

This downloads `G38`, `G55`, `G60` and `G70` into `data/GSet/`. The
runner expects a GSet-format graph file at the path supplied to
`--data_path`.

You can also download the listed instances manually:

``` bash
mkdir -p data/GSet
curl -fL https://web.stanford.edu/~yyye/yyye/Gset/G38 -o data/GSet/G38
curl -fL https://web.stanford.edu/~yyye/yyye/Gset/G55 -o data/GSet/G55
curl -fL https://web.stanford.edu/~yyye/yyye/Gset/G60 -o data/GSet/G60
curl -fL https://web.stanford.edu/~yyye/yyye/Gset/G70 -o data/GSet/G70
```

## Running MaxCut experiments

All commands below are run from the repository root. They create a
results directory per instance so experiment outputs remain separate.

### 1. Tune the PDHG step size

For example, tune $\tau$ on G38:

``` bash
run_inexact_pdhg_maxcut \
  --data_path data/GSet/G38 \
  --results_path results/G38 \
  --tune_pdhg \
  --tau_vals_tuning 1.0 0.0001 0.001 0.01 0.1 10.0 \
  --maxit 2500 \
  --freq 1 \
  --use_gpu
```

The runner calculates $\sigma$ from each candidate $\tau$. It supports
up to nine candidate $\tau$ values per tuning run. Remove `--use_gpu` to
request the CPU path (subject to the CuPy import note above).

### 2. Tune $\rho$ for inexact PDHG

After selecting $\tau$, tune $\rho$ on the same instance:

``` bash
run_inexact_pdhg_maxcut \
  --data_path data/GSet/G38 \
  --results_path results/G38 \
  --tau 1.0 \
  --tune_rho_inexact_pdhg \
  --rho_vals_tuning 0.0001 0.001 0.01 0.1 1.0 10.0 \
  --maxit 2500 \
  --freq 1 \
  --use_gpu \
  --preprocess_proj_input
```

The runner supports up to nine candidate $\rho$ values per tuning run.

### 3. Run exact PDHG

``` bash
run_inexact_pdhg_maxcut \
  --data_path data/GSet/G38 \
  --results_path results/G38 \
  --run_exact_pdhg \
  --tau 1.0 \
  --maxit 20000 \
  --freq 1 \
  --use_gpu
```

This was not used in the paper experiments.

### 4. Run the inexact methods

The inexact Frank--Wolfe projection method is selected with
`--run_sfw_inexact_pdhg`; the more accurate Lanczos-based variant is
selected with `--run_accurate_inexact_pdhg`. For example:

``` bash
run_inexact_pdhg_maxcut \
  --data_path data/GSet/G38 \
  --results_path results/G38 \
  --run_sfw_inexact_pdhg \
  --tau 1.0 \
  --rho 0.01 \
  --maxit 20000 \
  --freq 1 \
  --use_gpu \
  --preprocess_proj_input \
  --run_plotting
```

To run the accurate inexact variant instead, replace
`--run_sfw_inexact_pdhg` with `--run_accurate_inexact_pdhg`.

### 5. Compare against baseline error controls

``` bash
run_inexact_pdhg_maxcut \
  --data_path data/GSet/G38 \
  --results_path results/G38 \
  --run_baseline_error_control_inexact_pdhg \
  --tau 1.0 \
  --error_tolerance 0.001 \
  --error_tolerance_type fixed_tol \
  --maxit 20000 \
  --freq 1 \
  --use_gpu \
  --preprocess_proj_input
```

This was not used as part of the paper experiments.

Supported `--error_tolerance_type` values are:

-   `fixed_tol`: constant tolerance $\epsilon_t=\epsilon_0$.
-   `t_decay`: $\epsilon_t=\epsilon_0/(t+1)$.
-   `t_squared_decay`: $\epsilon_t=\epsilon_0/(t+1)^2$.
-   `relative_error_control_t_decay`: tolerance proportional to the
    squared iterate difference, decaying as $1/(t+1)$.
-   `relative_error_control_t_squared_decay`: tolerance proportional to
    the squared iterate difference, decaying as $1/(t+1)^2$.

The baseline scale $\epsilon_0$ is set by `--error_tolerance`. In the
current implementation, the decaying and relative rules apply a lower
tolerance floor of `1e-6`; `fixed_tol` returns the supplied value

### Useful command-line options

| Option | Meaning |
| --- | --- |
| `--data_path` | Path to a GSet graph instance; required. |
| `--results_path` | Directory for result files; required. |
| `--maxit` | Maximum number of iterations; required. |
| `--tau` | PDHG primal step size; default `0.1`. |
| `--rho` | Inexact-method parameter; default `0.1`. |
| `--freq` | Logging/saving frequency; default `1`. |
| `--use_gpu` | Enable GPU implementations. |
| `--preprocess_proj_input` | Shift/scale the input before Frank–Wolfe-based projection. |
| `--save_iterates` | Save iterates. |
| `--max_matvecs` | Optional cap on projection matrix-vector products. |
| `--snapshot_iters`, `--snapshot_dir` | Save projection snapshots at selected iterations. |
| `--run_plotting` | Generate plots from the run. |

The runner also supports `--run_cvx` and `--use_cvx` for CVXPY-based solving, subject to the installed CVXPY solver setup.

## Outputs

The runner writes experiment data beneath the directory supplied to
`--results_path`. Tuning routines create an epoch-data subdirectory
named `<results-directory-basename>_epoch_data`. Plotting and snapshot
outputs depend on the selected flags. Keep each experiment's
`--results_path` distinct to avoid mixing cached tuning results and
outputs.

## Reproducing paper experiments

The historical VS Code launch configurations used for the paper are not
included in this repository archive. Their current equivalents can be
run using the commands above, with the graph path, iteration count,
$\tau$, $\rho$, and flags taken from the relevant experiment
configuration. The exact command-line interface is defined by the parser
in `scripts/maxcut/run_inexact_pdhg_maxcut.py`; check `--help` before
reusing older launch settings.

## Citation

If you use this code in academic work, please cite the associated paper:

``` bibtex
@misc{prakhya2026inexact,
  author        = {Prakhya, Karthik and Wang, Zimeng},
  title         = {Inexact PDHG for Large-Scale Semidefinite Programming},
  year          = {2026},
  eprint        = {submit/8291925},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  note          = {Submitted October 27, 2026}
}
```

## License

The package metadata declares the MIT License. Ensure the repository
includes the corresponding `LICENSE` file before redistributing the
project.
