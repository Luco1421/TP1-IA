# TP1 — Optimization algorithms from scratch

Three optimization algorithms implemented manually in PyTorch (no
`torch.optim`, gradients computed by hand): **Gradient Descent**,
**RMSProp**, and **CMA-ES** (Covariance Matrix Adaptation Evolution
Strategy). Each is run against three 2D benchmark functions — a
McCormick-style function, a two-term trigonometric multimodal function, and
**Himmelblau's function** — from 10 random starting points.

Hyperparameters for each algorithm (learning rate for GD; decay/learning
rate for RMSProp; population size, learning rate, and initial step size for
CMA-ES) are tuned automatically with **Optuna** (50-trial studies per
function/algorithm combination), minimizing the final function value.
Results are tabulated with pandas and visualized as 3D surfaces, contour
plots with optimization trajectories, and convergence curves. Includes
assertion-based unit tests checking convergence behavior and boundary
clamping against the known minima (e.g. Himmelblau's ~[3, 2]).

<img width="740" height="240" alt="image" src="https://github.com/user-attachments/assets/9dd16df7-8d88-4828-9871-88c5a1628e84" />

## Stack

PyTorch (manual gradients), Optuna, pandas, matplotlib.

## Running

Open `TP1.ipynb` in Jupyter — the first cell installs its own dependencies
(`torch`, `optuna`, `matplotlib`, `pandas`, `plotly`, `nbformat`). Runs on
CPU; no GPU required.
