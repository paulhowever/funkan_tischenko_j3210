"""Generate every figure and CSV referenced by report_outputs/example.tex.

The script is idempotent: rerunning it overwrites all assets with the
current state of the code base, so the report is always in sync.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from case_6.data import (
    load_california_dataset,
    load_diabetes_dataset,
    make_sinusoidal_dataset,
    train_test_split,
)
from case_6.experiments import (
    kernel_vs_window_impact,
    lowess_diagnostic_artifacts,
    lowess_outlier_threshold_study,
    run_real_dataset_benchmark,
    synthetic_curve_artifacts,
    variable_vs_fixed_win_map,
)
from case_6.kernels import KERNELS
from case_6.lowess import lowess_fit_predict
from case_6.metrics import mae, r2, rmse
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable
from case_6.selection import (
    loo_score_fixed,
    loo_score_variable,
    select_fixed_window,
    select_variable_window,
)

ROOT = Path(".")
FIG = ROOT / "report_outputs" / "figures"
TAB = ROOT / "report_outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    }
)


def save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Case 4: functional regression with linear functionals
# ---------------------------------------------------------------------------
print("[case 4] generating synthetic functional dataset")
rng = np.random.default_rng(42)
n_samples, n_grid = 280, 320
t = np.linspace(0.0, 1.0, n_grid)
a = rng.normal(0.0, 1.0, size=n_samples)
b = rng.normal(0.0, 1.0, size=n_samples)
c = rng.normal(0.0, 0.8, size=n_samples)
phase = rng.uniform(-0.3, 0.3, size=n_samples)

x_clean = (
    a[:, None] * np.sin(2.0 * np.pi * t[None, :] + phase[:, None])
    + b[:, None] * np.cos(2.0 * np.pi * t[None, :] - 0.5 * phase[:, None])
    + c[:, None] * t[None, :]
)
x_func = x_clean + rng.normal(0.0, 0.06, size=x_clean.shape)
y_true = (
    2.0 * np.trapezoid(x_func * np.sin(2.0 * np.pi * t)[None, :], t, axis=1)
    - np.trapezoid(x_func * t[None, :], t, axis=1)
)
y = y_true + rng.normal(0.0, 0.10, size=n_samples)

idx_perm = rng.permutation(n_samples)
n_test = int(0.25 * n_samples)
idx_test = idx_perm[:n_test]
idx_train = idx_perm[n_test:]
x_train_fn = x_func[idx_train]
x_test_fn = x_func[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

# 4.0 sample functions
plt.figure(figsize=(9, 5))
for i in range(8):
    plt.plot(t, x_func[i], linewidth=1.0, alpha=0.85)
plt.title("Кейс 4: примеры сгенерированных функциональных данных $x_i(t)$")
plt.xlabel("t")
plt.ylabel("$x_i(t)$")
save(FIG / "sample_functions.png")


# basis builders
def indicator_basis(m: int):
    edges = np.linspace(0.0, 1.0, m + 1)
    return [
        (lambda tt, l=edges[k], r=edges[k + 1]: ((tt >= l) & (tt < r)).astype(float))
        for k in range(m)
    ]


def polynomial_basis(m: int):
    return [(lambda tt, p=k: tt**p) for k in range(m)]


def trigonometric_basis(m: int):
    base = [lambda tt: np.ones_like(tt)]
    h = 1
    while len(base) < m:
        base.append(lambda tt, hh=h: np.sin(2 * np.pi * hh * tt))
        if len(base) >= m:
            break
        base.append(lambda tt, hh=h: np.cos(2 * np.pi * hh * tt))
        h += 1
    return base[:m]


def feature_matrix(x_fn: np.ndarray, basis) -> np.ndarray:
    Z = np.zeros((x_fn.shape[0], len(basis)))
    for k, phi in enumerate(basis):
        v = phi(t)
        Z[:, k] = np.trapezoid(x_fn * v[None, :], t, axis=1)
    return Z


def fit_ridge(Z: np.ndarray, y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    Z_aug = np.column_stack([np.ones(Z.shape[0]), Z])
    A = Z_aug.T @ Z_aug
    if lam > 0:
        I = np.eye(A.shape[0])
        I[0, 0] = 0
        A = A + lam * I
    return np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), Z_aug.T @ y)


def predict(Z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(Z.shape[0]), Z]) @ beta


basis_makers = {
    "Индикаторный": indicator_basis,
    "Полиномиальный": polynomial_basis,
    "Тригонометрический": trigonometric_basis,
}
m_grid = [3, 5, 7, 9, 11, 13, 15]
print("[case 4] sweeping basis x m")
basis_table_rows = []
rmse_curves: dict[str, list[float]] = {n: [] for n in basis_makers}
train_curves: dict[str, list[float]] = {n: [] for n in basis_makers}
best_models: list[dict] = []
for name, maker in basis_makers.items():
    for m in m_grid:
        basis = maker(m)
        Ztr = feature_matrix(x_train_fn, basis)
        Zte = feature_matrix(x_test_fn, basis)
        beta = fit_ridge(Ztr, y_train)
        ptr, pte = predict(Ztr, beta), predict(Zte, beta)
        tr_r, te_r = rmse(y_train, ptr), rmse(y_test, pte)
        train_curves[name].append(tr_r)
        rmse_curves[name].append(te_r)
        basis_table_rows.append([name, m, round(tr_r, 5), round(te_r, 5),
                                 round(r2(y_train, ptr), 5), round(r2(y_test, pte), 5)])
        best_models.append(
            {"basis": name, "m": m, "te_rmse": te_r, "Ztr": Ztr, "Zte": Zte, "beta": beta, "maker": maker}
        )
write_csv(TAB / "linear_basis_comparison.csv",
          ["basis", "m", "train_rmse", "test_rmse", "train_r2", "test_r2"], basis_table_rows)

best_linear = min(best_models, key=lambda d: d["te_rmse"])

plt.figure(figsize=(6.4, 4.2))
xs = np.arange(len(basis_makers))
vals = [min(rmse_curves[n]) for n in basis_makers]
bars = plt.bar(xs, vals, color=["#4C78A8", "#F58518", "#54A24B"], width=0.55)
plt.xticks(xs, list(basis_makers.keys()), rotation=10)
plt.ylabel("min test RMSE (log scale)")
plt.yscale("log")
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width() / 2, v * 1.07, f"{v:.3f}",
             ha="center", va="bottom", fontsize=10)
plt.title("Кейс 4: лучший test RMSE по базисам (log-шкала)")
plt.grid(axis="y", which="both", linestyle=":", alpha=0.5)
plt.tight_layout()
save(FIG / "basis_comparison_rmse.png")

plt.figure()
for name in basis_makers:
    plt.plot(m_grid, rmse_curves[name], marker="o", label=name)
plt.xlabel("m"); plt.ylabel("test RMSE")
plt.title("Кейс 4: test RMSE от числа функционалов m")
plt.legend()
save(FIG / "rmse_vs_m.png")

plt.figure()
plt.plot(m_grid, train_curves[best_linear["basis"]], marker="o", label="train RMSE")
plt.plot(m_grid, rmse_curves[best_linear["basis"]], marker="s", label="test RMSE")
plt.xlabel("m"); plt.ylabel("RMSE")
plt.title(f"Кейс 4: train/test RMSE от m ({best_linear['basis']})")
plt.legend()
save(FIG / "train_test_rmse_vs_m.png")

# ridge sweep on best basis
ridge_lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ridge_rows = []
best_ridge_pred = None
best_ridge_te = float("inf")
best_ridge_lam = 0.0
for lam in ridge_lambdas:
    beta_l = fit_ridge(best_linear["Ztr"], y_train, lam)
    ptr = predict(best_linear["Ztr"], beta_l)
    pte = predict(best_linear["Zte"], beta_l)
    tr_r, te_r = rmse(y_train, ptr), rmse(y_test, pte)
    ridge_rows.append([best_linear["basis"], best_linear["m"], lam,
                       round(tr_r, 5), round(te_r, 5),
                       round(r2(y_test, pte), 5), round(mae(y_test, pte), 5)])
    if te_r < best_ridge_te:
        best_ridge_te, best_ridge_pred, best_ridge_lam = te_r, pte, lam
write_csv(TAB / "best_linear_models.csv",
          ["basis", "m", "lambda", "train_rmse", "test_rmse", "test_r2", "test_mae"], ridge_rows)

plt.figure()
plt.semilogx(ridge_lambdas, [r[4] for r in ridge_rows], marker="o")
plt.xlabel(r"$\lambda$"); plt.ylabel("test RMSE")
plt.title(r"Кейс 4: зависимость test RMSE от $\lambda$ (ridge)")
save(FIG / "ridge_lambda_curve.png")

# linear true vs pred + residuals
order = np.argsort(y_test)
plt.figure()
plt.plot(np.arange(y_test.size), y_test[order], label="истинные y", linewidth=2)
plt.plot(np.arange(y_test.size), best_ridge_pred[order], label="предсказанные y", linewidth=2)
plt.xlabel("индекс (сорт. по y)"); plt.ylabel("y"); plt.legend()
plt.title("Кейс 4: истинные и предсказанные значения")
save(FIG / "linear_true_vs_pred.png")

plt.figure()
plt.hist(y_test - best_ridge_pred, bins=20, edgecolor="black")
plt.xlabel(r"$y - \hat y$"); plt.ylabel("частота")
plt.title("Кейс 4: распределение остатков линейной модели")
save(FIG / "linear_residuals_distribution.png")

# weight functions per basis
plt.figure(figsize=(9, 5))
for name, maker in basis_makers.items():
    basis = maker(best_linear["m"])
    Ztr = feature_matrix(x_train_fn, basis)
    beta = fit_ridge(Ztr, y_train)
    w_t = sum(c * phi(t) for c, phi in zip(beta[1:], basis))
    plt.plot(t, w_t, label=name, linewidth=2)
plt.title("Кейс 4: восстановленные весовые функции $w(t)$")
plt.xlabel("t"); plt.ylabel("w(t)"); plt.legend()
save(FIG / "weight_functions_comparison.png")

# linearity check (LHS = LHS, manual numbers; use real integrals)
phi_sin = np.sin(2 * np.pi * t)
phi_t = t
lhs = float(np.trapezoid((phi_sin + phi_t) * x_func[0], t))
rhs = float(np.trapezoid(phi_sin * x_func[0], t)) + float(np.trapezoid(phi_t * x_func[0], t))
write_csv(TAB / "linearity_check.csv",
          ["check", "lhs", "rhs", "abs_diff"],
          [["L(phi1+phi2) vs L(phi1)+L(phi2)", round(lhs, 6), round(rhs, 6), round(abs(lhs - rhs), 9)]])

# ---------------------------------------------------------------------------
# Case 6: metric methods
# ---------------------------------------------------------------------------
print("[case 6] generating metric-method figures")
KERNEL_NAMES = ["gaussian", "epanechnikov", "triangular", "quartic"]

# 1D synthetic for plots
ds_1d = make_sinusoidal_dataset(n_samples=200, noise_std=0.13, seed=123)
tr_1d, te_1d = train_test_split(ds_1d, test_size=0.25, seed=123)
x_grid = np.linspace(-3.0, 3.0, 400)[:, None]

plt.figure(figsize=(9, 5))
plt.scatter(tr_1d.x[:, 0], tr_1d.y, s=10, alpha=0.4, label="train")
plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), label="sin(x)", linewidth=2)
for h in [0.1, 0.25, 0.5]:
    pred = nw_predict_fixed(tr_1d.x, tr_1d.y, x_grid, h=h, kernel_name="gaussian")
    plt.plot(x_grid[:, 0], pred, linewidth=1.5, label=f"NW, h={h}")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.title("Одномерная синтетика: NW при разных h")
save(FIG / "onedim_nw_different_h.png")

h_eval = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8]
rmse_h_1d = [
    rmse(te_1d.y, nw_predict_fixed(tr_1d.x, tr_1d.y, te_1d.x, h=h, kernel_name="gaussian"))
    for h in h_eval
]
plt.figure()
plt.plot(h_eval, rmse_h_1d, marker="o")
plt.xlabel("h"); plt.ylabel("test RMSE")
plt.title("Одномерная синтетика: test RMSE от h")
save(FIG / "onedim_rmse_vs_h.png")

# kernel comparison sweep on a slightly contaminated synthetic set
ds_main = make_sinusoidal_dataset(n_samples=240, noise_std=0.12, outlier_fraction=0.08, seed=42)
tr, te = train_test_split(ds_main, test_size=0.25, seed=42)
hs = [0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8]
ks = [3, 5, 8, 10, 12, 15, 20]
fixed_rows = []
variable_rows = []
loo_h: dict[str, list[float]] = {k: [] for k in KERNEL_NAMES}
loo_k: dict[str, list[float]] = {k: [] for k in KERNEL_NAMES}
for kern in KERNEL_NAMES:
    for h in hs:
        loo = loo_score_fixed(tr.x, tr.y, h=h, kernel_name=kern)
        pred = nw_predict_fixed(tr.x, tr.y, te.x, h=h, kernel_name=kern)
        fixed_rows.append([kern, h, round(loo, 5), round(rmse(te.y, pred), 5),
                           round(mae(te.y, pred), 5), round(r2(te.y, pred), 5)])
        loo_h[kern].append(loo)
    for k in ks:
        loo = loo_score_variable(tr.x, tr.y, k=k, kernel_name=kern)
        pred = nw_predict_variable(tr.x, tr.y, te.x, k=k, kernel_name=kern)
        variable_rows.append([kern, k, round(loo, 5), round(rmse(te.y, pred), 5),
                              round(mae(te.y, pred), 5), round(r2(te.y, pred), 5)])
        loo_k[kern].append(loo)

write_csv(TAB / "kernel_fixed_window_results.csv",
          ["kernel", "h", "loo_rmse", "test_rmse", "test_mae", "test_r2"], fixed_rows)
write_csv(TAB / "kernel_variable_window_results.csv",
          ["kernel", "k", "loo_rmse", "test_rmse", "test_mae", "test_r2"], variable_rows)

plt.figure()
for kern in KERNEL_NAMES:
    plt.plot(hs, loo_h[kern], marker="o", label=kern)
plt.xlabel("h"); plt.ylabel("LOO RMSE")
plt.title("Кейс 6: LOO RMSE от ширины окна $h$")
plt.legend()
save(FIG / "kernel_rmse_vs_h.png")

plt.figure()
for kern in KERNEL_NAMES:
    plt.plot(ks, loo_k[kern], marker="o", label=kern)
plt.xlabel("k"); plt.ylabel("LOO RMSE")
plt.title("Кейс 6: LOO RMSE от числа соседей $k$")
plt.legend()
save(FIG / "kernel_rmse_vs_k.png")

# best per-kernel models (by LOO)
best_per_kernel = []
for kern in KERNEL_NAMES:
    cands = [r for r in fixed_rows if r[0] == kern] + [r for r in variable_rows if r[0] == kern]
    best_per_kernel.append(min(cands, key=lambda r: r[2]))
write_csv(TAB / "best_kernel_models.csv",
          ["kernel", "param", "loo_rmse", "test_rmse", "test_mae", "test_r2"], best_per_kernel)
plt.figure()
plt.bar(np.arange(len(KERNEL_NAMES)), [r[3] for r in best_per_kernel],
        color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
plt.xticks(np.arange(len(KERNEL_NAMES)), KERNEL_NAMES)
plt.ylabel("test RMSE")
plt.title("Кейс 6: лучшая модель по каждому ядру")
save(FIG / "kernel_comparison_best_rmse.png")

# fixed vs variable
best_fixed = min(fixed_rows, key=lambda r: r[2])
best_variable = min(variable_rows, key=lambda r: r[2])
g_fixed = nw_predict_fixed(tr.x, tr.y, x_grid, h=float(best_fixed[1]), kernel_name=str(best_fixed[0]))
g_var = nw_predict_variable(tr.x, tr.y, x_grid, k=int(best_variable[1]), kernel_name=str(best_variable[0]))
plt.figure(figsize=(9, 5))
plt.scatter(tr.x[:, 0], tr.y, s=10, alpha=0.35, label="train")
plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), label="sin(x)", linewidth=2)
plt.plot(x_grid[:, 0], g_fixed, label=f"fixed ({best_fixed[0]}, h={best_fixed[1]})", linewidth=1.6)
plt.plot(x_grid[:, 0], g_var, label=f"variable ({best_variable[0]}, k={int(best_variable[1])})", linewidth=1.6)
plt.title("Кейс 6: фиксированное vs переменное окно")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
save(FIG / "fixed_vs_variable_window.png")

# best overall model — true vs pred + residuals
best_overall = best_fixed if best_fixed[2] <= best_variable[2] else best_variable
if best_overall is best_fixed:
    pred_best = nw_predict_fixed(tr.x, tr.y, te.x, h=float(best_overall[1]), kernel_name=str(best_overall[0]))
else:
    pred_best = nw_predict_variable(tr.x, tr.y, te.x, k=int(best_overall[1]), kernel_name=str(best_overall[0]))

order_te = np.argsort(te.x[:, 0])
plt.figure()
plt.scatter(te.x[order_te, 0], te.y[order_te], s=12, alpha=0.5, label="истинные y")
plt.plot(te.x[order_te, 0], pred_best[order_te], color="C1", linewidth=2, label="предсказание")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
plt.title("Кейс 6: истинные и предсказанные значения (лучшая модель)")
save(FIG / "kernel_true_vs_pred.png")

plt.figure()
plt.hist(te.y - pred_best, bins=20, edgecolor="black")
plt.xlabel(r"$y - \hat y$"); plt.ylabel("частота")
plt.title("Кейс 6: распределение остатков ядерной модели")
save(FIG / "kernel_residuals_distribution.png")

# outliers effect
print("[case 6] outlier sweep")
threshold = lowess_outlier_threshold_study(seed=42)
levels = [r[0] for r in threshold]
rmse_nw_l = [r[1] for r in threshold]
rmse_lo_l = [r[2] for r in threshold]
plt.figure()
plt.plot(levels, rmse_nw_l, marker="o", label="NW")
plt.plot(levels, rmse_lo_l, marker="s", label="LOWESS")
plt.xlabel("доля выбросов"); plt.ylabel("test RMSE")
plt.title("Кейс 6: влияние выбросов на качество регрессии")
plt.legend()
save(FIG / "outliers_effect_rmse.png")
write_csv(TAB / "lowess_outlier_comparison.csv",
          ["outlier_fraction", "nw_rmse", "lowess_rmse"],
          [[round(r[0], 4), round(r[1], 5), round(r[2], 5)] for r in threshold])

# lowess gammas + 1D illustration
print("[case 6] lowess diagnostics")
diag = lowess_diagnostic_artifacts(seed=42)
order_lo = np.argsort(diag["x_train"])
plt.figure(figsize=(9, 5))
plt.plot(diag["x_train"][order_lo], diag["gamma"][order_lo], linewidth=1.4)
plt.scatter(diag["x_train"][order_lo], diag["gamma"][order_lo], s=10, alpha=0.7)
plt.xlabel("x"); plt.ylabel(r"$\gamma_i$")
plt.title("Кейс 6: финальные веса $\\gamma_i$ (LOWESS)")
save(FIG / "lowess_final_weights.png")
write_csv(TAB / "lowess_final_weights.csv", ["x", "gamma"],
          [[float(diag["x_train"][i]), float(diag["gamma"][i])] for i in order_lo])

# 1D LOWESS vs NW under outliers
ds_out = make_sinusoidal_dataset(n_samples=220, noise_std=0.12, outlier_fraction=0.14, seed=456)
tr_o, _ = train_test_split(ds_out, test_size=0.25, seed=456)
nw_grid = nw_predict_fixed(tr_o.x, tr_o.y, x_grid,
                            h=float(best_fixed[1]), kernel_name=str(best_fixed[0]))
low_train, _ = lowess_fit_predict(tr_o.x, tr_o.y, k=10, kernel_name="quartic")
from case_6.lowess import lowess_predict_query  # noqa: E402

low_grid = lowess_predict_query(tr_o.x, low_train, x_grid, k=10, kernel_name="quartic")
plt.figure(figsize=(9, 5))
plt.scatter(tr_o.x[:, 0], tr_o.y, s=12, alpha=0.35, label="наблюдения с выбросами")
plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), label="истинная sin(x)", linewidth=2)
plt.plot(x_grid[:, 0], nw_grid, label="обычная ядерная регрессия", linewidth=1.6)
plt.plot(x_grid[:, 0], low_grid, label="LOWESS", linewidth=1.6)
plt.title("Одномерная задача: NW и LOWESS при выбросах")
plt.xlabel("x"); plt.ylabel("y"); plt.legend()
save(FIG / "onedim_lowess_outliers.png")

# real datasets metrics + final comparison
print("[case 6] real datasets benchmark")
real = run_real_dataset_benchmark(seed=42)
final_rows = []
for ds_name, scores in real.items():
    for model, m_obj in scores.items():
        final_rows.append([f"{ds_name}/{model}",
                           round(m_obj.mae, 5), round(m_obj.rmse, 5), round(m_obj.r2, 5)])
final_rows.append(["synthetic/best_fixed", round(mae(te.y, pred_best), 5),
                   round(rmse(te.y, pred_best), 5), round(r2(te.y, pred_best), 5)])
write_csv(TAB / "final_model_comparison.csv",
          ["scenario", "mae", "rmse", "r2"], final_rows)

# dataset description
dataset_desc = [
    ["case4_functional_synthetic", n_samples, n_grid, "scalar", "x(t) на [0,1] с шумом"],
    ["case6_sinusoidal", 240, 1, "scalar", "sin(x)+noise, доля выбросов 8%"],
]
try:
    db = load_diabetes_dataset(standardize=True)
    dataset_desc.append(["diabetes", db.x.shape[0], db.x.shape[1], "scalar", "стандартизованные признаки"])
except Exception:
    pass
try:
    cal = load_california_dataset(standardize=True, max_samples=3000)
    dataset_desc.append(["california_housing", cal.x.shape[0], cal.x.shape[1], "scalar", "стандартизованный sub-sample 3000"])
except Exception:
    pass
write_csv(TAB / "dataset_description.csv",
          ["dataset", "n_samples", "n_features", "target", "notes"], dataset_desc)

print("DONE")
print(f"  figures -> {FIG}")
print(f"  tables  -> {TAB}")
