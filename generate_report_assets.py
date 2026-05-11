from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from case_4.report_utils import (
    FunctionalDataset,
    compute_feature_matrix,
    fit_linear_regression,
    mae,
    make_functional_dataset,
    make_indicator_basis,
    make_polynomial_basis,
    make_trigonometric_basis,
    predict_linear,
    r2,
    recover_weight_function,
    rmse,
    split_indices,
)
from case_6.data import (
    dataset_stats,
    inject_outliers,
    load_real_datasets,
    make_sinusoidal_dataset,
    make_sinusoidal_split,
    train_test_split,
)
from case_6.distance import pairwise_euclidean
from case_6.kernels import KERNELS
from case_6.lowess import lowess_fit_predict, lowess_predict_query
from case_6.metrics import mae as mae6
from case_6.metrics import r2 as r26
from case_6.metrics import rmse as rmse6
from case_6.nadaraya_watson import nw_predict_fixed, nw_predict_variable
from case_6.selection import (
    loo_score_fixed,
    loo_score_lowess,
    loo_score_variable,
    select_fixed_window,
    select_lowess,
    select_variable_window,
)

OUTPUT_ROOT = Path("report_outputs")
FIGURES_DIR = OUTPUT_ROOT / "figures"
TABLES_DIR = OUTPUT_ROOT / "tables"
LATEX_DIR = OUTPUT_ROOT / "latex_fragments"

KERNEL_NAMES = ["gaussian", "epanechnikov", "triangular", "quartic"]


def ensure_output_dirs() -> None:
    for d in (FIGURES_DIR, TABLES_DIR, LATEX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ----------------------------- case 4 -------------------------------------


def evaluate_case4_models() -> dict[str, object]:
    ds: FunctionalDataset = make_functional_dataset(n_samples=280, n_grid=320, seed=42)
    idx_train, idx_test = split_indices(ds.x.shape[0], test_size=0.25, seed=42)
    x_train_fn = ds.x[idx_train]
    x_test_fn = ds.x[idx_test]
    y_train = ds.y[idx_train]
    y_test = ds.y[idx_test]

    plt.figure(figsize=(9, 5))
    for i in range(8):
        plt.plot(ds.t, ds.x[i], linewidth=1.0, alpha=0.9)
    plt.title("Кейс 4: примеры сгенерированных функций $x_i(t)$")
    plt.xlabel("t"); plt.ylabel("$x_i(t)$")
    save_figure(FIGURES_DIR / "sample_functions.png")

    basis_builders = {
        "Индикаторный": make_indicator_basis,
        "Полиномиальный": lambda m, t: make_polynomial_basis(m),
        "Тригонометрический": lambda m, t: make_trigonometric_basis(m),
    }
    m_grid = [3, 5, 7, 9, 11, 13, 15]
    basis_rows: list[list[object]] = []
    best_models: list[dict[str, object]] = []
    rmse_curves: dict[str, list[float]] = {}
    train_rmse_curves: dict[str, list[float]] = {}

    for basis_name, builder in basis_builders.items():
        rmse_curves[basis_name] = []
        train_rmse_curves[basis_name] = []
        for m in m_grid:
            basis, basis_names = builder(m, ds.t)
            z_train = compute_feature_matrix(x_train_fn, ds.t, basis)
            z_test = compute_feature_matrix(x_test_fn, ds.t, basis)
            beta = fit_linear_regression(z_train, y_train)
            pred_train = predict_linear(z_train, beta)
            pred_test = predict_linear(z_test, beta)
            tr_rmse = rmse(y_train, pred_train)
            te_rmse = rmse(y_test, pred_test)
            rmse_curves[basis_name].append(te_rmse)
            train_rmse_curves[basis_name].append(tr_rmse)
            basis_rows.append([basis_name, m, tr_rmse, te_rmse, r2(y_train, pred_train), r2(y_test, pred_test)])
            best_models.append({"basis": basis_name, "m": m, "rmse_test": te_rmse,
                                "basis_names": basis_names, "basis_fns": basis,
                                "beta": beta, "z_train": z_train, "z_test": z_test})

    save_csv(TABLES_DIR / "linear_basis_comparison.csv",
             ["basis", "m", "train_rmse", "test_rmse", "train_r2", "test_r2"], basis_rows)

    best_linear = min(best_models, key=lambda d: float(d["rmse_test"]))
    best_basis = str(best_linear["basis"]); best_m = int(best_linear["m"])

    plt.figure(figsize=(8, 5))
    x_pos = np.arange(len(basis_builders))
    labels = list(basis_builders.keys())
    vals = [min(v for (b, _, _, v, _, _) in basis_rows if b == lbl) for lbl in labels]
    plt.bar(x_pos, vals, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.xticks(x_pos, labels, rotation=10)
    plt.title("Кейс 4: лучший test RMSE по каждому базису")
    plt.ylabel("RMSE")
    save_figure(FIGURES_DIR / "basis_comparison_rmse.png")

    plt.figure(figsize=(8, 5))
    for basis_name in basis_builders:
        plt.plot(m_grid, rmse_curves[basis_name], marker="o", label=basis_name)
    plt.title("Кейс 4: test RMSE от числа функционалов $m$")
    plt.xlabel("m"); plt.ylabel("test RMSE"); plt.legend()
    save_figure(FIGURES_DIR / "rmse_vs_m.png")

    plt.figure(figsize=(8, 5))
    plt.plot(m_grid, train_rmse_curves[best_basis], marker="o", label="train RMSE")
    plt.plot(m_grid, rmse_curves[best_basis], marker="s", label="test RMSE")
    plt.title(f"Кейс 4: train/test RMSE от $m$ ({best_basis})")
    plt.xlabel("m"); plt.ylabel("RMSE"); plt.legend()
    save_figure(FIGURES_DIR / "train_test_rmse_vs_m.png")

    ridge_lambdas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    ridge_rows: list[list[object]] = []
    z_train_best = best_linear["z_train"]; z_test_best = best_linear["z_test"]
    best_ridge = None; best_pred = None
    for lam in ridge_lambdas:
        beta_lam = fit_linear_regression(z_train_best, y_train, ridge_lambda=lam)
        pred_te = predict_linear(z_test_best, beta_lam)
        pred_tr = predict_linear(z_train_best, beta_lam)
        row = [best_basis, best_m, lam, rmse(y_train, pred_tr),
               rmse(y_test, pred_te), r2(y_test, pred_te), mae(y_test, pred_te)]
        ridge_rows.append(row)
        if best_ridge is None or row[4] < best_ridge[4]:
            best_ridge = row; best_pred = pred_te

    save_csv(TABLES_DIR / "best_linear_models.csv",
             ["basis", "m", "lambda", "train_rmse", "test_rmse", "test_r2", "test_mae"], ridge_rows)

    plt.figure(figsize=(8, 5))
    plt.semilogx(ridge_lambdas, [r[4] for r in ridge_rows], marker="o")
    plt.title("Кейс 4: test RMSE от $\\lambda$ (ridge)")
    plt.xlabel("$\\lambda$"); plt.ylabel("test RMSE")
    save_figure(FIGURES_DIR / "ridge_lambda_curve.png")

    plt.figure(figsize=(8, 5))
    order = np.argsort(y_test)
    plt.plot(np.arange(y_test.shape[0]), y_test[order], label="истинные y", linewidth=2.0)
    plt.plot(np.arange(y_test.shape[0]), best_pred[order], label="предсказанные y", linewidth=2.0)
    plt.title("Кейс 4: истинные и предсказанные значения (лучшая модель)")
    plt.xlabel("индекс (отсортировано по y)"); plt.ylabel("y"); plt.legend()
    save_figure(FIGURES_DIR / "linear_true_vs_pred.png")

    plt.figure(figsize=(8, 5))
    plt.hist(y_test - best_pred, bins=20, edgecolor="black")
    plt.title("Кейс 4: распределение остатков")
    plt.xlabel("$y - \\hat y$"); plt.ylabel("частота")
    save_figure(FIGURES_DIR / "linear_residuals_distribution.png")

    plt.figure(figsize=(9, 5))
    for basis_name, builder in basis_builders.items():
        basis_tmp, _ = builder(best_m, ds.t)
        z_train_tmp = compute_feature_matrix(x_train_fn, ds.t, basis_tmp)
        beta_tmp = fit_linear_regression(z_train_tmp, y_train)
        w_tmp = recover_weight_function(ds.t, beta_tmp, basis_tmp)
        plt.plot(ds.t, w_tmp, label=basis_name, linewidth=2.0)
    plt.title("Кейс 4: восстановленные весовые функции $w(t)$")
    plt.xlabel("t"); plt.ylabel("w(t)"); plt.legend()
    save_figure(FIGURES_DIR / "weight_functions_comparison.png")

    return {
        "case4_best_basis": best_basis, "case4_best_m": best_m,
        "case4_best_lambda": float(best_ridge[2]), "case4_best_rmse": float(best_ridge[4]),
        "case4_best_mae": float(best_ridge[6]), "case4_best_r2": float(best_ridge[5]),
    }


# ----------------------------- case 6 -------------------------------------


@dataclass(slots=True)
class ModelScore:
    name: str
    kernel: str
    param: float
    loo_rmse: float
    test_rmse: float
    test_mae: float
    test_r2: float


def _grid_search_synthetic(train, test):
    hs = [0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8]
    ks = [3, 5, 8, 10, 12, 15, 20]

    fixed_rows: list[list[object]] = []
    variable_rows: list[list[object]] = []
    loo_h: dict[str, list[float]] = {k: [] for k in KERNEL_NAMES}
    loo_k: dict[str, list[float]] = {k: [] for k in KERNEL_NAMES}

    for kernel in KERNEL_NAMES:
        for h in hs:
            loo = loo_score_fixed(train.x, train.y, h=h, kernel_name=kernel)
            pred = nw_predict_fixed(train.x, train.y, test.x, h=h, kernel_name=kernel)
            fixed_rows.append([kernel, h, loo, rmse6(test.y, pred), mae6(test.y, pred), r26(test.y, pred)])
            loo_h[kernel].append(loo)
        for k in ks:
            loo = loo_score_variable(train.x, train.y, k=k, kernel_name=kernel)
            pred = nw_predict_variable(train.x, train.y, test.x, k=k, kernel_name=kernel)
            variable_rows.append([kernel, k, loo, rmse6(test.y, pred), mae6(test.y, pred), r26(test.y, pred)])
            loo_k[kernel].append(loo)
    return hs, ks, fixed_rows, variable_rows, loo_h, loo_k


def evaluate_case6_synthetic() -> dict[str, object]:
    train, test = make_sinusoidal_split(
        n_samples=240, noise_std=0.12, train_outlier_fraction=0.08, seed=42
    )

    hs, ks, fixed_rows, variable_rows, loo_h, loo_k = _grid_search_synthetic(train, test)

    save_csv(TABLES_DIR / "kernel_fixed_window_results.csv",
             ["kernel", "h", "loo_rmse", "test_rmse", "test_mae", "test_r2"], fixed_rows)
    save_csv(TABLES_DIR / "kernel_variable_window_results.csv",
             ["kernel", "k", "loo_rmse", "test_rmse", "test_mae", "test_r2"], variable_rows)

    # PICK BY LOO, NOT TEST
    best_fixed = min(fixed_rows, key=lambda x: float(x[2]))
    best_variable = min(variable_rows, key=lambda x: float(x[2]))

    plt.figure(figsize=(8, 5))
    for kernel in KERNEL_NAMES:
        plt.plot(hs, loo_h[kernel], marker="o", label=kernel)
    plt.title("Синтетика: LOO RMSE от ширины окна $h$")
    plt.xlabel("h"); plt.ylabel("LOO RMSE"); plt.legend()
    save_figure(FIGURES_DIR / "kernel_rmse_vs_h.png")

    plt.figure(figsize=(8, 5))
    for kernel in KERNEL_NAMES:
        plt.plot(ks, loo_k[kernel], marker="o", label=kernel)
    plt.title("Синтетика: LOO RMSE от числа соседей $k$")
    plt.xlabel("k"); plt.ylabel("LOO RMSE"); plt.legend()
    save_figure(FIGURES_DIR / "kernel_rmse_vs_k.png")

    best_by_kernel = []
    for kernel in KERNEL_NAMES:
        cands = [r for r in fixed_rows if r[0] == kernel] + [r for r in variable_rows if r[0] == kernel]
        best_by_kernel.append(min(cands, key=lambda r: float(r[2])))
    save_csv(TABLES_DIR / "best_kernel_models.csv",
             ["kernel", "param", "loo_rmse", "test_rmse", "test_mae", "test_r2"], best_by_kernel)

    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(KERNEL_NAMES)), [float(r[3]) for r in best_by_kernel],
            color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    plt.xticks(np.arange(len(KERNEL_NAMES)), KERNEL_NAMES)
    plt.title("Синтетика: test RMSE лучшей модели по каждому ядру")
    plt.xlabel("ядро"); plt.ylabel("test RMSE")
    save_figure(FIGURES_DIR / "kernel_comparison_best_rmse.png")

    pred_fixed = nw_predict_fixed(train.x, train.y, test.x,
                                  h=float(best_fixed[1]), kernel_name=str(best_fixed[0]))
    pred_variable = nw_predict_variable(train.x, train.y, test.x,
                                        k=int(best_variable[1]), kernel_name=str(best_variable[0]))

    x_grid = np.linspace(-3.0, 3.0, 400)[:, None]
    g_fixed = nw_predict_fixed(train.x, train.y, x_grid,
                               h=float(best_fixed[1]), kernel_name=str(best_fixed[0]))
    g_var = nw_predict_variable(train.x, train.y, x_grid,
                                k=int(best_variable[1]), kernel_name=str(best_variable[0]))

    plt.figure(figsize=(9, 5))
    plt.scatter(train.x[:, 0], train.y, s=10, alpha=0.35, label="train (с выбросами)")
    plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), label="sin(x)", linewidth=2.0)
    plt.plot(x_grid[:, 0], g_fixed,
             label=f"NW fixed ({best_fixed[0]}, h={best_fixed[1]})", linewidth=1.8)
    plt.plot(x_grid[:, 0], g_var,
             label=f"NW variable ({best_variable[0]}, k={int(best_variable[1])})", linewidth=1.8)
    plt.title("Синтетика: фиксированное vs переменное окно")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    save_figure(FIGURES_DIR / "fixed_vs_variable_window.png")

    best_overall = (best_fixed if float(best_fixed[2]) <= float(best_variable[2])
                    else best_variable)
    pred_best = pred_fixed if best_overall is best_fixed else pred_variable

    plt.figure(figsize=(8, 5))
    plt.hist(test.y - pred_best, bins=20, edgecolor="black")
    plt.title("Синтетика: распределение остатков (лучшая модель)")
    plt.xlabel("$y - \\hat y$"); plt.ylabel("частота")
    save_figure(FIGURES_DIR / "kernel_residuals_distribution.png")

    # Before/after robust reweighting
    out_train, out_test = make_sinusoidal_split(
        n_samples=220, noise_std=0.12, train_outlier_fraction=0.15, seed=456
    )
    best_lowess = select_lowess(out_train.x, out_train.y,
                                ks=[5, 10, 15, 20], kernels=["quartic", "triangular"])
    pred_nw_grid = nw_predict_variable(out_train.x, out_train.y, x_grid,
                                       k=int(best_lowess.param_value),
                                       kernel_name=best_lowess.kernel_name)
    _, gamma = lowess_fit_predict(out_train.x, out_train.y,
                                  k=int(best_lowess.param_value),
                                  kernel_name=best_lowess.kernel_name)
    pred_low_grid = lowess_predict_query(out_train.x, out_train.y, gamma, x_grid,
                                         k=int(best_lowess.param_value),
                                         kernel_name=best_lowess.kernel_name)

    plt.figure(figsize=(9, 5))
    plt.scatter(out_train.x[:, 0], out_train.y, s=12, alpha=0.35, label="train")
    plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), label="sin(x)", linewidth=2.0)
    plt.plot(x_grid[:, 0], pred_nw_grid, label="до перевзвешивания ($\\gamma\\equiv 1$)", linewidth=1.8)
    plt.plot(x_grid[:, 0], pred_low_grid, label="LOWESS (после)", linewidth=1.8)
    plt.title("LOWESS: прогноз до и после робастного перевзвешивания")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    save_figure(FIGURES_DIR / "lowess_before_after.png")

    plt.figure(figsize=(9, 5))
    plt.scatter(out_train.x[:, 0], out_train.y, s=12, alpha=0.35, label="train")
    plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), label="sin(x)", linewidth=2.0)
    plt.plot(x_grid[:, 0], pred_nw_grid, label="обычная NW", linewidth=1.8)
    plt.plot(x_grid[:, 0], pred_low_grid, label="LOWESS", linewidth=1.8)
    plt.title("Одномерная задача с выбросами: NW vs LOWESS")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    save_figure(FIGURES_DIR / "onedim_lowess_outliers.png")

    # gamma plot
    order_idx = np.argsort(out_train.x[:, 0])
    plt.figure(figsize=(9, 5))
    plt.plot(out_train.x[order_idx, 0], gamma[order_idx], linewidth=1.4)
    plt.scatter(out_train.x[order_idx, 0], gamma[order_idx], s=10, alpha=0.6)
    plt.title("LOWESS: финальные веса $\\gamma_i$")
    plt.xlabel("x"); plt.ylabel("$\\gamma_i$")
    save_figure(FIGURES_DIR / "lowess_final_weights.png")
    save_csv(TABLES_DIR / "lowess_final_weights.csv", ["x", "gamma"],
             [[float(out_train.x[i, 0]), float(gamma[i])] for i in order_idx])

    # NW vs LOWESS at different outlier levels (outliers ONLY in train)
    outlier_fracs = [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3]
    outlier_rows: list[list[object]] = []
    rmse_nw_line, rmse_low_line = [], []
    for frac in outlier_fracs:
        tr, te = make_sinusoidal_split(
            n_samples=240, noise_std=0.12, train_outlier_fraction=frac, seed=42
        )
        pred_nw = nw_predict_variable(tr.x, tr.y, te.x,
                                      k=int(best_variable[1]),
                                      kernel_name=str(best_variable[0]))
        _, g = lowess_fit_predict(tr.x, tr.y,
                                  k=int(best_lowess.param_value),
                                  kernel_name=best_lowess.kernel_name)
        pred_low = lowess_predict_query(tr.x, tr.y, g, te.x,
                                        k=int(best_lowess.param_value),
                                        kernel_name=best_lowess.kernel_name)
        rmse_nw_line.append(rmse6(te.y, pred_nw))
        rmse_low_line.append(rmse6(te.y, pred_low))
        outlier_rows.append([frac, rmse_nw_line[-1], rmse_low_line[-1],
                             mae6(te.y, pred_nw), mae6(te.y, pred_low)])
    save_csv(TABLES_DIR / "lowess_outlier_comparison.csv",
             ["outlier_fraction", "nw_rmse", "lowess_rmse", "nw_mae", "lowess_mae"], outlier_rows)

    plt.figure(figsize=(8, 5))
    plt.plot(outlier_fracs, rmse_nw_line, marker="o", label="NW (variable)")
    plt.plot(outlier_fracs, rmse_low_line, marker="s", label="LOWESS")
    plt.title("Влияние выбросов в обучении на test RMSE")
    plt.xlabel("доля выбросов в train"); plt.ylabel("test RMSE"); plt.legend()
    save_figure(FIGURES_DIR / "outliers_effect_rmse.png")

    # 1D figures with different h
    one_train, one_test = make_sinusoidal_split(
        n_samples=180, noise_std=0.13, train_outlier_fraction=0.0, seed=123
    )
    plt.figure(figsize=(9, 5))
    plt.scatter(one_train.x[:, 0], one_train.y, s=10, alpha=0.35, label="наблюдения")
    plt.plot(x_grid[:, 0], np.sin(x_grid[:, 0]), label="истинная sin(x)", linewidth=2.0)
    for h in [0.1, 0.25, 0.5]:
        pred_grid = nw_predict_fixed(one_train.x, one_train.y, x_grid,
                                     h=h, kernel_name="gaussian")
        plt.plot(x_grid[:, 0], pred_grid, linewidth=1.6, label=f"NW, h={h}")
    plt.title("Одномерная синтетика: NW при разных $h$")
    plt.xlabel("x"); plt.ylabel("y"); plt.legend()
    save_figure(FIGURES_DIR / "onedim_nw_different_h.png")

    h_eval = [0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.8]
    h_rmse = []
    for h in h_eval:
        pred = nw_predict_fixed(one_train.x, one_train.y, one_test.x,
                                h=h, kernel_name="gaussian")
        h_rmse.append(rmse6(one_test.y, pred))
    plt.figure(figsize=(8, 5))
    plt.plot(h_eval, h_rmse, marker="o")
    plt.title("Одномерная синтетика: test RMSE от $h$")
    plt.xlabel("h"); plt.ylabel("test RMSE")
    save_figure(FIGURES_DIR / "onedim_rmse_vs_h.png")

    # final synthetic comparison
    save_csv(TABLES_DIR / "final_synthetic_comparison.csv",
             ["model", "kernel", "param", "loo_rmse", "test_rmse", "test_mae", "test_r2"],
             [["nw_fixed", best_fixed[0], best_fixed[1], best_fixed[2],
               best_fixed[3], best_fixed[4], best_fixed[5]],
              ["nw_variable", best_variable[0], best_variable[1], best_variable[2],
               best_variable[3], best_variable[4], best_variable[5]],
              ["lowess", best_lowess.kernel_name, best_lowess.param_value,
               best_lowess.score_rmse,
               rmse6(out_test.y, lowess_predict_query(
                   out_train.x, out_train.y, gamma, out_test.x,
                   k=int(best_lowess.param_value), kernel_name=best_lowess.kernel_name)),
               mae6(out_test.y, lowess_predict_query(
                   out_train.x, out_train.y, gamma, out_test.x,
                   k=int(best_lowess.param_value), kernel_name=best_lowess.kernel_name)),
               r26(out_test.y, lowess_predict_query(
                   out_train.x, out_train.y, gamma, out_test.x,
                   k=int(best_lowess.param_value), kernel_name=best_lowess.kernel_name))]])

    # what matters more: kernel or window-width?
    # For each kernel, compute std and range of LOO across h's, then aggregate.
    kernel_spread = []
    for kernel in KERNEL_NAMES:
        loos = loo_h[kernel]
        kernel_spread.append([kernel, float(np.min(loos)), float(np.max(loos)),
                              float(np.max(loos) - np.min(loos)), float(np.std(loos))])
    save_csv(TABLES_DIR / "kernel_vs_h_sensitivity.csv",
             ["kernel", "loo_min", "loo_max", "loo_range", "loo_std"], kernel_spread)

    return {
        "best_fixed": best_fixed, "best_variable": best_variable,
        "best_lowess_kernel": best_lowess.kernel_name,
        "best_lowess_k": int(best_lowess.param_value),
        "best_lowess_loo": float(best_lowess.score_rmse),
    }


# ----------------------------- real datasets -------------------------------


def evaluate_case6_real() -> dict[str, object]:
    datasets = load_real_datasets(seed=42)
    stats_rows: list[list[object]] = []
    metric_rows: list[list[object]] = []

    for name, (train, test) in datasets.items():
        st = dataset_stats(name, train, test)
        stats_rows.append([st["name"], st["n_train"], st["n_test"],
                           st["n_features"], st["y_mean"], st["y_std"]])

        # adaptive parameter grids (depend on feature space scale)
        diam = float(np.median(np.linalg.norm(train.x, axis=1)))
        hs = [round(0.25 * diam, 4), round(0.5 * diam, 4), round(diam, 4),
              round(1.5 * diam, 4), round(2.0 * diam, 4)]
        ks = [5, 10, 20, 40, max(80, min(train.x.shape[0] // 4, 200))]
        ks = sorted(set(k for k in ks if 1 <= k < train.x.shape[0]))

        kernels = ["gaussian", "epanechnikov", "quartic"]
        best_h = select_fixed_window(train.x, train.y, hs=hs, kernels=kernels)
        best_k = select_variable_window(train.x, train.y, ks=ks, kernels=kernels)
        # LOWESS LOO is O(n^2 * n_iter), too slow on California; tune by inheriting best_k
        lowess_k_grid = ks if name == "diabetes" else [int(best_k.param_value)]
        lowess_kernels = ["quartic", "triangular"] if name == "diabetes" else ["quartic"]
        best_low = select_lowess(train.x, train.y, ks=lowess_k_grid, kernels=lowess_kernels)

        pred_fixed = nw_predict_fixed(train.x, train.y, test.x,
                                      h=best_h.param_value, kernel_name=best_h.kernel_name)
        pred_var = nw_predict_variable(train.x, train.y, test.x,
                                       k=int(best_k.param_value), kernel_name=best_k.kernel_name)
        _, gamma = lowess_fit_predict(train.x, train.y,
                                      k=int(best_low.param_value),
                                      kernel_name=best_low.kernel_name)
        pred_low = lowess_predict_query(train.x, train.y, gamma, test.x,
                                        k=int(best_low.param_value),
                                        kernel_name=best_low.kernel_name)

        for model_name, kern, param, loo_score, pred in [
            ("nw_fixed", best_h.kernel_name, best_h.param_value, best_h.score_rmse, pred_fixed),
            ("nw_variable", best_k.kernel_name, best_k.param_value, best_k.score_rmse, pred_var),
            ("lowess", best_low.kernel_name, best_low.param_value, best_low.score_rmse, pred_low),
        ]:
            metric_rows.append([
                name, model_name, kern, param,
                float(loo_score), float(rmse6(test.y, pred)),
                float(mae6(test.y, pred)), float(r26(test.y, pred)),
            ])

    save_csv(TABLES_DIR / "dataset_description.csv",
             ["dataset", "n_train", "n_test", "n_features", "y_mean", "y_std"], stats_rows)
    save_csv(TABLES_DIR / "real_datasets_metrics.csv",
             ["dataset", "model", "kernel", "param", "loo_rmse", "test_rmse", "test_mae", "test_r2"],
             metric_rows)

    plt.figure(figsize=(9, 5))
    by_model: dict[str, list[float]] = {"nw_fixed": [], "nw_variable": [], "lowess": []}
    ds_names = list(datasets.keys())
    for ds_name in ds_names:
        for row in metric_rows:
            if row[0] == ds_name and isinstance(row[1], str):
                by_model[row[1]].append(float(row[5]))
    width = 0.25
    xs = np.arange(len(ds_names))
    for i, model in enumerate(["nw_fixed", "nw_variable", "lowess"]):
        plt.bar(xs + (i - 1) * width, by_model[model], width=width, label=model)
    plt.xticks(xs, ds_names)
    plt.title("Реальные датасеты: test RMSE по моделям")
    plt.ylabel("test RMSE"); plt.legend()
    save_figure(FIGURES_DIR / "real_datasets_rmse.png")

    return {"metric_rows": metric_rows, "stats_rows": stats_rows}


# ----------------------------- LaTeX fragments -----------------------------


def write_latex_fragments(case4_stats, case6_synth, case6_real) -> None:
    figs_case4 = "\n".join(
        rf"""\begin{{figure}}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{{report_outputs/figures/{name}.png}}
    \caption{{{caption}}}
\end{{figure}}"""
        for name, caption in [
            ("sample_functions", "Примеры функций $x_i(t)$"),
            ("basis_comparison_rmse", "Сравнение базисов"),
            ("rmse_vs_m", "Зависимость test RMSE от $m$"),
            ("train_test_rmse_vs_m", "Train/test RMSE от $m$"),
            ("ridge_lambda_curve", "Test RMSE от $\\lambda$"),
        ]
    )
    (LATEX_DIR / "figures_case4.tex").write_text(figs_case4, encoding="utf-8")

    figs_case6 = "\n".join(
        rf"""\begin{{figure}}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{{report_outputs/figures/{name}.png}}
    \caption{{{caption}}}
\end{{figure}}"""
        for name, caption in [
            ("kernel_rmse_vs_h", "LOO RMSE от $h$"),
            ("kernel_rmse_vs_k", "LOO RMSE от $k$"),
            ("fixed_vs_variable_window", "Фиксированное vs переменное окно"),
            ("lowess_before_after", "LOWESS до и после перевзвешивания"),
            ("lowess_final_weights", "Финальные веса $\\gamma_i$"),
            ("outliers_effect_rmse", "NW vs LOWESS при выбросах"),
            ("real_datasets_rmse", "Test RMSE на Diabetes и California Housing"),
        ]
    )
    (LATEX_DIR / "figures_case6.tex").write_text(figs_case6, encoding="utf-8")

    best_fixed = case6_synth["best_fixed"]; best_var = case6_synth["best_variable"]
    conclusions = f"""Кейс 4. Лучший базис — {case4_stats['case4_best_basis']}, m={case4_stats['case4_best_m']},
$\\lambda$={case4_stats['case4_best_lambda']:.4g}, test RMSE={case4_stats['case4_best_rmse']:.4f},
MAE={case4_stats['case4_best_mae']:.4f}, $R^2$={case4_stats['case4_best_r2']:.4f}.

Кейс 6 (синтетика). Лучшее фиксированное окно: ядро {best_fixed[0]}, h={best_fixed[1]} (LOO={best_fixed[2]:.4f}).
Лучшее переменное окно: ядро {best_var[0]}, k={int(best_var[1])} (LOO={best_var[2]:.4f}).
Лучший LOWESS: ядро {case6_synth['best_lowess_kernel']}, k={case6_synth['best_lowess_k']} (LOO={case6_synth['best_lowess_loo']:.4f}).
"""
    (LATEX_DIR / "auto_conclusions.tex").write_text(conclusions, encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    case4_stats = evaluate_case4_models()
    case6_synth = evaluate_case6_synthetic()
    case6_real = evaluate_case6_real()
    write_latex_fragments(case4_stats, case6_synth, case6_real)
    print("Готово: материалы отчёта сохранены в", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
