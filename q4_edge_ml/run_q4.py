"""
Q4 entry point: trains, compresses, and accounts for the cost.

    python q4_edge_ml/run_q4.py              # full run, roughly 25-45 minutes
    python q4_edge_ml/run_q4.py --quick      # reduced epochs and folds, ~4 min

Results are written to outputs/metrics.json and outputs/*.png, so the numbers
can be read without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from q4_edge_ml.budget import (analyse, format_table, summarise as budget_summary,
                               verify_fixed_point)
from q4_edge_ml.data import build_dataset, random_window_folds, subject_folds
from q4_edge_ml.nn import Adam, build_model, gradient_check, huber_loss
from q4_edge_ml.quantize import calibrate_and_quantise, magnitude_prune
from q4_edge_ml.train import cross_validate, evaluate, summarise, train_one

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = Path(__file__).resolve().parent / "outputs"

C_FLOAT = "#0072B2"
C_INT8 = "#D55E00"
C_PRUNED = "#009E73"
C_LEAK = "#CC79A7"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False,
})


def prune_and_finetune(net, data, train_idx, val_idx, sparsity, mean, std,
                       epochs=25, lr=5e-4, seed=0):
    """Zero the smallest weights, then fine-tune with those weights held at zero.

    Fine-tuning after pruning is not optional. Removing a third of the weights
    from a network this small perturbs every downstream activation, and the
    remaining weights need to move to compensate; without it the reported
    accuracy cost of pruning is mostly the cost of *not having retrained*,
    which is a different and much less interesting measurement.

    The mask is re-applied after every optimiser step. Adam carries momentum,
    so a pruned weight left alone would drift back off zero within a few
    updates and the sparsity would silently evaporate.
    """
    rng = np.random.default_rng(seed)
    zeroed = magnitude_prune(net, sparsity)
    masks = [{k: (v != 0) for k, v in layer.params.items() if k == "w"}
             for layer in net.layers]

    x_train, y_train = data.x[train_idx], (data.y[train_idx] - mean) / std
    x_val, y_val = data.x[val_idx], data.y[val_idx]
    optimiser = Adam(net, lr=lr)

    best = {"mae": np.inf, "state": net.state()}
    for _ in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            _, grad = huber_loss(net.forward(x_train[batch], training=True),
                                 y_train[batch])
            net.backward(grad)
            optimiser.step()
            for layer, mask in zip(net.layers, masks):
                for name, m in mask.items():
                    layer.params[name] *= m

        pred = net.forward(x_val, training=False).ravel() * std + mean
        mae = float(np.mean(np.abs(pred - y_val)))
        if mae < best["mae"]:
            best = {"mae": mae, "state": net.state()}

    net.load_state(best["state"])
    for layer, mask in zip(net.layers, masks):
        for name, m in mask.items():
            layer.params[name] *= m
    return zeroed


def stage_correctness(results: dict) -> None:
    print("[1/5] Correctness gates")
    grad = gradient_check()
    fixed = verify_fixed_point()
    print(f"  analytic vs numerical gradients: max relative error "
          f"{grad['max']:.2e} ({'PASS' if grad['passed'] else 'FAIL'})")
    print(f"  fixed-point requantiser vs real multiplier: max relative error "
          f"{fixed['max_relative_error']:.2e}")
    results["correctness"] = {"gradient_check": grad, "fixed_point": fixed}
    if not grad["passed"]:
        raise SystemExit("gradient check failed -- refusing to report results")


def stage_main(data, folds, epochs, results, sparsity=0.5) -> list:
    """Train once per fold, then evaluate float, int8 and pruned+int8."""
    print(f"\n[2/5] Subject-disjoint cross-validation ({len(folds)} folds)")
    print("      the honest split: no subject appears in both train and test")

    rows = []
    for i, (train_idx, val_idx, test_idx) in enumerate(folds):
        net, info = train_one(data, train_idx, val_idx, epochs=epochs, seed=i)
        mean, std = info["target_mean"], info["target_std"]
        subject = np.unique(data.subject[test_idx])[0]

        float_metrics = evaluate(net, data.x[test_idx], data.y[test_idx], mean, std)

        int8_net = calibrate_and_quantise(net, data.x[train_idx], mean, std)
        int8_pred = int8_net.predict(data.x[test_idx])
        int8_mae = float(np.mean(np.abs(int8_pred - data.y[test_idx])))

        float_pred = net.forward(data.x[test_idx], training=False).ravel() * std + mean
        agreement = float(np.max(np.abs(int8_pred - float_pred)))

        zeroed = prune_and_finetune(net, data, train_idx, val_idx, sparsity,
                                    mean, std, epochs=max(epochs // 5, 5), seed=i)
        pruned_metrics = evaluate(net, data.x[test_idx], data.y[test_idx], mean, std)
        pruned_int8 = calibrate_and_quantise(net, data.x[train_idx], mean, std)
        pruned_int8_mae = float(np.mean(
            np.abs(pruned_int8.predict(data.x[test_idx]) - data.y[test_idx])))

        rows.append({
            "fold": i, "subject": subject, "n_test": float_metrics["n"],
            "float32_mae": float_metrics["mae"], "float32_rmse": float_metrics["rmse"],
            "int8_mae": int8_mae,
            "pruned_float_mae": pruned_metrics["mae"],
            "pruned_int8_mae": pruned_int8_mae,
            "weights_zeroed": zeroed,
            "max_pred_disagreement_bpm": agreement,
        })
        print(f"  fold {i:2d} test={subject:4s} n={float_metrics['n']:4d}  "
              f"float32 {float_metrics['mae']:6.2f}  int8 {int8_mae:6.2f}  "
              f"pruned+int8 {pruned_int8_mae:6.2f}")

    def agg(key):
        values = np.array([r[key] for r in rows])
        return {"mean": round(float(values.mean()), 3),
                "std": round(float(values.std()), 3),
                "worst": round(float(values.max()), 3),
                "best": round(float(values.min()), 3)}

    summary = {k: agg(k) for k in
               ("float32_mae", "int8_mae", "pruned_float_mae", "pruned_int8_mae")}
    summary["max_pred_disagreement_bpm"] = round(
        float(np.max([r["max_pred_disagreement_bpm"] for r in rows])), 3)

    print(f"\n  {'variant':18s} {'MAE mean':>9s} {'sd':>7s} {'worst subj':>11s}")
    for key, label in (("float32_mae", "float32"), ("int8_mae", "int8 PTQ"),
                       ("pruned_float_mae", f"pruned {sparsity:.0%} fp32"),
                       ("pruned_int8_mae", f"pruned {sparsity:.0%} + int8")):
        s = summary[key]
        print(f"  {label:18s} {s['mean']:9.2f} {s['std']:7.2f} {s['worst']:11.2f}")

    results["subject_disjoint"] = {"per_fold": rows, "summary": summary,
                                   "sparsity": sparsity}
    return rows


def stage_leakage(data, epochs, results, n_folds) -> None:
    """The same model, evaluated the wrong way, to measure how much it flatters."""
    print(f"\n[3/5] Leakage demonstration: random per-window split")
    print("      identical model and training; only the split changes")

    folds = random_window_folds(data, n_folds=n_folds)
    leaky = cross_validate(data, folds, epochs=epochs, label="  ", verbose=False)
    stats = summarise(leaky)
    honest = results["subject_disjoint"]["summary"]["float32_mae"]["mean"]

    print(f"  random per-window split MAE : {stats['mae_mean']:6.2f} bpm")
    print(f"  subject-disjoint split MAE  : {honest:6.2f} bpm")
    print(f"  the leaky split understates the error by "
          f"{honest - stats['mae_mean']:.2f} bpm "
          f"({100 * (honest - stats['mae_mean']) / honest:.0f}% of the true value)")

    results["random_window_split"] = {
        "summary": stats,
        "understatement_bpm": round(honest - stats["mae_mean"], 3),
        "understatement_fraction": round((honest - stats["mae_mean"]) / honest, 4),
    }


def stage_budget(results: dict) -> None:
    print("\n[4/5] Microcontroller budget (Cortex-M4F @ 80 MHz assumed)")
    net = build_model()
    costs = analyse(net, (2, 200))
    print(format_table(costs))
    budget = budget_summary(costs, input_bytes=2 * 200)

    print(f"\n  flash, model only      int8 {budget['flash_model_int8_bytes']:,} B"
          f"   float32 {budget['flash_model_float32_bytes']:,} B")
    print(f"  flash, with runtime    {budget['flash_with_runtime_bytes'][0]:,}"
          f" - {budget['flash_with_runtime_bytes'][1]:,} B")
    print(f"  peak activation RAM    {budget['ram_total_bytes']:,} B")
    print(f"  latency per inference  int8 {budget['latency_ms']['int8_pessimistic']:.2f}"
          f" ms (pessimistic), float32 "
          f"{budget['latency_ms']['float32_pessimistic']:.2f} ms")
    print(f"  CPU duty cycle         {100 * budget['cpu_duty_cycle']:.3f}%"
          f"  ({budget['energy_per_inference_uj']:.0f} uJ per inference)")

    results["budget"] = budget
    results["budget"]["per_layer"] = [
        {"layer": c.name, "output": list(c.output_shape), "macs": c.macs,
         "int8_bytes": c.weight_bytes_int8, "float32_bytes": c.weight_bytes_float32}
        for c in costs
    ]


def stage_width_sweep(data, folds, epochs, results, widths) -> None:
    """Where does the budget actually start to bind?

    Run on a reduced number of folds and labelled as such: the point is the
    shape of the size-accuracy curve, and spending an hour to resolve it more
    precisely would not change the conclusion.
    """
    print(f"\n[5/5] Width sweep ({len(folds)} folds each) -- finding the binding constraint")
    rows = []
    for width in widths:
        net = build_model(width=width)
        costs = analyse(net, (2, 200))
        budget = budget_summary(costs, input_bytes=2 * 200)

        maes = []
        for i, (train_idx, val_idx, test_idx) in enumerate(folds):
            trained, info = train_one(data, train_idx, val_idx, epochs=epochs,
                                      width=width, seed=i)
            int8_net = calibrate_and_quantise(trained, data.x[train_idx],
                                              info["target_mean"], info["target_std"])
            maes.append(float(np.mean(np.abs(
                int8_net.predict(data.x[test_idx]) - data.y[test_idx]))))

        rows.append({
            "width": width,
            "parameters": net.parameter_count(),
            "flash_int8_bytes": budget["flash_model_int8_bytes"],
            "ram_bytes": budget["ram_total_bytes"],
            "latency_ms": budget["latency_ms"]["int8_pessimistic"],
            "int8_mae_mean": round(float(np.mean(maes)), 3),
            "int8_mae_std": round(float(np.std(maes)), 3),
        })
        print(f"  width {width:4.2f}  params {net.parameter_count():6,d}  "
              f"flash {budget['flash_model_int8_bytes']:7,d} B  "
              f"RAM {budget['ram_total_bytes']:6,d} B  "
              f"{budget['latency_ms']['int8_pessimistic']:6.2f} ms  "
              f"MAE {np.mean(maes):6.2f}")

    results["width_sweep"] = {"n_folds": len(folds), "rows": rows}


def make_plots(results: dict) -> None:
    rows = results["subject_disjoint"]["per_fold"]
    subjects = [r["subject"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    ax = axes[0]
    x = np.arange(len(rows))
    width = 0.27
    ax.bar(x - width, [r["float32_mae"] for r in rows], width,
           color=C_FLOAT, label="float32")
    ax.bar(x, [r["int8_mae"] for r in rows], width, color=C_INT8, label="int8 PTQ")
    ax.bar(x + width, [r["pruned_int8_mae"] for r in rows], width,
           color=C_PRUNED, label="pruned + int8")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, fontsize=7)
    ax.set_ylabel("MAE (bpm)")
    ax.set_xlabel("held-out subject")
    ax.set_title("Error varies far more across people than across precisions",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    honest = results["subject_disjoint"]["summary"]["float32_mae"]
    leaky = results.get("random_window_split", {}).get("summary")
    labels = ["subject-disjoint\n(honest)", "random per-window\n(leaky)"]
    values = [honest["mean"], leaky["mae_mean"] if leaky else 0]
    errors = [honest["std"], leaky["mae_std"] if leaky else 0]
    ax.bar(labels, values, yerr=errors, capsize=5,
           color=[C_FLOAT, C_LEAK], width=0.55)
    for i, v in enumerate(values):
        ax.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("The same model, scored two ways", loc="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png")
    plt.close(fig)

    if "width_sweep" in results:
        sweep = results["width_sweep"]["rows"]
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        flash = [r["flash_int8_bytes"] / 1024 for r in sweep]
        mae = [r["int8_mae_mean"] for r in sweep]
        ax.errorbar(flash, mae, yerr=[r["int8_mae_std"] for r in sweep],
                    marker="o", color=C_FLOAT, capsize=4)
        for r, f, m in zip(sweep, flash, mae):
            ax.annotate(f"w={r['width']:g}", (f, m), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("int8 model size (KB, log scale)")
        ax.set_ylabel("MAE (bpm), subject-disjoint")
        ax.set_title("More capacity does not buy accuracy here", loc="left",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "width_sweep.png")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="fewer epochs and folds; for checking it runs")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict = {}

    epochs = 25 if args.quick else 100
    print("Loading TROIKA and building windows...")
    data = build_dataset()
    print(f"  {len(data)} windows, {len(np.unique(data.subject))} subjects, "
          f"input {data.x.shape[1]}x{data.x.shape[2]} "
          f"(8 s at 25 Hz), heart rate {data.y.min():.0f}-{data.y.max():.0f} bpm")
    results["dataset"] = {
        "n_windows": len(data),
        "n_subjects": int(len(np.unique(data.subject))),
        "input_shape": list(data.x.shape[1:]),
        "bpm_range": [float(data.y.min()), float(data.y.max())],
    }

    folds = subject_folds(data)
    if args.quick:
        folds = folds[:3]

    stage_correctness(results)
    stage_main(data, folds, epochs, results)
    stage_leakage(data, epochs, results, n_folds=len(folds))
    stage_budget(results)
    stage_width_sweep(data, folds[:3], epochs, results,
                      widths=[0.5, 1.0, 2.0] if not args.quick else [0.5, 1.0])
    make_plots(results)

    results["runtime_seconds"] = round(time.time() - started, 1)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT / 'metrics.json'} and figures "
          f"({results['runtime_seconds'] / 60:.1f} min).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
