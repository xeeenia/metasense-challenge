"""
Q3 entry point: runs every experiment and writes results to outputs/.

    python q3_signal_quality/run_q3.py

Everything it prints is also written to outputs/metrics.json, and every figure
to outputs/*.png, so the results can be read without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# Make `common` and `q3_signal_quality` importable however this file is invoked
# -- `python q3_signal_quality/run_q3.py`, `python -m q3_signal_quality.run_q3`,
# or from another directory. A reader cloning this repo should not have to know
# or care about PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.datasets import load_troika, load_welltory, welltory_reference_bpm
from q3_signal_quality.baselines import BASELINES
from q3_signal_quality.quality import assess, heart_rate_from_accepted
from q3_signal_quality.validate import (
    USABLE_TOLERANCE_BPM,
    adaptive_decision_accuracy,
    bootstrap_auc_difference,
    auc_from_scores,
    build_window_records,
    inject_artefacts,
    random_rejection_baseline,
    risk_coverage,
    threshold_transfer_test,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

OUT = Path(__file__).resolve().parent / "outputs"

# One palette for every figure, colour-blind safe (Okabe-Ito). Our method is
# always blue, baselines always grey-to-orange, the null always dashed grey, so
# a reader can compare figures without re-reading each legend.
C_OURS = "#0072B2"
C_BASE = ["#D55E00", "#009E73", "#CC79A7"]
C_NULL = "#999999"
C_GOOD = "#0072B2"
C_BAD = "#D55E00"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False,
})


# --------------------------------------------------------------------------

def experiment_troika(results: dict) -> tuple[list, dict]:
    """Main experiment: does the quality score predict downstream failure?"""
    print("\n[1/5] TROIKA -- does the quality score predict heart-rate error?")
    recs = load_troika()
    windows, per_record = [], {}

    for rec in recs:
        assessed = assess(rec.ppg[0], rec.fs)
        rows = build_window_records(rec, assessed)
        windows += rows
        per_record[rec.record_id] = {
            "subject": rec.subject,
            "n_beats": int(len(assessed.peaks)),
            "accept_fraction": round(assessed.accept_fraction, 4),
            "threshold": round(float(assessed.threshold), 4),
            "threshold_mode": assessed.threshold_mode,
        }

    quality = np.array([w.quality for w in windows])
    errors = np.array([w.error for w in windows])
    usable = np.array([w.usable for w in windows])
    subjects = np.array([w.subject for w in windows])

    aucs = {"ours": auc_from_scores(quality, usable)}
    for name in BASELINES:
        aucs[name] = auc_from_scores(np.array([w.baseline[name] for w in windows]),
                                     usable)

    print(f"  {len(windows)} windows from {len(np.unique(subjects))} subjects; "
          f"{100 * usable.mean():.0f}% usable at +/-{USABLE_TOLERANCE_BPM:.0f} bpm")
    print("  AUC (score predicts a usable window):")
    for name, value in sorted(aucs.items(), key=lambda kv: -kv[1]):
        print(f"    {name:16s} {value:.3f}")

    # Is the margin over the strongest baseline real, or 11 subjects of luck?
    best_baseline = max((n for n in BASELINES), key=lambda n: aucs[n])
    ci = bootstrap_auc_difference(
        quality, np.array([w.baseline[best_baseline] for w in windows]),
        usable, subjects)
    print(f"  AUC(ours) - AUC({best_baseline}) = {ci['mean']:+.3f} "
          f"[95% CI {ci['lo']:+.3f}, {ci['hi']:+.3f}] (subject bootstrap)")
    results["auc_margin_vs_best_baseline"] = {"baseline": best_baseline, **ci}

    transfer = {"ours_adaptive": adaptive_decision_accuracy(quality >= 0.5, usable,
                                                            subjects)}
    for name in BASELINES:
        transfer[name] = threshold_transfer_test(
            np.array([w.baseline[name] for w in windows]), usable, subjects)

    print("  Threshold transfer (balanced accuracy, leave-one-subject-out):")
    print(f"    {'method':16s} {'in-domain':>10s} {'transferred':>12s} {'drop':>7s}")
    for name, value in transfer.items():
        print(f"    {name:16s} {value['in_domain']:10.3f} "
              f"{value['transferred']:12.3f} {value['drop']:+7.3f}")

    results["troika"] = {
        "n_windows": len(windows),
        "n_subjects": int(len(np.unique(subjects))),
        "usable_rate": round(float(usable.mean()), 4),
        "median_abs_error_bpm": round(float(np.nanmedian(errors)), 3),
        "auc": {k: round(v, 4) for k, v in aucs.items()},
        "threshold_transfer": transfer,
        "per_record": per_record,
    }
    return windows, aucs


def plot_risk_coverage(windows: list) -> None:
    quality = np.array([w.quality for w in windows])
    errors = np.array([w.error for w in windows])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    cov, risk = risk_coverage(quality, errors)
    ax.plot(cov, risk, color=C_OURS, lw=2.2, label="ours (self-supervised)", zorder=3)

    for colour, name in zip(C_BASE, BASELINES):
        scores = np.array([w.baseline[name] for w in windows])
        cov_b, risk_b = risk_coverage(scores, errors)
        ax.plot(cov_b, risk_b, color=colour, lw=1.3, label=name)

    cov_r, risk_r = random_rejection_baseline(errors)
    ax.plot(cov_r, risk_r, color=C_NULL, lw=1.3, ls="--", label="random rejection")

    ax.set_xlabel("coverage (fraction of windows retained)")
    ax.set_ylabel("mean |HR error| on retained windows (bpm)")
    ax.set_title("Rejecting low-quality windows should lower the error\n"
                 "TROIKA, 1726 windows, 11 subjects", loc="left", fontsize=9)
    ax.invert_xaxis()
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "risk_coverage.png")
    plt.close(fig)


def plot_example(results: dict) -> None:
    """Show the detector's decision on one recording, and what it keeps."""
    rec = load_troika()[0]
    assessed = assess(rec.ppg[0], rec.fs)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))

    ax = axes[0]
    good = assessed.scores >= assessed.threshold
    bins = np.linspace(0, 1, 40)
    ax.hist(assessed.scores[good], bins=bins, color=C_GOOD, alpha=0.85, label="accepted")
    ax.hist(assessed.scores[~good], bins=bins, color=C_BAD, alpha=0.85, label="rejected")
    ax.axvline(assessed.threshold, color="k", lw=1.2, ls="--")
    ax.set_xlabel("beat quality score")
    ax.set_ylabel("beats")
    ax.set_title(f"Threshold found without labels\n(mode: {assessed.threshold_mode})",
                 loc="left", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    for beat in assessed.templates[good][:200]:
        ax.plot(beat, color=C_GOOD, alpha=0.05, lw=0.8)
    ax.plot(assessed.mean_template, color="k", lw=2, label="mean template")
    ax.set_xlabel("sample within beat")
    ax.set_title("Local templates agree with each other", loc="left", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[2]
    fs = rec.fs
    start = int(60 * fs)
    stop = start + int(12 * fs)
    t = np.arange(start, stop) / fs
    ax.plot(t, assessed.filtered[start:stop], color="0.35", lw=0.8)
    in_view = (assessed.peaks >= start) & (assessed.peaks < stop)
    for peak, ok in zip(assessed.peaks[in_view], assessed.accepted[in_view]):
        ax.plot(peak / fs, assessed.filtered[peak], "o", ms=4,
                color=C_GOOD if ok else C_BAD)
    ax.set_xlabel("time (s)")
    ax.set_title("Per-beat decisions on the trace", loc="left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "detector_example.png")
    plt.close(fig)


def experiment_ablation(results: dict) -> None:
    """Which parts of the method are doing the work?"""
    print("\n[2/5] Ablation -- which components and which fusion rule matter?")
    recs = load_troika()

    per_mode = {}
    for mode in ("min", "product", "geometric", "mean"):
        windows = []
        for rec in recs:
            windows += build_window_records(rec, assess(rec.ppg[0], rec.fs,
                                                        combine=mode))
        quality = np.array([w.quality for w in windows])
        usable = np.array([w.usable for w in windows])
        per_mode[mode] = round(auc_from_scores(quality, usable), 4)
        print(f"    combine={mode:10s} AUC={per_mode[mode]:.3f}")

    # Single-component detectors: rebuild the window score from one term only.
    per_component = {}
    for component in ("morphology", "rhythm", "amplitude", "subharmonic"):
        windows_auc = []
        scores_all, usable_all = [], []
        for rec in recs:
            assessed = assess(rec.ppg[0], rec.fs)
            assessed.accepted = (assessed.components[component]
                                 >= np.median(assessed.components[component]))
            rows = build_window_records(rec, assessed)
            scores_all += [w.quality for w in rows]
            usable_all += [w.usable for w in rows]
        per_component[component] = round(
            auc_from_scores(np.array(scores_all), np.array(usable_all)), 4)
        print(f"    only {component:12s} AUC={per_component[component]:.3f}")

    results["ablation"] = {"combine_mode": per_mode, "single_component": per_component}


def experiment_causal(results: dict) -> None:
    """Does the result survive the filter a device could actually run?"""
    print("\n[3/5] Causal filter -- does zero-phase filtering flatter the result?")
    recs = load_troika()
    out = {}
    for label, causal in (("zero_phase", False), ("causal", True)):
        windows = []
        for rec in recs:
            windows += build_window_records(
                rec, assess(rec.ppg[0], rec.fs, causal_filter=causal))
        quality = np.array([w.quality for w in windows])
        usable = np.array([w.usable for w in windows])
        out[label] = round(auc_from_scores(quality, usable), 4)
        print(f"    {label:12s} AUC={out[label]:.3f}")
    results["causal_filter"] = out


def experiment_welltory(results: dict) -> None:
    """Cross-sensor check: a different modality, different site, different cohort."""
    print("\n[4/5] Welltory -- does anything transfer to a different sensor?")
    rows = []
    for rec in load_welltory():
        assessed = assess(rec.ppg[0], rec.fs)
        n = rec.ppg.shape[-1]
        rows.append({
            "record": rec.record_id,
            "channel": rec.channel,
            "n_beats": int(len(assessed.peaks)),
            "accept_fraction": round(assessed.accept_fraction, 3),
            "ref_bpm": round(welltory_reference_bpm(rec), 2),
            "gated_bpm": round(heart_rate_from_accepted(assessed, 0, n, True), 2),
            "ungated_bpm": round(heart_rate_from_accepted(assessed, 0, n, False), 2),
        })

    gated = np.array([abs(r["gated_bpm"] - r["ref_bpm"]) for r in rows])
    ungated = np.array([abs(r["ungated_bpm"] - r["ref_bpm"]) for r in rows])
    channels = {}
    for r in rows:
        channels[r["channel"]] = channels.get(r["channel"], 0) + 1

    print(f"    {len(rows)} recordings; channel chosen automatically: {channels}")
    print(f"    MAE vs Polar H10 -- gated {np.nanmean(gated):.2f} bpm, "
          f"ungated {np.nanmean(ungated):.2f} bpm")
    print(f"    mean accepted fraction {np.mean([r['accept_fraction'] for r in rows]):.1%}")

    results["welltory"] = {
        "n_recordings": len(rows),
        "channels_selected": channels,
        "mae_gated_bpm": round(float(np.nanmean(gated)), 3),
        "mae_ungated_bpm": round(float(np.nanmean(ungated)), 3),
        "per_record": rows,
    }


def experiment_synthetic(results: dict) -> None:
    """Controlled corruption: we know exactly which samples we ruined."""
    print("\n[5/5] Injected artefacts -- does it find the damage we caused?")
    rng = np.random.default_rng(0)
    recs = load_welltory()

    per_kind_scores: list[float] = []
    per_kind_labels: list[bool] = []
    for rec in recs:
        clean = rec.ppg[0]
        corrupted, is_corrupt = inject_artefacts(clean, rec.fs, rng)
        assessed = assess(corrupted, rec.fs)
        if len(assessed.peaks) == 0:
            continue
        # Label each detected beat by whether its peak landed in a ruined stretch.
        labels = is_corrupt[assessed.peaks]
        per_kind_scores += list(assessed.scores)
        per_kind_labels += list(labels)

    scores = np.asarray(per_kind_scores)
    corrupt = np.asarray(per_kind_labels, dtype=bool)
    # AUC for the score separating clean beats from beats inside injected damage.
    auc = auc_from_scores(scores, ~corrupt)
    print(f"    {len(scores)} beats, {100 * corrupt.mean():.0f}% inside injected damage")
    print(f"    AUC (score separates clean beats from corrupted) = {auc:.3f}")

    results["synthetic"] = {
        "n_beats": int(len(scores)),
        "corrupt_fraction": round(float(corrupt.mean()), 4),
        "auc": round(float(auc), 4),
    }

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bins = np.linspace(0, 1, 40)
    ax.hist(scores[~corrupt], bins=bins, color=C_GOOD, alpha=0.8, density=True,
            label="beats in clean stretches")
    ax.hist(scores[corrupt], bins=bins, color=C_BAD, alpha=0.8, density=True,
            label="beats inside injected artefacts")
    ax.set_xlabel("beat quality score")
    ax.set_ylabel("density")
    ax.set_title(f"Controlled corruption, AUC = {auc:.3f}", loc="left", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_corruption.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="skip the ablation and causal-filter sweeps")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: dict = {"tolerance_bpm": USABLE_TOLERANCE_BPM}

    windows, _ = experiment_troika(results)
    plot_risk_coverage(windows)
    plot_example(results)

    if not args.quick:
        experiment_ablation(results)
        experiment_causal(results)

    experiment_welltory(results)
    experiment_synthetic(results)

    results["runtime_seconds"] = round(time.time() - started, 1)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT / 'metrics.json'} and 3 figures "
          f"({results['runtime_seconds']}s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
