"""
Does the Q3 quality detector predict which subjects the Q4 model fails on?

    python cross_link.py

The two answers in this repository were built independently: Q3's detector knows
nothing about the regressor, and Q4's regressor is never gated by the quality
score. If both are measuring something real, the subjects Q3 judges to have
unusable signal should be the subjects Q4 gets wrong. If neither is, the
correlation will be noise.

This reads Q4's committed per-fold results rather than retraining, so it needs
`q4_edge_ml/run_q4.py` to have been run first (its outputs are committed, so a
fresh clone can run this immediately).

Spearman rather than Pearson is the headline statistic, and both are reported.
The relationship is monotonic but not linear -- one subject is an extreme
outlier in MAE and would dominate a Pearson correlation -- so a rank statistic
is the honest choice. With eleven subjects neither has much power, which is why
the p-values are reported rather than just the coefficients.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import load_troika
from q3_signal_quality.quality import assess
from q3_signal_quality.validate import build_window_records

warnings.filterwarnings("ignore", category=RuntimeWarning)

Q4_METRICS = Path(__file__).resolve().parent / "q4_edge_ml" / "outputs" / "metrics.json"


def main() -> int:
    if not Q4_METRICS.exists():
        print(f"missing {Q4_METRICS}; run q4_edge_ml/run_q4.py first")
        return 1

    q4 = json.loads(Q4_METRICS.read_text())
    mae_by_subject = {row["subject"]: row["float32_mae"]
                      for row in q4["subject_disjoint"]["per_fold"]}

    print("Assessing signal quality per subject (Q3 detector, no knowledge of Q4)...")
    quality_by_subject: dict[str, list] = {}
    for rec in load_troika():
        rows = build_window_records(rec, assess(rec.ppg[0], rec.fs))
        quality_by_subject.setdefault(rec.subject, []).extend(rows)

    subjects = sorted(set(mae_by_subject) & set(quality_by_subject))
    quality, usable, mae = [], [], []

    print(f"\n{'subject':>8s} {'Q3 mean quality':>16s} {'Q3 usable rate':>15s} "
          f"{'Q4 MAE (bpm)':>13s}")
    for subject in subjects:
        rows = quality_by_subject[subject]
        q = float(np.mean([r.quality for r in rows]))
        u = float(np.mean([r.usable for r in rows]))
        m = mae_by_subject[subject]
        quality.append(q)
        usable.append(u)
        mae.append(m)
        print(f"{subject:>8s} {q:16.3f} {u:15.2f} {m:13.2f}")

    quality, usable, mae = np.array(quality), np.array(usable), np.array(mae)

    results = {}
    for name, scores in (("mean_quality", quality), ("usable_rate", usable)):
        rho, p_rho = stats.spearmanr(scores, mae)
        r, p_r = stats.pearsonr(scores, mae)
        results[name] = {"spearman": round(float(rho), 4),
                         "spearman_p": round(float(p_rho), 5),
                         "pearson": round(float(r), 4),
                         "pearson_p": round(float(p_r), 5)}
        print(f"\n{name} vs Q4 MAE  (n = {len(mae)} subjects)")
        print(f"  Spearman rho = {rho:+.3f}  p = {p_rho:.4f}")
        print(f"  Pearson    r = {r:+.3f}  p = {p_r:.4f}")

    worst = subjects[int(np.argmax(mae))]
    print(f"\nWorst Q4 subject: {worst} (MAE {mae.max():.2f} bpm), "
          f"Q3 usable rate {usable[int(np.argmax(mae))]:.0%} "
          f"-- lowest of any subject."
          if usable[int(np.argmax(mae))] == usable.min() else "")

    out = Path(__file__).resolve().parent / "q4_edge_ml" / "outputs" / "cross_link.json"
    out.write_text(json.dumps({
        "subjects": subjects,
        "q3_mean_quality": quality.tolist(),
        "q3_usable_rate": usable.tolist(),
        "q4_float32_mae": mae.tolist(),
        "correlations": results,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
