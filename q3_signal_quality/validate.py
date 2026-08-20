"""
Validation for Q3 -- built to catch the method fooling itself.

The brief asks for a validation plan that "could actually catch the method
fooling itself". An unsupervised detector has two easy ways to look good
without being good, and each check below exists to close one of them.

Failure 1: sorting instead of detecting.
    A method that always rejects the worst-scoring third of any recording will
    show a lower error on what it keeps, on *any* data, including data that is
    uniformly clean or uniformly ruined. The risk-coverage analysis closes this
    by comparing against a random rejector operating at the identical coverage.
    If quality-ordered rejection is no better than coin-flip rejection, the
    score carries no information, however good the retained subset looks.

Failure 2: circular labels.
    We have no expert quality annotations, so the temptation is to define
    "clean" using the same statistics the detector uses, and then discover that
    the detector agrees with itself. We avoid this by taking ground truth from
    a different instrument entirely: TROIKA's simultaneously recorded ECG. The
    reference heart rate is electrical, the detector sees only the optical
    trace, and the two share no processing. A window is "usable" if the
    PPG-derived estimate lands within tolerance of the ECG -- a fact the
    detector cannot observe.

That second definition deserves one more sentence, because it looks circular at
first glance and is not. We label a window by whether the *downstream estimate*
was correct, then ask whether the *quality score* predicted that. The score
never sees the reference, so this measures precisely the property a clinical
device needs: can the device know, at the time of measurement and without a
gold standard, that it is about to report something wrong?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.datasets import Recording, TROIKA_HOP_S, TROIKA_WINDOW_S
from q3_signal_quality.baselines import BASELINES
from q3_signal_quality.quality import (
    QualityResult,
    heart_rate_from_accepted,
    window_accept_fraction,
)

# A window counts as "usable" if the PPG-only heart rate is within this many
# beats per minute of the ECG reference. 5 bpm is the tolerance the IEEE and
# ANSI/AAMI wearable heart-rate literature conventionally treats as clinically
# acceptable, and it is not tuned here -- the sensitivity of every conclusion
# to this choice is reported in the outputs rather than hidden.
USABLE_TOLERANCE_BPM = 5.0


@dataclass
class WindowRecord:
    """One analysis window, with everything needed to score it after the fact."""

    subject: str
    record_id: str
    index: int
    ref_bpm: float
    est_bpm_all: float        # PPG estimate using every detected beat
    quality: float            # our score: fraction of beats accepted
    baseline: dict[str, float]

    @property
    def error(self) -> float:
        return abs(self.est_bpm_all - self.ref_bpm)

    @property
    def usable(self) -> bool:
        return bool(np.isfinite(self.error) and self.error <= USABLE_TOLERANCE_BPM)


def build_window_records(rec: Recording, result: QualityResult,
                         channel: int = 0) -> list[WindowRecord]:
    """Score every reference window of a TROIKA recording.

    Note which heart rate goes into `est_bpm_all`: the one computed from *all*
    detected beats, with the quality decision ignored. That is deliberate. We
    want to measure whether the quality score predicts the error of an ungated
    estimator, which is the honest question -- if we gated the estimate first
    and then correlated the score with the gated error, the score would be
    predicting a quantity it had already altered.
    """
    fs = rec.fs
    win = int(round(TROIKA_WINDOW_S * fs))
    hop = int(round(TROIKA_HOP_S * fs))
    raw = rec.ppg[channel]

    out: list[WindowRecord] = []
    for i, ref in enumerate(rec.ref_bpm):
        start, stop = i * hop, i * hop + win
        if stop > raw.shape[-1]:
            break
        segment = raw[start:stop]
        out.append(
            WindowRecord(
                subject=rec.subject,
                record_id=rec.record_id,
                index=i,
                ref_bpm=float(ref),
                est_bpm_all=heart_rate_from_accepted(result, start, stop,
                                                     use_quality=False),
                quality=window_accept_fraction(result, start, stop),
                baseline={name: fn(segment, fs) for name, fn in BASELINES.items()},
            )
        )
    return out


# --------------------------------------------------------------------------
# Risk-coverage analysis
# --------------------------------------------------------------------------

def risk_coverage(scores: np.ndarray, errors: np.ndarray,
                  n_points: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Mean error among the highest-scoring fraction of windows, versus coverage.

    This is the selective-prediction view of signal quality, and it is the
    right one: a quality gate is not a classifier to be scored in isolation, it
    is a mechanism for trading how often the device answers against how wrong
    it is when it does. A useful score makes this curve fall monotonically as
    coverage shrinks. A useless score leaves it flat.

    Windows where the estimator returned no value at all are treated as
    maximally wrong rather than dropped. Dropping them would let a method that
    silently fails to produce an estimate look accurate by abstaining, which is
    the same trick from a different direction.
    """
    finite = np.isfinite(scores)
    scores, errors = scores[finite], errors[finite]
    if scores.size == 0:
        return np.array([]), np.array([])

    errors = np.where(np.isfinite(errors), errors, np.nanmax(errors[np.isfinite(errors)])
                      if np.isfinite(errors).any() else 0.0)

    order = np.argsort(-scores)          # best first
    sorted_errors = errors[order]
    coverages = np.linspace(1.0, 0.1, n_points)

    risks = []
    for coverage in coverages:
        keep = max(int(round(coverage * len(sorted_errors))), 1)
        risks.append(float(np.mean(sorted_errors[:keep])))
    return coverages, np.asarray(risks)


def random_rejection_baseline(errors: np.ndarray, n_points: int = 40,
                              n_repeats: int = 200,
                              seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage for rejecting windows at random -- the null hypothesis.

    Any score must beat this to have demonstrated anything. Rejecting at random
    leaves the expected error unchanged at every coverage, so this curve is
    flat; the gap between it and a method's curve is the method's entire value.
    """
    rng = np.random.default_rng(seed)
    errors = errors[np.isfinite(errors)]
    coverages = np.linspace(1.0, 0.1, n_points)

    risks = []
    for coverage in coverages:
        keep = max(int(round(coverage * len(errors))), 1)
        draws = [np.mean(rng.choice(errors, keep, replace=False))
                 for _ in range(n_repeats)]
        risks.append(float(np.mean(draws)))
    return coverages, np.asarray(risks)


def auc_from_scores(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve, computed as the Mann-Whitney statistic.

    Written out rather than imported so ties are handled explicitly: with a
    coarse score such as "fraction of beats accepted", ties are common, and a
    tie should count as half a correct ordering, not a whole one.
    """
    finite = np.isfinite(scores)
    scores, positive = scores[finite], positive[finite]
    pos, neg = scores[positive], scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)

    # Average ranks within tied groups.
    values = np.concatenate([pos, neg])[order]
    i = 0
    sorted_ranks = ranks[order]
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        if j > i:
            sorted_ranks[i:j + 1] = sorted_ranks[i:j + 1].mean()
        i = j + 1
    ranks[order] = sorted_ranks

    rank_sum_pos = ranks[:pos.size].sum()
    return float((rank_sum_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def bootstrap_auc_difference(score_a: np.ndarray, score_b: np.ndarray,
                             positive: np.ndarray, groups: np.ndarray,
                             n_boot: int = 2000,
                             seed: int = 0) -> dict[str, float]:
    """Confidence interval on AUC(a) - AUC(b), resampling *subjects*, not windows.

    Resampling windows would be wrong and would make every difference look
    significant. Windows overlap by 75% and come in runs of a few hundred from
    the same person, so they are nowhere near independent; a window bootstrap
    treats 1726 correlated observations as 1726 independent ones and shrinks
    the interval by roughly the square root of that correlation. Resampling
    whole subjects respects the level at which the data actually varies, and is
    the level at which we want to generalise anyway -- the question is whether
    the method would still win on the next eleven people, not on the next
    window of these eleven.

    With 11 subjects the interval will be wide. That is the honest width.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    diffs = []

    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        mask = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        pos = positive[mask]
        if len(np.unique(pos)) < 2:
            continue
        diffs.append(auc_from_scores(score_a[mask], pos)
                     - auc_from_scores(score_b[mask], pos))

    if not diffs:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    diffs = np.asarray(diffs)
    return {
        "mean": float(np.mean(diffs)),
        "lo": float(np.percentile(diffs, 2.5)),
        "hi": float(np.percentile(diffs, 97.5)),
    }


# --------------------------------------------------------------------------
# Threshold transfer: the experiment that isolates the actual claim
# --------------------------------------------------------------------------

def balanced_accuracy(decisions: np.ndarray, positive: np.ndarray) -> float:
    """Mean of sensitivity and specificity.

    Balanced rather than plain accuracy because the usable/unusable split is
    not guaranteed to be even, and a detector that simply accepts everything
    should not be able to score well by exploiting a class imbalance.
    """
    tp = np.count_nonzero(decisions & positive)
    fn = np.count_nonzero(~decisions & positive)
    tn = np.count_nonzero(~decisions & ~positive)
    fp = np.count_nonzero(decisions & ~positive)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return float(np.nanmean([sens, spec]))


def threshold_transfer_test(scores: np.ndarray, positive: np.ndarray,
                            subjects: np.ndarray) -> dict[str, float]:
    """Fit a threshold on all subjects but one, then apply it to that one.

    This is the experiment the whole method is built around, and the only one
    that separates our claim from the baselines' on equal terms.

    An AUC comparison flatters a fixed-threshold index, because AUC is
    threshold-free: it asks whether the statistic *ranks* windows correctly
    within the pool it is given, and quietly grants the index an oracle
    threshold chosen with knowledge of the answers. That is not how a deployed
    device works. A deployed device carries a number that was chosen on
    somebody else's data, and the problem statement's complaint is precisely
    that this number does not travel -- across skin tones, sensor placements
    and motion conditions.

    So: pick the threshold that maximises balanced accuracy on the other
    subjects, apply it unchanged to the held-out subject, and report the drop.
    A method with no transferable threshold cannot lose anything here, which is
    the point. Its per-subject boundary is re-derived from the subject's own
    score distribution and so is reported unchanged in both columns.
    """
    finite = np.isfinite(scores)
    scores, positive, subjects = scores[finite], positive[finite], subjects[finite]
    unique = np.unique(subjects)

    in_domain, transferred = [], []
    for held_out in unique:
        test = subjects == held_out
        train = ~test
        if not train.any() or not test.any():
            continue
        if len(np.unique(positive[train])) < 2:
            continue

        candidates = np.unique(scores[train])
        if candidates.size > 200:  # keep the sweep cheap without changing it
            candidates = np.quantile(scores[train], np.linspace(0, 1, 200))
        best = max(candidates,
                   key=lambda t: balanced_accuracy(scores[train] >= t, positive[train]))

        in_domain.append(balanced_accuracy(scores[train] >= best, positive[train]))
        if len(np.unique(positive[test])) >= 2:
            transferred.append(balanced_accuracy(scores[test] >= best, positive[test]))

    return {
        "in_domain": float(np.mean(in_domain)) if in_domain else float("nan"),
        "transferred": float(np.mean(transferred)) if transferred else float("nan"),
        "drop": float(np.mean(in_domain) - np.mean(transferred))
        if in_domain and transferred else float("nan"),
        "n_folds": len(transferred),
    }


def adaptive_decision_accuracy(decisions: np.ndarray, positive: np.ndarray,
                               subjects: np.ndarray) -> dict[str, float]:
    """Balanced accuracy of a method that sets its own boundary per recording.

    Reported in both columns of the transfer table because there is nothing to
    transfer: the same procedure runs on every subject, having seen no other
    subject's data at any point.
    """
    finite = np.isfinite(decisions.astype(float))
    decisions, positive, subjects = decisions[finite], positive[finite], subjects[finite]
    per_subject = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        if len(np.unique(positive[mask])) < 2:
            continue
        per_subject.append(balanced_accuracy(decisions[mask], positive[mask]))
    value = float(np.mean(per_subject)) if per_subject else float("nan")
    return {"in_domain": value, "transferred": value, "drop": 0.0,
            "n_folds": len(per_subject)}


# --------------------------------------------------------------------------
# Controlled corruption: ground truth we construct ourselves
# --------------------------------------------------------------------------

def inject_artefacts(x: np.ndarray, fs: float, rng: np.random.Generator,
                     n_events: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt known stretches of a clean recording. Returns (signal, is_corrupt).

    This complements, and does not replace, the real-data analysis. Its purpose
    is to answer a question real data cannot: given that we know exactly which
    samples are ruined, does the detector find *those* samples? On real data we
    only ever observe a downstream consequence.

    The three artefact types are chosen because they fail in three different
    ways, and a method that catches only one of them would otherwise look
    complete:

      * burst      -- a large transient, the classic motion spike. Should be
                      caught by the amplitude term.
      * flatline   -- sensor lift-off. Amplitude collapses; morphology becomes
                      meaningless. Should be caught by amplitude and morphology.
      * baseline   -- a slow, large excursion, e.g. from limb movement or a
                      pressure change. This is the interesting one: it survives
                      the band-pass partially and does *not* look like noise.

    We do not inject a periodic artefact at a plausible heart rate, because we
    already know the method cannot catch it and say so in the README. Adding an
    artefact class we would fail and then omitting it from the report is the
    kind of quiet curation the brief warns against; inventing one we happen to
    catch would be worse.
    """
    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    corrupt = np.zeros(n, dtype=bool)
    scale = float(np.std(x))
    if scale <= 0:
        return x, corrupt

    kinds = ["burst", "flatline", "baseline"]
    for k in range(n_events):
        kind = kinds[k % len(kinds)]
        length = int(rng.uniform(1.0, 3.0) * fs)
        if length >= n:
            continue
        start = int(rng.integers(0, n - length))
        stop = start + length
        span = np.arange(length)

        if kind == "burst":
            amp = rng.uniform(5.0, 15.0) * scale
            envelope = np.hanning(length)
            carrier = np.sin(2 * np.pi * rng.uniform(2.0, 8.0) * span / fs)
            x[start:stop] += amp * envelope * carrier
        elif kind == "flatline":
            x[start:stop] = x[start] + rng.normal(0, 0.02 * scale, length)
        else:  # baseline excursion
            amp = rng.uniform(8.0, 20.0) * scale
            x[start:stop] += amp * np.hanning(length)

        corrupt[start:stop] = True

    return x, corrupt
