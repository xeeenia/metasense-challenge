"""
The hand-tuned signal-quality heuristics that Q3's problem statement calls out.

These exist so the proposed method is measured against something, rather than
merely asserted to be better. Both are real published indices in routine use,
and both are exactly the sort of "threshold on autocorrelation" the brief says
works poorly across skin tones, motion and sensor placements. Reproducing that
failure -- or failing to reproduce it -- is more informative than any claim we
could make about our own method in isolation.

They are evaluated per window, since that is how they are normally applied.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from common.dsp import bandpass, dominant_period


def skewness_sqi(window: np.ndarray, fs: float, prefilter: bool = True) -> float:
    """Elgendi's skewness signal-quality index.

    The rationale is sound and worth stating, because it is the strongest of
    the classical indices: a clean PPG pulse is markedly asymmetric -- a fast
    systolic upstroke followed by a slow diastolic decay -- so the amplitude
    distribution of a clean window is positively skewed. Gaussian-ish noise is
    symmetric and scores near zero.

    Its weakness is equally structural, and is what we expect to see in the
    results: skewness is a property of the amplitude *histogram* and is blind
    to time ordering. Shuffle a clean window's samples and the skewness is
    unchanged, though every trace of a heartbeat is gone. Anything with
    occasional large positive excursions -- a motion spike, a saturating
    sensor -- scores well.

    Reference: Elgendi, "Optimal signal quality index for photoplethysmogram
    signals", Bioengineering 3(4), 2016.
    """
    x = bandpass(window, fs) if prefilter else np.asarray(window, dtype=float)
    if x.std() <= 0:
        return 0.0
    return float(stats.skew(x))


def autocorrelation_sqi(window: np.ndarray, fs: float,
                        prefilter: bool = True) -> float:
    """Periodicity index: the autocorrelation height at the dominant lag.

    This is the index the problem statement names directly. It asks a genuinely
    relevant question -- does this window repeat at a plausible heart rate? --
    and in practice it is applied with a fixed threshold, typically somewhere
    between 0.4 and 0.7, chosen on whatever development set was to hand.

    The fixed threshold is the problem, not the statistic. Absolute
    autocorrelation height depends on how much of the window's variance is
    pulsatile, which depends on perfusion, on melanin absorption at the LED
    wavelength, on contact pressure and on sensor site. A threshold calibrated
    on one cohort systematically rejects another. Our method uses the same
    statistic as one input but never compares it to a fixed number.
    """
    x = bandpass(window, fs) if prefilter else np.asarray(window, dtype=float)
    _, strength = dominant_period(x, fs)
    return float(strength)


def zero_crossing_sqi(window: np.ndarray, fs: float,
                      prefilter: bool = True) -> float:
    """Negated zero-crossing rate: high-frequency noise detector.

    A clean band-passed PPG crosses zero about twice per beat. Noise crosses
    far more often, so the negated rate behaves like a quality score. Included
    as a third reference point because it fails in a different direction from
    the other two -- it is fooled by smooth, low-frequency motion artefacts,
    which cross zero *less* often than a real pulse and therefore score
    excellently.
    """
    x = bandpass(window, fs) if prefilter else np.asarray(window, dtype=float)
    if x.std() <= 0:
        return -np.inf
    crossings = np.count_nonzero(np.diff(np.signbit(x)))
    return -float(crossings) / (len(x) / fs)


BASELINES = {
    "skewness": skewness_sqi,
    "autocorrelation": autocorrelation_sqi,
    "zero_crossing": zero_crossing_sqi,
}
