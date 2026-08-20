"""
Signal-processing primitives shared by Q3 and Q4.

Everything here is deliberately plain NumPy/SciPy. Two reasons: the operations
need to be countable when we later argue about microcontroller cost (Q3), and
a reader should be able to check each step against the physiology rather than
against a library's defaults.

A note on causality. The filters below use `filtfilt`, which is zero-phase and
therefore non-causal: it looks into the future of the signal. That is correct
for the offline analysis in this repository and wrong for a device. Where it
matters we say so explicitly, and `bandpass_causal` provides the single-pass
biquad form an MCU would actually run.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

# Physiological bounds used throughout. A wearable that must work during
# exercise has to admit tachycardia; one worn overnight has to admit
# bradycardia. 30-220 bpm covers both with margin, and keeps the search space
# for period estimation small enough to be cheap.
HR_MIN_BPM = 30.0
HR_MAX_BPM = 220.0


def bandpass(x: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0,
             order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    The band is chosen from the signal, not by tuning:

    * 0.5 Hz lower edge sits below 30 bpm (0.5 Hz), so the slowest heart rate we
      claim to support survives, while baseline wander from breathing
      (~0.1-0.4 Hz) and from vasomotion and postural drift (below that) is
      attenuated. Baseline wander is the artefact most likely to dominate a
      raw PPG trace, and it carries no beat information.
    * 8 Hz upper edge keeps the fundamental (up to 3.7 Hz at 220 bpm) plus the
      first one or two harmonics. Those harmonics are what make the upstroke
      steep and the dicrotic notch visible, so cutting much lower would blunt
      exactly the morphology Q3 scores. Above 8 Hz a fingertip or wrist PPG
      carries essentially sensor and quantisation noise.

    `order=4` per direction, doubled by filtfilt, gives a sharp enough skirt
    without the numerical fragility of high-order IIR designs.
    """
    nyq = 0.5 * fs
    high = min(high, 0.95 * nyq)  # guard for the ~30 Hz smartphone recordings
    if not 0 < low < high < nyq:
        raise ValueError(f"invalid band {low}-{high} Hz for fs={fs} Hz")
    b, a = sps.butter(order, [low / nyq, high / nyq], btype="band")
    # padlen guards against filtfilt raising on very short segments.
    padlen = min(3 * max(len(a), len(b)), max(len(x) - 1, 0))
    return sps.filtfilt(b, a, x, padlen=padlen)


def bandpass_causal(x: np.ndarray, fs: float, low: float = 0.5, high: float = 8.0,
                    order: int = 2) -> np.ndarray:
    """Single-pass (causal) band-pass: the form a microcontroller would run.

    Used only to check that the Q3 scores do not depend on the zero-phase
    filtering that a real device cannot perform. Phase distortion shifts peak
    locations slightly, which is why we do not use it for the offline results,
    but it must not change which segments are judged clean.
    """
    nyq = 0.5 * fs
    high = min(high, 0.95 * nyq)
    sos = sps.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sps.sosfilt(sos, x)


def dominant_period(x: np.ndarray, fs: float,
                    hr_min: float = HR_MIN_BPM,
                    hr_max: float = HR_MAX_BPM,
                    prominence: float = 0.03,
                    subharmonic_ratio: float = 0.7) -> tuple[float, float]:
    """Estimate the dominant cardiac period by autocorrelation.

    Returns (period_samples, normalised_autocorrelation_at_that_lag).

    Why autocorrelation rather than an FFT peak: we want the *period* of a
    repeating shape, not the strongest spectral line. A PPG pulse is sharply
    asymmetric, so its energy is spread across a fundamental and several
    harmonics; when the fundamental is attenuated -- a common effect of poor
    perfusion or a loose sensor -- the largest FFT bin can land on the second
    harmonic and halve the reported rate. Autocorrelation asks the question we
    actually care about: at what lag does this waveform repeat? The second
    return value quantifies how strongly it repeats at all, which is itself a
    quality signal, and Q3 uses it.

    Why the *first* strong peak rather than the global maximum
    ----------------------------------------------------------
    This matters more than any other choice in the file, and an earlier version
    got it wrong. Taking `argmax` over the physiological lag range fails in a
    specific and common way: a PPG carries strong low-frequency content --
    respiration, vasomotion, postural drift -- and during exercise respiration
    alone reaches 30-50 breaths per minute, squarely inside the heart-rate
    search range. The autocorrelation of that slow modulation can easily exceed
    the cardiac peak, so `argmax` returns the respiratory period and the
    estimated heart rate comes out around 40 bpm while the true rate is 120.
    Measured on TROIKA, `argmax` reported 38-49 bpm for windows whose reference
    rate was 110-155 bpm.

    The fundamental period is the *first* lag at which the waveform repeats;
    every later peak is a multiple of it. Taking the earliest prominent peak
    therefore recovers the fundamental and is immune to a stronger, slower
    component sitting further out. This is the standard fix in pitch detection,
    where the same failure is known as pitch-halving.

    The `subharmonic_ratio` guard stops the opposite error. A dicrotic notch
    can raise a small early bump in the autocorrelation, and locking onto it
    would double the reported rate. So we take the earliest prominent peak that
    reaches at least this fraction of the strongest prominent peak: earliest,
    but not negligible.

    Computed through the FFT, so cost is O(N log N) rather than O(N^2).
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 8 or not np.any(x):
        return float("nan"), 0.0

    # Zero-pad to at least 2N to get the linear (not circular) autocorrelation.
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spectrum = np.fft.rfft(x, nfft)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[:n]
    if acf[0] <= 0:
        return float("nan"), 0.0
    acf = acf / acf[0]  # normalise so acf[0] == 1

    lag_min = int(np.floor(fs * 60.0 / hr_max))
    lag_max = min(int(np.ceil(fs * 60.0 / hr_min)), n - 1)
    if lag_max <= lag_min:
        return float("nan"), 0.0

    window = acf[lag_min:lag_max + 1]
    peaks, _ = sps.find_peaks(window, prominence=prominence)

    if peaks.size == 0:
        # No lag in the physiological range at which this waveform repeats.
        # Reporting the argmax anyway would manufacture a heart rate out of
        # noise, so we decline; callers treat NaN as "no cardiac content here".
        return float("nan"), 0.0

    heights = window[peaks]
    strong = peaks[heights >= subharmonic_ratio * heights.max()]
    best = int(strong[0] if strong.size else peaks[int(np.argmax(heights))])
    lag = best + lag_min
    return float(lag), float(acf[lag])


def detect_beats(x: np.ndarray, fs: float, period: float | None = None) -> np.ndarray:
    """Locate systolic peaks. Returns sample indices.

    The only tuned quantity is the refractory distance, and it is derived from
    the signal rather than fixed: peaks must be at least 0.6 of the dominant
    period apart. 0.6 rather than 1.0 because heart rate varies within a
    recording and we would rather admit a slightly early beat -- the rhythm
    term in Q3 can reject it later -- than silently drop a real one. Missing a
    beat and inventing a beat are not symmetric errors here: a missed beat
    corrupts two inter-beat intervals, an extra beat corrupts two as well but
    is far easier to spot from morphology.

    Prominence is set from the signal's own robust scale (the median absolute
    deviation) so that the detector does not need to know the sensor's gain,
    the subject's perfusion, or the units of the input.
    """
    x = np.asarray(x, dtype=float)
    if period is None or not np.isfinite(period):
        period, _ = dominant_period(x, fs)
    if not np.isfinite(period) or period <= 0:
        return np.array([], dtype=int)

    scale = 1.4826 * np.median(np.abs(x - np.median(x)))  # MAD -> sigma-equivalent
    if scale <= 0:
        return np.array([], dtype=int)

    peaks, _ = sps.find_peaks(
        x,
        distance=max(int(0.6 * period), 1),
        prominence=0.3 * scale,
    )
    return peaks


def extract_beats(x: np.ndarray, peaks: np.ndarray, period: float,
                  length: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fixed-length, amplitude-normalised waveform around each peak.

    Returns (beats, kept_peaks) where `beats` has shape (n_beats, length).

    The window runs from 0.35 periods before the peak to 0.65 after. That
    split is not arbitrary: the systolic upstroke occupies roughly the first
    third of a cardiac cycle and the diastolic decay with its dicrotic notch
    the remaining two thirds, so this captures one full pulse aligned on its
    most reliable landmark. Aligning on the peak rather than on the foot
    matters because the foot is the harder point to find precisely, and any
    jitter in alignment shows up directly as a loss of template correlation --
    i.e. it would masquerade as poor signal quality.

    Each beat is z-scored individually. This is what makes the method blind to
    absolute amplitude, and therefore to skin tone, sensor gain and contact
    pressure -- the things the problem statement says break hand-tuned
    thresholds. Amplitude is not discarded; it is scored separately, on its own
    terms, in the Q3 amplitude term.
    """
    before = int(round(0.35 * period))
    after = int(round(0.65 * period))
    beats, kept = [], []

    for p in peaks:
        start, stop = p - before, p + after
        if start < 0 or stop >= len(x):
            continue  # partial beats at the edges are dropped, not padded
        segment = x[start:stop]
        if len(segment) < 4:
            continue
        # Resample to a common length so beats from different heart rates are
        # directly comparable; morphology, not duration, is what we compare.
        resampled = np.interp(
            np.linspace(0, len(segment) - 1, length),
            np.arange(len(segment)),
            segment,
        )
        sd = resampled.std()
        if sd <= 0:
            continue
        beats.append((resampled - resampled.mean()) / sd)
        kept.append(p)

    if not beats:
        return np.empty((0, length)), np.array([], dtype=int)
    return np.asarray(beats), np.asarray(kept, dtype=int)


def robust_scale(x: np.ndarray) -> float:
    """Median absolute deviation, scaled to be comparable to a standard deviation.

    Used instead of `std` wherever a scale estimate has to survive the very
    artefacts we are trying to detect. A single motion spike can inflate a
    standard deviation by an order of magnitude; the MAD has a breakdown point
    of 50%, so it stays meaningful until half the samples are corrupt.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def resample_to(x: np.ndarray, n_out: int) -> np.ndarray:
    """Linear resampling to a fixed number of points along the last axis."""
    x = np.asarray(x, dtype=float)
    src = np.arange(x.shape[-1])
    dst = np.linspace(0, x.shape[-1] - 1, n_out)
    if x.ndim == 1:
        return np.interp(dst, src, x)
    return np.stack([np.interp(dst, src, row) for row in x])
