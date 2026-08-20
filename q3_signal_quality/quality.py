"""
Q3: a self-supervised, zero-shot signal-quality detector for PPG.

The idea in one sentence
------------------------
Over a short stretch of time, real heartbeats resemble their neighbours and
artefacts resemble nothing; so a recording can supply its own training signal,
and no labels or pre-trained weights are needed at deployment.

Why that is the right thing to exploit
--------------------------------------
A photoplethysmogram is the optical shadow of a pressure wave that the same
heart, pushing through the same arteries, produces a few dozen times a minute.
Neighbouring beats are therefore near-copies of one another: their shape drifts
slowly -- over tens of seconds, with respiration and vascular tone -- but never
abruptly. Motion, contact loss and ambient-light leakage have no such
constraint. They are transient, they are not phase-locked to the cardiac cycle,
and crucially they are *not reproducible*: a second motion artefact looks
nothing like the first.

That asymmetry is the whole method. We never need to model what an artefact
looks like, which is fortunate, because the space of artefacts is unbounded and
the space of clean beats is not. We only need to know what this subject's beats
look like *right now*, and the recording tells us that for free.

Why everything here is local
----------------------------
An earlier version of this file estimated one cardiac period and one template
per recording. On the exercise data used here that is simply wrong: heart rate
ranges over more than 90 bpm within a single 5-minute session, so a global
period mis-segments most of the recording and a global template blurs together
waveforms taken at 70 and at 166 bpm. Both quantities are therefore estimated
in a moving neighbourhood -- short enough that heart rate and morphology are
quasi-stationary, long enough to average tens of beats. This also happens to be
what a device would do, since it processes a rolling buffer rather than a file.

Why this is genuinely zero-shot
-------------------------------
The problem statement's complaint about hand-tuned heuristics is that
thresholds calibrated on one population fail on another: darker skin absorbs
more green light and reduces the AC/DC ratio, a wrist sensor sees a different
waveform from a fingertip, a loose strap changes everything. Every quantity in
this method is either dimensionless (a correlation coefficient) or expressed
relative to the recording's own robust statistics (a MAD-scaled deviation).
Nothing is measured in the units of the sensor, so nothing has to be
re-calibrated when the sensor, the site or the subject changes. The template is
re-estimated from scratch every time; there are no learned parameters to carry
over and therefore nothing to transfer badly.

Relationship to self-supervised representation learning
-------------------------------------------------------
This is deliberately the degenerate case of a contrastive method. In SimSiam or
SimCLR terms: the positives are temporally adjacent beats, the representation
is the amplitude-normalised beat itself, and the similarity metric is
correlation. We keep the self-supervision -- the supervisory signal comes from
the data's own structure -- and drop the learned encoder.

That is not laziness. A learned encoder must be pre-trained on some corpus, and
the distribution of that corpus is exactly what the problem statement says goes
wrong at deployment: an encoder trained mostly on light skin and stationary
subjects has silently baked in a prior that a locally re-estimated template
never acquires. The encoder would buy a richer similarity metric; it would cost
the property we are being asked for. Where a learned representation would
genuinely help, and how we would find out, is set out in the README.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture

from common.dsp import (
    bandpass,
    bandpass_causal,
    detect_beats,
    dominant_period,
    extract_beats,
    robust_scale,
)

# The single absolute constant in the method, and the one place where "no
# tuning" is not literally achievable. See `_choose_threshold` for why some
# absolute anchor is unavoidable and why correlation is the least arbitrary
# place to put one. r = 0.5 means a beat shares a quarter of its variance with
# its own neighbours; below that, calling it the same waveform is not
# defensible on any sensor or any skin tone.
MIN_PLAUSIBLE_CORRELATION = 0.5

# Length of the moving neighbourhood used to estimate the local cardiac period,
# in seconds. Long enough to hold ~10-30 beats so the autocorrelation has
# something to average, short enough that heart rate is quasi-stationary across
# it. At 8 s a sprint start would smear; at 30 s the autocorrelation peak would
# be blunted by the rate change itself.
PERIOD_WINDOW_S = 12.0

# Number of neighbouring beats forming each local template (odd, centred).
# 31 beats is 15-30 s of signal depending on rate: enough for a pointwise
# median to be stable, short enough that vascular tone has not drifted.
TEMPLATE_NEIGHBOURS = 31


@dataclass
class QualityResult:
    """Per-beat quality assessment for one recording."""

    peaks: np.ndarray                   # sample index of each detected beat
    scores: np.ndarray                  # combined quality in [0, 1], per beat
    accepted: np.ndarray                # bool, per beat
    components: dict[str, np.ndarray]   # the three sub-scores, for ablation
    templates: np.ndarray               # (n_beats, L) local template per beat
    threshold: float                    # where the decision boundary landed
    periods: np.ndarray                 # local cardiac period at each beat
    acf_strength: np.ndarray            # local periodicity at each beat
    threshold_mode: str                 # which branch of _choose_threshold fired
    fs: float = 0.0
    filtered: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def accept_fraction(self) -> float:
        return float(self.accepted.mean()) if self.accepted.size else 0.0

    @property
    def mean_template(self) -> np.ndarray:
        """Average of the local templates -- for plotting only, never for scoring."""
        return self.templates.mean(axis=0) if self.templates.size else np.array([])


# --------------------------------------------------------------------------
# Local period tracking and beat detection
# --------------------------------------------------------------------------

def _track_period(x: np.ndarray, fs: float,
                  window_s: float = PERIOD_WINDOW_S) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the cardiac period in overlapping windows across the recording.

    Returns (period_per_sample, acf_strength_per_sample), both interpolated
    back onto the sample grid so any point in the signal can ask "what is the
    local heart period here?".

    The autocorrelation strength is carried along because it is a quality
    signal in its own right: a window whose waveform does not repeat at *any*
    lag in the physiological range has no cardiac content to assess. It is used
    later as a diagnostic rather than as a gate, since a low value can also
    mean a genuinely irregular rhythm.
    """
    n = len(x)
    win = int(round(window_s * fs))
    hop = max(win // 4, 1)

    if n < win:
        period, strength = dominant_period(x, fs)
        return (np.full(n, period if np.isfinite(period) else np.nan),
                np.full(n, strength))

    centres, periods, strengths = [], [], []
    for start in range(0, n - win + 1, hop):
        period, strength = dominant_period(x[start:start + win], fs)
        centres.append(start + win / 2)
        periods.append(period)
        strengths.append(strength)

    centres = np.asarray(centres, dtype=float)
    periods = np.asarray(periods, dtype=float)
    strengths = np.asarray(strengths, dtype=float)

    valid = np.isfinite(periods)
    if not valid.any():
        return np.full(n, np.nan), np.zeros(n)

    # A three-window median smooths out the occasional window where the
    # autocorrelation locks onto a harmonic. It cannot repair a systematically
    # wrong estimate, and does not pretend to.
    smoothed = periods.copy()
    for i in range(len(periods)):
        lo, hi = max(0, i - 1), min(len(periods), i + 2)
        chunk = periods[lo:hi][np.isfinite(periods[lo:hi])]
        if chunk.size:
            smoothed[i] = np.median(chunk)

    grid = np.arange(n, dtype=float)
    period_per_sample = np.interp(grid, centres[valid], smoothed[valid])
    strength_per_sample = np.interp(grid, centres[valid], strengths[valid])
    return period_per_sample, strength_per_sample


def _detect_beats_adaptive(x: np.ndarray, fs: float, period_track: np.ndarray,
                           window_s: float = PERIOD_WINDOW_S) -> np.ndarray:
    """Detect beats window by window using the locally valid refractory period.

    `scipy.find_peaks` takes a single scalar minimum distance, which is why
    detection is done per window rather than in one pass: at 70 bpm the correct
    refractory distance is nearly twice what it is at 166 bpm, and using either
    value globally loses beats at one end of the recording.

    Windows overlap, so a beat near a boundary can be found twice; duplicates
    closer together than half a period are merged afterwards.
    """
    n = len(x)
    win = int(round(window_s * fs))
    hop = max(win // 2, 1)

    found: list[int] = []
    starts = range(0, max(n - win + 1, 1), hop)
    for start in starts:
        stop = min(start + win, n)
        segment = x[start:stop]
        local_period = np.nanmedian(period_track[start:stop])
        if not np.isfinite(local_period) or local_period <= 0:
            continue
        peaks = detect_beats(segment, fs, local_period)
        found.extend((peaks + start).tolist())

    if not found:
        return np.array([], dtype=int)

    peaks = np.unique(np.asarray(found, dtype=int))

    # Merge duplicates introduced by the overlap: two detections closer than
    # half a local period cannot both be beats.
    merged = [peaks[0]]
    for p in peaks[1:]:
        local_period = period_track[p] if np.isfinite(period_track[p]) else np.inf
        if p - merged[-1] < 0.5 * local_period:
            # Keep whichever is the taller peak -- the true systolic maximum.
            if x[p] > x[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    return np.asarray(merged, dtype=int)


# --------------------------------------------------------------------------
# The three score components
# --------------------------------------------------------------------------

def _morphology_score(beats: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Correlation of each beat with its own local template, clipped to [0, 1].

    Beats and templates are zero-mean, unit-variance, so this is a normalised
    dot product: r = mean(beat * template).

    Negative correlation is clipped to zero rather than rectified. A beat
    anti-correlated with its neighbours is not a good beat that happened to
    flip sign; on a PPG it usually means the detector locked onto a diastolic
    trough, or the sensor lifted and the trace inverted. Either way it should
    score zero, not |r|.
    """
    if beats.size == 0:
        return np.array([])
    return np.clip(np.einsum("ij,ij->i", beats, templates) / beats.shape[1], 0.0, 1.0)


def _rhythm_score(peaks: np.ndarray, local_beats: int = 9) -> np.ndarray:
    """How well each inter-beat interval agrees with its local neighbourhood.

    This catches the failure mode morphology structurally cannot see: a missed
    or spurious detection. When a beat is missed the interval spanning the gap
    roughly doubles, yet the waveforms on either side may correlate perfectly
    with the template. Morphology notices nothing. Rhythm does.

    The reference is a rolling median of nearby intervals, not a global mean,
    because heart rate legitimately changes -- in this data by more than 90 bpm
    within one session. A global reference would flag the fast passages of a
    perfectly clean recording as arrhythmic.

    The 20% tolerance is roughly the largest beat-to-beat change respiratory
    sinus arrhythmia produces in a healthy adult at rest, and comfortably
    larger than anything exercise produces from one beat to the next. The score
    decays smoothly rather than switching, so a borderline interval degrades a
    beat's score instead of condemning it.
    """
    if len(peaks) < 3:
        return np.ones(len(peaks))

    ibi = np.diff(peaks).astype(float)
    # Mirror the first interval so every beat has one; dropping the first beat
    # instead would silently shorten every recording.
    ibi_per_beat = np.concatenate([[ibi[0]], ibi])

    half = local_beats // 2
    local_median = np.empty_like(ibi_per_beat)
    for i in range(len(ibi_per_beat)):
        lo, hi = max(0, i - half), min(len(ibi_per_beat), i + half + 1)
        local_median[i] = np.median(ibi_per_beat[lo:hi])

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_dev = np.abs(ibi_per_beat - local_median) / np.maximum(local_median, 1e-9)

    tolerance = 0.20
    return 1.0 / (1.0 + (rel_dev / tolerance) ** 2)


def _amplitude_score(x: np.ndarray, peaks: np.ndarray, periods: np.ndarray,
                     local_beats: int = 31) -> np.ndarray:
    """Plausibility of each beat's pulse amplitude relative to its neighbours.

    Beats are amplitude-normalised before the morphology comparison, which is
    what buys tolerance to skin tone and sensor gain -- but it also means a
    motion spike a hundred times the pulse amplitude, and a flatline where the
    sensor has lifted, both survive normalisation looking like ordinary beats.
    This term is where that information comes back.

    Scale comes from the median absolute deviation of neighbouring amplitudes,
    not their standard deviation. With a standard deviation a handful of large
    spikes inflate the scale until they no longer look unusual -- the artefact
    hides itself. The MAD does not break down until half the beats are corrupt.

    The comparison is local for the same reason the template is: perfusion
    changes over a recording, and during exercise it changes a lot. A global
    amplitude reference would reject the whole quiet half of a session.

    The deviation is two-sided on purpose. Unusually large amplitude means
    motion; unusually small means lost contact or collapsed perfusion. Both
    make downstream morphometry meaningless.
    """
    if len(peaks) == 0:
        return np.array([])

    amplitudes = np.empty(len(peaks))
    for i, p in enumerate(peaks):
        half = max(int(0.35 * periods[i]), 1) if np.isfinite(periods[i]) else 1
        lo, hi = max(0, p - half), min(len(x), p + half + 1)
        segment = x[lo:hi]
        amplitudes[i] = segment.max() - segment.min() if segment.size else 0.0

    half_n = local_beats // 2
    scores = np.empty(len(peaks))
    tolerance = 3.0  # ~3 robust sigmas, the usual outlier convention
    for i in range(len(peaks)):
        lo, hi = max(0, i - half_n), min(len(peaks), i + half_n + 1)
        neighbourhood = amplitudes[lo:hi]
        centre = np.median(neighbourhood)
        scale = robust_scale(neighbourhood)
        if scale <= 0:
            # Every neighbour has identical amplitude. Either the signal is
            # constant -- the morphology term will already have rejected it --
            # or it is unusually regular. Neither is evidence against this beat.
            scores[i] = 1.0
            continue
        z = abs(amplitudes[i] - centre) / scale
        scores[i] = 1.0 / (1.0 + (z / tolerance) ** 2)
    return scores


def _subharmonic_score(x: np.ndarray, peaks: np.ndarray,
                       relative_height: float = 0.5) -> np.ndarray:
    """Evidence that the beat train has locked onto *half* the true rate.

    Why this term exists
    --------------------
    It was added after the first version of this detector was caught being
    confidently wrong. On TROIKA recording 01 at t = 60 s the reference rate is
    103 bpm; the detector reported 51 bpm and accepted 86% of the beats it
    found. Nothing was malfunctioning: it had locked onto every second pulse,
    and a half-rate beat train is *internally consistent*. Its intervals are
    regular, so the rhythm term is satisfied. Its waveforms are near-identical,
    so the morphology term is satisfied -- more satisfied than usual, since
    every second pulse in this stretch is the taller one. Its amplitudes are
    uniform, so the amplitude term is satisfied.

    That is the structural blind spot of any self-consistency method, and it
    cannot be closed by making the other three terms stricter: consistency
    cannot detect an error that is itself consistent. It needs evidence of a
    different kind -- evidence about what was *left out*.

    So we look between the beats. If a peak of comparable height sits near the
    midpoint of an interval, that interval probably spans two beats rather than
    one. This is the classic octave-error check from pitch detection, and it is
    cheap: one max over a short slice per interval.

    The term deliberately does not try to *repair* the beat train. Re-running
    detection at double rate would be the obvious next step and is the right
    engineering answer, but it changes the beats other terms have already
    scored, and a quality module that silently rewrites its own input is harder
    to reason about than one that says "I do not trust this stretch". Refusing
    to answer is the behaviour this whole module exists to enable.
    """
    n = len(peaks)
    if n < 3:
        return np.ones(n)

    scores = np.ones(n)
    for i in range(n - 1):
        left, right = peaks[i], peaks[i + 1]
        interval = right - left
        if interval < 6:
            continue

        # Look in the middle 30% of the interval -- wide enough to find a
        # genuine intervening beat, narrow enough to exclude the flanking ones.
        lo = left + int(0.35 * interval)
        hi = left + int(0.65 * interval)
        if hi <= lo:
            continue

        middle = x[lo:hi]
        trough = min(x[left:right].min(), middle.min())
        flanking = 0.5 * (x[left] + x[right]) - trough
        if flanking <= 0:
            continue

        midpoint_height = (middle.max() - trough) / flanking
        # 1.0 means the intervening peak is as tall as the beats around it.
        # Score falls as that ratio approaches `relative_height`.
        excess = max(midpoint_height - relative_height, 0.0) / (1.0 - relative_height)
        penalty = min(excess, 1.0)
        # A suspect interval taints both beats that bound it.
        scores[i] = min(scores[i], 1.0 - penalty)
        scores[i + 1] = min(scores[i + 1], 1.0 - penalty)

    return scores


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------

def _choose_threshold(scores: np.ndarray, morph: np.ndarray) -> tuple[float, str]:
    """Split the score distribution into accept/reject without any labels.

    The honest difficulty: a two-component mixture will happily fit a unimodal
    distribution, inventing a boundary through the middle of a set of perfectly
    good beats -- or, worse, through a set of uniformly bad ones, thereby
    "accepting" the better half of a recording that should have been discarded
    entirely. A detector that always rejects some fixed share of the data is
    not detecting anything; it is just sorting.

    So the mixture is trusted only when it is justified. We fit one- and
    two-component models and compare BIC. If one component wins, the recording
    is homogeneous, and the remaining question -- all good or all bad -- cannot
    be answered from the shape of the distribution at all. That is the one
    place an absolute anchor is unavoidable, and we put it on the correlation
    term because a correlation coefficient means the same thing on every sensor
    and every subject.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return 0.0, "empty"
    if scores.size < 20:
        # Too few beats to infer a distribution; the anchor alone decides.
        return MIN_PLAUSIBLE_CORRELATION, "absolute-fallback-few-beats"

    X = scores.reshape(-1, 1)
    one = GaussianMixture(1, random_state=0).fit(X)
    two = GaussianMixture(2, random_state=0, n_init=3).fit(X)

    if two.bic(X) >= one.bic(X):
        median_morph = float(np.median(morph)) if morph.size else 0.0
        if median_morph >= MIN_PLAUSIBLE_CORRELATION:
            return float(scores.min()) - 1e-9, "uniform-good"
        return float(scores.max()) + 1e-9, "uniform-bad"

    # Two genuine populations: put the boundary where the posteriors cross.
    means = two.means_.ravel()
    grid = np.linspace(float(means.min()), float(means.max()), 512).reshape(-1, 1)
    posterior = two.predict_proba(grid)
    bad_component = int(np.argmin(means))
    crossing = int(np.argmin(np.abs(posterior[:, bad_component] - 0.5)))
    threshold = float(grid[crossing, 0])

    # Guard: never accept beats failing the absolute plausibility anchor, even
    # if the mixture would like to. This is what stops a recording of pure
    # noise from being neatly divided into "good noise" and "bad noise".
    floor = MIN_PLAUSIBLE_CORRELATION * 0.5
    if threshold < floor:
        return floor, "mixture-clamped-to-anchor"
    return threshold, "mixture"


# --------------------------------------------------------------------------
# The detector
# --------------------------------------------------------------------------

def assess(x: np.ndarray,
           fs: float,
           n_refine: int = 2,
           combine: str = "min",
           prefilter: bool = True,
           causal_filter: bool = False,
           template_neighbours: int = TEMPLATE_NEIGHBOURS) -> QualityResult:
    """Score every beat in one PPG recording.

    Parameters
    ----------
    x : raw single-channel PPG.
    fs : sampling rate in Hz.
    n_refine : template re-estimation rounds. Round 0 builds each local
        template from all neighbouring beats; each further round rebuilds it
        from the neighbours currently accepted. This is the
        expectation-maximisation step, and it matters most when a stretch is
        badly corrupted -- the initial median is dragged toward the artefacts,
        and refinement pulls it back.
    combine : how the three sub-scores become one ("min", "product",
        "geometric", "mean"). Provided as an option so the choice can be tested
        rather than asserted; see the ablation in the README.
    causal_filter : use the single-pass filter a microcontroller would run,
        to confirm results do not depend on zero-phase filtering.
    """
    x = np.asarray(x, dtype=float).ravel()
    filtered = ((bandpass_causal(x, fs) if causal_filter else bandpass(x, fs))
                if prefilter else x)

    period_track, acf_track = _track_period(filtered, fs)

    def _empty(mode: str) -> QualityResult:
        return QualityResult(
            peaks=np.array([], dtype=int), scores=np.array([]),
            accepted=np.array([], dtype=bool),
            components={k: np.array([]) for k in
                        ("morphology", "rhythm", "amplitude", "subharmonic")},
            templates=np.empty((0, 100)), threshold=0.0,
            periods=np.array([]), acf_strength=np.array([]),
            threshold_mode=mode, fs=fs, filtered=filtered,
        )

    if not np.isfinite(period_track).any():
        return _empty("no-period")

    peaks = _detect_beats_adaptive(filtered, fs, period_track)
    if len(peaks) < 3:
        return _empty("too-few-beats")

    # Segment each beat using the period that applies where it occurs. A single
    # median period would stretch slow beats and clip fast ones, and that
    # misalignment would show up as a loss of correlation -- i.e. it would
    # masquerade as poor signal quality.
    beat_periods = period_track[peaks]
    beats, peaks = _extract_beats_local(filtered, peaks, beat_periods)
    if len(peaks) < 3:
        return _empty("too-few-beats")
    beat_periods = period_track[peaks]

    rhythm = _rhythm_score(peaks)
    amplitude = _amplitude_score(filtered, peaks, beat_periods)
    subharmonic = _subharmonic_score(filtered, peaks)

    # Round 0: each beat's template is the pointwise median of its neighbours.
    #
    # Median, not mean. This is the load-bearing choice of the whole method. A
    # mean template is pulled toward whatever the artefacts look like, and a
    # template contaminated by artefacts then scores artefacts highly -- the
    # method would teach itself the wrong lesson and confirm it. The pointwise
    # median is unmoved until artefacts outnumber real beats in the
    # neighbourhood, which is the regime where no unsupervised method can work
    # anyway, and which we detect and report rather than paper over.
    mask = np.ones(len(peaks), dtype=bool)
    templates = _local_templates(beats, mask, template_neighbours)
    morphology = _morphology_score(beats, templates)
    scores = _combine(morphology, rhythm, amplitude, combine, subharmonic)
    threshold, mode = _choose_threshold(scores, morphology)
    accepted = scores >= threshold

    for _ in range(n_refine):
        if accepted.sum() < 3:
            break  # not enough clean beats left to define anything
        templates = _local_templates(beats, accepted, template_neighbours)
        morphology = _morphology_score(beats, templates)
        scores = _combine(morphology, rhythm, amplitude, combine, subharmonic)
        threshold, mode = _choose_threshold(scores, morphology)
        new_accepted = scores >= threshold
        if np.array_equal(new_accepted, accepted):
            break  # converged
        accepted = new_accepted

    return QualityResult(
        peaks=peaks,
        scores=scores,
        accepted=accepted,
        components={"morphology": morphology, "rhythm": rhythm,
                    "amplitude": amplitude, "subharmonic": subharmonic},
        templates=templates,
        threshold=threshold,
        periods=beat_periods,
        acf_strength=acf_track[peaks],
        threshold_mode=mode,
        fs=fs,
        filtered=filtered,
    )


def _extract_beats_local(x: np.ndarray, peaks: np.ndarray, periods: np.ndarray,
                         length: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fixed-length, amplitude-normalised waveform around each peak.

    The window runs from 0.35 periods before the peak to 0.65 after. That split
    is not arbitrary: the systolic upstroke occupies roughly the first third of
    a cardiac cycle and the diastolic decay with its dicrotic notch the
    remaining two thirds, so this captures one full pulse aligned on its most
    reliable landmark. Aligning on the peak rather than the foot matters
    because the foot is much harder to locate precisely, and alignment jitter
    would show up directly as lost correlation.

    Each beat is z-scored individually. This is what makes the method blind to
    absolute amplitude, and therefore to skin tone, sensor gain and contact
    pressure -- the things the problem statement says break hand-tuned
    thresholds. Amplitude is not discarded, it is scored separately on its own
    terms.
    """
    kept_beats, kept_peaks = [], []
    for p, period in zip(peaks, periods):
        if not np.isfinite(period) or period <= 0:
            continue
        start, stop = p - int(round(0.35 * period)), p + int(round(0.65 * period))
        if start < 0 or stop >= len(x) or stop - start < 4:
            continue  # partial beats at the edges are dropped, not padded
        segment = x[start:stop]
        resampled = np.interp(
            np.linspace(0, len(segment) - 1, length),
            np.arange(len(segment)), segment,
        )
        sd = resampled.std()
        if sd <= 0:
            continue
        kept_beats.append((resampled - resampled.mean()) / sd)
        kept_peaks.append(p)

    if not kept_beats:
        return np.empty((0, length)), np.array([], dtype=int)
    return np.asarray(kept_beats), np.asarray(kept_peaks, dtype=int)


def _local_templates(beats: np.ndarray, usable: np.ndarray,
                     n_neighbours: int) -> np.ndarray:
    """Pointwise median of each beat's temporal neighbourhood.

    A beat is excluded from its *own* template. Without that, a large artefact
    contributes to the very template it is scored against, inflating its
    correlation -- the classic form of self-confirmation, and the one an
    unsupervised method is most likely to fall into unnoticed.
    """
    n, length = beats.shape
    half = n_neighbours // 2
    templates = np.empty_like(beats)

    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = np.arange(lo, hi)
        idx = idx[(idx != i) & usable[idx]]
        if idx.size < 3:
            # Not enough accepted neighbours nearby: widen to the whole
            # recording rather than fabricate a template from two beats.
            idx = np.flatnonzero(usable)
            idx = idx[idx != i]
        if idx.size == 0:
            templates[i] = np.zeros(length)
            continue
        template = np.median(beats[idx], axis=0)
        sd = template.std()
        templates[i] = (template - template.mean()) / sd if sd > 0 else template
    return templates


def _combine(morphology: np.ndarray, rhythm: np.ndarray, amplitude: np.ndarray,
             how: str, subharmonic: np.ndarray | None = None) -> np.ndarray:
    """Fuse the three sub-scores.

    "min" is the default. The three terms describe three independent ways a
    beat can be unusable, and downstream morphometry does not care which one
    failed -- a beat with a perfect waveform sitting at an impossible interval
    is still unusable. The minimum asks "is this beat anomalous along *any*
    axis?", which is the question that matters, and it keeps the score
    interpretable: the combined value is always literally one of the three
    sub-scores, so when a beat is rejected we can say which test it failed.
    """
    terms = [morphology, rhythm, amplitude]
    if subharmonic is not None:
        terms.append(subharmonic)
    stack = np.vstack(terms)
    if how == "min":
        return stack.min(axis=0)
    if how == "product":
        return stack.prod(axis=0)
    if how == "geometric":
        return np.exp(np.mean(np.log(np.clip(stack, 1e-9, None)), axis=0))
    if how == "mean":
        return stack.mean(axis=0)
    raise ValueError(f"unknown combine mode: {how}")


# --------------------------------------------------------------------------
# Turning per-beat decisions into per-window answers
# --------------------------------------------------------------------------

def heart_rate_from_accepted(result: QualityResult, start: int, stop: int,
                             use_quality: bool = True,
                             min_intervals: int = 2) -> float:
    """Heart rate over [start, stop) from beat intervals.

    Returns NaN when too few usable intervals fall in the window. Declining to
    report is the correct behaviour for a quality gate -- the entire premise of
    Q3 is that a refused measurement beats a wrong one.

    A subtlety that is easy to get wrong, and that we got wrong first
    ------------------------------------------------------------------
    The obvious implementation -- keep the accepted beats, then difference them
    -- is a bug, and a bug that makes the quality gate look actively harmful.
    Dropping a rejected beat from the middle of a sequence does not remove its
    interval; it *merges* the two intervals either side of it into one that is
    roughly twice as long. Gating then injects exactly the kind of doubled
    interval it is supposed to protect against, and measured heart rate falls.
    On the Welltory recordings this made the gated estimate worse than the
    ungated one (MAE 8.8 vs 5.1 bpm) and the gate looked useless.

    The fix is to treat the *interval*, not the beat, as the unit of
    measurement: an interval is usable only if it is bounded by two beats that
    were adjacent in the original detection sequence and that were both
    accepted. A rejected beat therefore invalidates the two intervals touching
    it and creates none.

    The median is used rather than the mean because one surviving bad interval
    should not drag the estimate, and because heart rate over an 8 s window is
    not expected to be constant.

    `use_quality=False` gives the ungated control: the same estimator fed every
    detected beat. The comparison between the two is the core validation.
    """
    peaks = result.peaks
    if len(peaks) < 2:
        return float("nan")

    intervals = np.diff(peaks).astype(float)
    # An interval belongs to the window if its starting beat does.
    lefts = peaks[:-1]
    in_window = (lefts >= start) & (lefts < stop)

    if use_quality:
        # Both endpoints must have survived the quality test.
        both_accepted = result.accepted[:-1] & result.accepted[1:]
        in_window = in_window & both_accepted

    usable = intervals[in_window]
    if usable.size < min_intervals:
        return float("nan")
    return float(60.0 * result.fs / np.median(usable))


def window_accept_fraction(result: QualityResult, start: int, stop: int) -> float:
    """Fraction of detected beats in a window that passed the quality test.

    Reported as a fraction rather than a hard verdict so a consumer can choose
    its own operating point: a heart-rate estimate tolerates a couple of
    rejected beats, a pulse-transit-time measurement does not.
    """
    in_window = (result.peaks >= start) & (result.peaks < stop)
    n = int(in_window.sum())
    return float(result.accepted[in_window].mean()) if n else 0.0
