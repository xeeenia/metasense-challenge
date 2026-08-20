# Q3 — A self-supervised, zero-shot signal-quality detector for PPG

**Run it:** `python q3_signal_quality/run_q3.py` (about 3½ minutes; `--quick` skips
the ablation and causal-filter sweeps). All numbers below are reproduced in
`outputs/metrics.json`, and the figures in `outputs/` are committed so the results
can be read without running anything.

---

## The idea

Over a short stretch of time, real heartbeats resemble their neighbours and
artefacts resemble nothing. That asymmetry is enough to separate them without any
labels.

A PPG pulse is the optical shadow of a pressure wave that the same heart, in the
same arteries, produces a few dozen times a minute. Adjacent beats are near-copies:
their shape drifts with respiration and vascular tone, but over tens of seconds,
never abruptly. Motion, contact loss and ambient-light leakage obey no such
constraint — they are transient, not phase-locked to the cardiac cycle, and crucially
**not reproducible**. A second motion artefact looks nothing like the first.

So the method never models what an artefact looks like, which is fortunate: the space
of artefacts is unbounded and the space of clean beats is not. It only needs to know
what this subject's beats look like *right now*, and the recording supplies that for
free.

**Pipeline.** Band-pass → track the local cardiac period by autocorrelation → detect
beats with a locally-adaptive refractory distance → cut and amplitude-normalise each
beat → score each beat on four axes against a template built from its own temporal
neighbours → split the score distribution with a mixture model that has to justify
itself first.

## Why it should generalise where fixed thresholds do not

The brief's complaint about hand-tuned heuristics is that thresholds calibrated on one
population fail on another — darker skin absorbs more green light and lowers the AC/DC
ratio, a wrist sensor sees a different waveform from a fingertip, a loose strap changes
everything.

Every quantity here is either **dimensionless** (a correlation coefficient) or
**expressed relative to the recording's own robust statistics** (a MAD-scaled
deviation). Nothing is measured in the units of the sensor, so nothing has to be
recalibrated when the sensor, the site or the subject changes. The template is
re-estimated from scratch every time; there are no learned parameters to carry over
and therefore nothing to transfer badly.

### The four score components

| Term | Exploits | Catches | Blind to |
|---|---|---|---|
| **Morphology** — correlation with a local template | Beat-to-beat self-similarity | Distorted, clipped or fabricated waveforms | A perfect waveform at an impossible time |
| **Rhythm** — deviation from the rolling median inter-beat interval | Cardiac timing is smooth | Missed and spurious detections | Genuine arrhythmia (see limitations) |
| **Amplitude** — MAD-scaled deviation from neighbouring pulse amplitudes | Perfusion changes slowly | Motion spikes, flatlines, lost contact | Artefacts that preserve amplitude |
| **Subharmonic** — is there a comparable peak *between* two detected beats? | What the beat train left out | Octave errors, where the detector has locked onto half the true rate | Rate *doubling* (the mirror error) |

Combined with a **minimum**: four independent ways a beat can be unusable, and
downstream morphometry does not care which one failed. The minimum also keeps the
score interpretable — it is always literally one of the four sub-scores, so a
rejection can be explained.

### The fourth term exists because the first three were caught being wrong

This is the most instructive thing in this submission, so it gets its own section.

The first working version scored morphology, rhythm and amplitude, reached AUC 0.785,
and looked reasonable. Then a diagnostic plot showed the detector marking beats about
1.15 s apart on a stretch where the trace visibly peaked twice that often. Checking
against the reference:

```
t = 60 s   reference 103.2 bpm   detector 51.4 bpm   86% of beats ACCEPTED
```

It had locked onto every second pulse — and a half-rate beat train is **internally
consistent**. Its intervals are regular, so the rhythm term was satisfied. Its
waveforms were near-identical (every second pulse in that stretch is the taller one),
so the morphology term was *more* satisfied than usual. Its amplitudes were uniform,
so the amplitude term was satisfied. All three agreed enthusiastically on an answer
that was wrong by a factor of two.

That is the structural blind spot of any self-consistency method, and it cannot be
closed by making the existing terms stricter: **consistency cannot detect an error
that is itself consistent.** It needs evidence of a different kind — evidence about
what was left *out*. So the fourth term looks between the beats: if a peak of
comparable height sits near the midpoint of an interval, that interval probably spans
two beats. This is the classic octave-error check from pitch detection, and it costs
one max over a short slice per interval.

The effect:

| | AUC | Welltory MAE (gated) | Threshold-transfer accuracy |
|---|---|---|---|
| Three terms | 0.785 | 1.71 bpm *(worse than ungated)* | 0.698 |
| **Four terms** | **0.881** | **1.36 bpm** *(better than ungated)* | **0.821** |

And on the specific windows that exposed it, the detector now abstains instead of
answering confidently:

| t (s) | Reference | Estimate (ungated) | Accepted | Gated output |
|---|---|---|---|---|
| 10 | 71.7 | 73.5 ✓ | 100% | 73.5 ✓ |
| 60 | 103.2 | 51.4 ✗ *(halved)* | 29% | **refuses** ✓ |
| 90 | 118.5 | 73.5 ✗ | 0% | **refuses** ✓ |
| 120 | 143.3 | 150.0 ✓ | 68% | 163.0 |
| 150 | 155.3 | 82.9 ✗ *(halved)* | 8% | **refuses** ✓ |

The term does **not** try to repair the beat train. Re-running detection at double
rate is the obvious next step and is the right engineering answer, but it changes
beats the other terms have already scored, and a quality module that silently
rewrites its own input is harder to reason about than one that says "I do not trust
this stretch". Refusing to answer is the behaviour this module exists to enable.

### Two other choices that carry weight

**The template is a pointwise median, not a mean, and excludes the beat it scores.**
A mean template is pulled toward whatever the artefacts look like, and a template
contaminated by artefacts then scores artefacts highly — the method would teach itself
the wrong lesson and confirm it. The median is unmoved until artefacts outnumber real
beats in the neighbourhood. Excluding the beat from its own template closes the other
self-confirmation route.

**The threshold has to earn the right to exist.** A two-component mixture will happily
fit a unimodal distribution, inventing a boundary through the middle of a set of
perfectly good beats — or, worse, through a set of uniformly bad ones, "accepting" the
better half of a recording that should have been discarded entirely. A detector that
always rejects a fixed share of the data is sorting, not detecting. So we fit one- and
two-component models and compare BIC; if one wins, the recording is homogeneous and
the remaining question (all good or all bad) cannot be answered from the distribution's
shape. That is the one place an absolute anchor is unavoidable, and we put it on the
correlation term (r ≥ 0.5 — a beat sharing at least a quarter of its variance with its
neighbours) because a correlation coefficient means the same thing on every sensor.

## What we rejected, and why

**SimSiam-style contrastive learning / adversarial autoencoders.** The brief invites
these and we deliberately declined. This method *is* the degenerate case of a
contrastive one: positives are temporally adjacent beats, the representation is the
amplitude-normalised beat, the similarity metric is correlation. We keep the
self-supervision and drop the learned encoder — because an encoder must be pre-trained
on some corpus, and that corpus's distribution is exactly what the brief says goes
wrong at deployment. An encoder trained mostly on light skin and stationary subjects
has silently baked in a prior that a locally re-estimated template never acquires.
*Where it would genuinely help:* learning what a physiologically plausible pulse looks
like across the full range of vascular states, so a subject whose every beat is
abnormal can be flagged rather than accepted as internally consistent — the one
failure below that no local method can fix.

**A global template and a global cardiac period.** The first implementation, and
simply wrong: heart rate ranges over more than 90 bpm within a single TROIKA session,
so a global period mis-segments most of the recording and a global template averages
waveforms taken at 70 and 166 bpm. Everything is now estimated in a moving
neighbourhood — which is also what a device does, since it processes a rolling buffer.

**Autocorrelation `argmax` for period estimation.** Also wrong, and instructive. A PPG
carries strong low-frequency content, and during exercise respiration alone reaches
30–50 breaths/min — squarely inside the heart-rate search range. `argmax` returned
38–49 bpm for windows whose true rate was 110–155 bpm. The fundamental period is the
*first* lag at which the waveform repeats, so we take the earliest prominent
autocorrelation peak, with a guard against locking onto the dicrotic notch.

## Validation — designed to catch the method fooling itself

An unsupervised detector has two easy ways to look good without being good.

**Failure 1: sorting instead of detecting.** A method that always rejects the
worst-scoring third will show lower error on what it keeps, on *any* data. Closed by
comparing against a random rejector at identical coverage.

**Failure 2: circular labels.** With no expert annotations, the temptation is to
define "clean" using the same statistics the detector uses. Closed by taking ground
truth from a **different instrument**: TROIKA's simultaneously recorded ECG. The
reference heart rate is electrical, the detector sees only the optical trace, and the
two share no processing.

A window is labelled *usable* if the PPG-derived heart rate lands within ±5 bpm of the
ECG reference. This looks circular and is not: we label a window by whether the
**downstream estimate** was correct, then ask whether the **quality score** predicted
that. The score never sees the reference. This measures precisely the property a
clinical device needs — can it know, at the time of measurement and without a gold
standard, that it is about to report something wrong?

*(The section above is the third check: the failure mode that AUC alone hid was found
by plotting the detector's decisions against the raw trace and noticing the beat
spacing looked wrong. Aggregate metrics did not flag it — 0.785 looked fine.)*

### Data

| Dataset | What | Why this one |
|---|---|---|
| **TROIKA** (12 recordings, 11 subjects, 125 Hz) | Wrist PPG ×2 + ECG + accelerometer, treadmill exercise | Severe motion artefact and an electrical ground truth in the same recording. This is the deployment condition, not a laboratory one. |
| **Welltory** (21 recordings, 13 subjects, 30 Hz) | Smartphone-camera fingertip PPG + Polar H10 intervals | A different sensor, wavelength, site and cohort. Tests the cross-sensor claim rather than asserting it. |

*A note on channel selection:* in 7 of the 21 Welltory recordings the green channel is
flat to three decimal places while red carries a clean pulse. Hard-coding green — the
physiologically obvious default — would have silently discarded a third of the cohort,
and discarding the subjects whose sensor behaved unusually is exactly the quiet
selection bias that makes a quality detector look better than it is. The loader picks
the channel with the most cardiac content by autocorrelation strength.

---

## Results

**Does the score predict downstream failure?** TROIKA, 1726 windows, 11 subjects. 50%
of windows are usable at ±5 bpm — single-channel PPG during running is genuinely hard,
and that is the honest baseline.

| Method | AUC (predicts a usable window) |
|---|---|
| **Ours (self-supervised)** | **0.881** |
| Skewness SQI (Elgendi 2016) | 0.762 |
| Autocorrelation SQI | 0.607 |
| Zero-crossing SQI | 0.487 |

**Margin over the best baseline: +0.122, 95% CI [+0.049, +0.197]** — bootstrapped over
*subjects*, which is the level at which the data actually varies (windows overlap by
75% and come in runs of hundreds per person; a window bootstrap would treat 1726
correlated observations as independent and shrink the interval dishonestly). The
interval excludes zero, so this margin is real — unlike the three-term version's
+0.029 [−0.066, +0.124], which was not, and which is reported here rather than
quietly dropped.

**Risk–coverage** (`outputs/risk_coverage.png`) — mean |HR error| among retained
windows. Random rejection is flat by construction; the gap is the method's value.

| Coverage | Ours | Random rejection |
|---|---|---|
| 100% | 26.3 bpm | 26.3 bpm |
| 77% | 20.8 | 26.4 |
| 54% | 16.0 | 26.2 |
| 31% | 11.4 | 26.4 |
| 10% | 10.2 | 26.3 |

### The experiment that separates the methods on equal terms

AUC flatters a fixed-threshold index, because AUC is threshold-free — it quietly grants
the index an oracle threshold chosen with knowledge of the answers. A deployed device
carries a number chosen on somebody else's data. So: fit the threshold on ten subjects,
apply it unchanged to the eleventh.

| Method | In-domain | Transferred to a new subject | Drop |
|---|---|---|---|
| **Ours (adaptive — no threshold to transfer)** | **0.821** | **0.821** | **0.000** |
| Skewness SQI | 0.713 | 0.657 | −0.056 |
| Autocorrelation SQI | 0.589 | 0.566 | −0.024 |
| Zero-crossing SQI | 0.525 | 0.492 | −0.033 |

*(balanced accuracy, leave-one-subject-out)*

### Cross-sensor check

Welltory, 21 recordings, versus Polar H10 chest-strap reference — a different sensor,
wavelength, measurement site and cohort, with no parameter changed:

| | MAE | Beats retained |
|---|---|---|
| Ungated | 1.66 bpm | 100% |
| **Quality-gated** | **1.36 bpm** | 88% |

An 18% error reduction for a 12% data cost, on a modality the method was never
designed against. (With three terms this was 1.71 bpm — *worse* than ungated — which
is what first suggested something was wrong.)

### Controlled corruption

Injecting artefacts of known type at known times into the clean Welltory recordings —
the one experiment real data cannot provide, since it asks whether the detector finds
*the specific samples we ruined* rather than a downstream consequence:

**AUC = 0.891** over 1939 beats, 15% of which fall inside injected damage
(`outputs/synthetic_corruption.png`). Three artefact types: transient bursts,
flatlines and slow baseline excursions.

### Ablation

| Fusion rule | AUC | | Single component only | AUC |
|---|---|---|---|---|
| product | 0.889 | | **subharmonic** | **0.871** |
| **min (default)** | **0.881** | | morphology | 0.753 |
| geometric mean | 0.806 | | rhythm | 0.605 |
| mean | 0.712 | | amplitude | 0.532 |

Two honest observations.

**The subharmonic term alone nearly matches the full method** (0.871 vs 0.881). On
this dataset the dominant failure is not waveform corruption but beat-detection octave
error, and the term that catches it does most of the work. The self-similarity
machinery that motivated the whole design contributes real but secondary value. That
ordering was not what I expected and is worth stating plainly.

**`product` edges out `min`** by 0.008 — well inside the bootstrap width. We kept `min`
because the combined score stays interpretable (it is always literally one sub-score,
so a rejection can be attributed to a specific failure). That is a judgement about
explainability over a difference we cannot resolve, not a claim of superiority;
`--combine` exposes both.

Amplitude is nearly useless alone (0.532) but earns its place in the
controlled-corruption test, where it catches flatlines and spikes — a failure mode
TROIKA's continuous exercise artefacts do not represent.

### Does zero-phase filtering flatter the result?

`filtfilt` looks into the future; no device can. Re-running with the single-pass biquad
an MCU would actually execute: **AUC 0.881 → 0.854**. A real but modest cost, and the
conclusions hold.

---

## What this misses — stated plainly

1. **Periodic artefacts at the heart rate.** The load-bearing failure. Running cadence
   is typically 150–170 steps/min, overlapping the exercise heart-rate range. Such an
   artefact is periodic, self-similar, and — unlike an octave error — leaves no missing
   beat for the subharmonic term to find. It passes all four tests. This is structural.
   **The fix is not in this modality:** the accelerometer is already in the device and
   is the correct reference. We did not use it because Q3 asks about PPG signal quality,
   but a shipping version should.

2. **Rate doubling, the mirror of the error we fixed.** The subharmonic term asks
   whether a beat was *missed*. It says nothing about whether a dicrotic notch was
   counted as a beat. The rhythm term catches the intermittent case; a systematically
   doubled beat train would be as self-consistent as a halved one was, and would pass.
   The same "look at what was left out" trick does not have an obvious dual here.

3. **Uniformly corrupted recordings.** The median template survives until artefacts
   outnumber real beats. Past that, the template *is* the artefact and the method
   confidently accepts garbage. The BIC test detects homogeneity and the absolute
   correlation anchor is the backstop, but a recording that is uniformly, plausibly
   wrong will pass.

4. **Arrhythmia read as artefact.** Atrial fibrillation produces genuinely irregular
   intervals and variable morphology. Our rhythm term will reject those beats — the
   method discards exactly the patients a cardiometabolic wearable most needs to
   observe. This is the most clinically serious limitation here. Distinguishing
   "irregular because the sensor moved" from "irregular because the heart is
   fibrillating" needs the morphology and rhythm terms to disagree in a characteristic
   way, and validating that needs annotated AF data we do not have.

5. **The single absolute constant.** `MIN_PLAUSIBLE_CORRELATION = 0.5` is the one
   number not derived from the recording. Defensible and dimensionless, but a constant
   — "zero-tuning" would be an overclaim without naming it.

6. **Eleven subjects.** Confidence intervals are wide. The threshold-transfer result is
   the one we would most want replicated on a larger, more diverse cohort —
   particularly across skin tones, which neither dataset documents, and which is the
   specific generalisation the method claims.

7. **Beat detection is upstream of everything.** Sections above show quality assessment
   and beat detection are not really separable — the fourth term is a beat-detection
   diagnostic wearing a quality-score costume. Treating them as one problem would
   probably be better engineering than the clean separation this module pretends to.

---

## Files

| File | Contents |
|---|---|
| `quality.py` | The detector: period tracking, beat segmentation, the four score terms, template refinement, threshold selection |
| `baselines.py` | Skewness, autocorrelation and zero-crossing SQIs |
| `validate.py` | Risk–coverage, subject-level bootstrap, threshold transfer, artefact injection |
| `run_q3.py` | Runs all five experiments, writes `outputs/` |
| `../common/dsp.py` | Filtering, autocorrelation period estimation, beat extraction |
