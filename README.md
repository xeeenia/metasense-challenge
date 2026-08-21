# Metasense Research Engineer — Technical Screening Response

**Xu Yintong** · Questions **3** (self-supervised signal quality) and **4** (fitting a
model into a microcontroller budget)

---

## Quick start

```bash
git clone https://github.com/xeeenia/metasense-challenge.git && cd metasense-challenge
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt

python download_data.py                    # ~12 MB, two public datasets, ~1 min

python q3_signal_quality/run_q3.py         # ~3.5 minutes
python q4_edge_ml/run_q4.py                # ~46 minutes (--quick for ~4 min)
python cross_link.py                       # ~2 min; needs run_q3 + run_q4 outputs
```

Python 3.10+. Dependencies are NumPy, SciPy, scikit-learn and Matplotlib —
nothing else, and no deep-learning framework (see *Why no PyTorch or TensorFlow*
below).

**You do not have to run anything to read the results.** Every metric is committed
in `q3_signal_quality/outputs/metrics.json` and `q4_edge_ml/outputs/metrics.json`,
and every figure alongside them.

---

## Why these two questions

My background is machine learning and data science rather than embedded firmware,
so Q1 (flash write scheduling without DMA) would have produced a plausible-sounding
document I could not defend under questioning — and the brief is explicit that
indefensible work fails this screening regardless of how it reads.

Q3 and Q4 are where I can do work that stands up: unsupervised method design with a
validation scheme built to falsify itself, and quantified compression trade-offs
under a split that does not leak. Both are also the questions where the honest
answer turned out to be more interesting than the flattering one, which is most of
what is written up below.

I have tried throughout to make the *reasoning* the deliverable. Where a result is
weak, ambiguous, or contradicts what I set out to show, it is reported as such and
in the same typeface as the rest.

---

## Q3 — A self-supervised, zero-shot signal-quality detector

**[Full write-up →](q3_signal_quality/README.md)**

**The idea.** Over a short stretch of time, real heartbeats resemble their
neighbours and artefacts resemble nothing. That asymmetry is enough to separate
them without labels: build a robust template from each beat's temporal neighbours,
score each beat on morphology, rhythm, amplitude and a missing-beat check against
it, and let a mixture model split the score distribution — but only after it has proved, by BIC, that
there are two populations to split.

Every quantity is dimensionless or scaled by the recording's own robust statistics,
so nothing needs recalibrating when the sensor, site or subject changes. That is the
whole claim, and the validation is built to test it specifically rather than
generically.

**Headline results** (TROIKA, 1726 windows, 11 subjects, ECG ground truth):

- **AUC 0.881** for the quality score predicting whether a heart-rate estimate will
  land within ±5 bpm of the simultaneous ECG — versus 0.762 for the best classical
  index (skewness) and 0.607 for a fixed autocorrelation threshold. Margin **+0.122,
  95% CI [+0.049, +0.197]**, bootstrapped over subjects.
- **Threshold transfer:** fit a threshold on ten subjects, apply it to the eleventh,
  and skewness drops from 0.713 to 0.657 balanced accuracy. Ours has no threshold to
  transfer and stays at 0.821. Not a better statistic — one that does not need a
  constant carried across people.
- **Risk–coverage:** retaining the best-scoring 31% of windows cuts mean error from
  26.3 to 11.4 bpm; random rejection at the same coverage stays flat at 26.4.
- **Cross-sensor:** on smartphone-camera recordings against a Polar H10 reference,
  gating cuts error 1.66 → 1.36 bpm while retaining 88% of beats — a modality the
  method was never designed against.
- **Controlled corruption:** AUC 0.891 at finding artefacts we injected ourselves.

**The most instructive part of this answer is a bug.** The first version scored three
terms, reached AUC 0.785, and looked fine — while confidently reporting 51 bpm for a
window whose true rate was 103, with 86% of beats accepted. It had locked onto every
second pulse, and a half-rate beat train is *internally consistent*: regular
intervals, identical waveforms, uniform amplitudes. All three terms agreed
enthusiastically on an answer wrong by a factor of two. Consistency cannot detect an
error that is itself consistent. The fix — a fourth term asking whether a beat was
*left out* between two detections — took AUC from 0.785 to 0.881 and turned the
cross-sensor result from a negative into a positive. It was found by plotting
decisions against the raw trace, not by any aggregate metric.

**The failure that remains:** an artefact periodic at the heart rate — running cadence
at 150–170 steps/min — is self-similar, leaves no missing beat, and passes all four
tests. Structural, not a tuning problem; the correct fix is the accelerometer already
in the device.

---

## Q4 — Fitting a heart-rate regressor into a Cortex-M budget

**[Full write-up →](q4_edge_ml/README.md)**

**The setup.** A small 1-D CNN regresses heart rate from an 8-second, two-channel
PPG window (decimated to 25 Hz, so 2×200 samples), trained and evaluated
**leave-one-subject-out** across 11 subjects, then compressed to full-integer int8
and profiled against a stated Cortex-M4F budget.

**The result the brief specifically asked about.** The same model, same training,
evaluated with a random per-window split instead of a subject-disjoint one reports
**5.09 bpm against the honest 10.47 — understating the error by 51%.** TROIKA windows
advance 2 s but span 8 s, so a random split puts a test window's 75%-overlapping
neighbour in the training set. The leak is not just subject identity; it is temporal
overlap, and that is the larger half.

**What compression actually costs, in context.** int8 quantisation costs **0.35 bpm**
(10.47 → 10.82). Which subject you hold out costs up to **25 bpm** (best fold 4.97,
worst 30.28). The spread across people is roughly twenty times the cost of every
compression decision combined, so the per-subject table is the headline here and the
averaged MAE is a footnote. 50% magnitude pruning is free in accuracy — and worthless
in practice, because CMSIS-NN kernels are dense and a zeroed weight is still stored
and still multiplied.

**Everything is written out in NumPy** — the convolution's forward and backward
passes, the per-channel int8 quantiser, and an inference engine that after the input
conversion uses no floating point at all: int8 weights, int32 accumulators, and
rescaling by a fixed-point multiplier and arithmetic shift, exactly as CMSIS-NN
does. Both are checked rather than asserted: analytic gradients match central finite
differences to 6×10⁻⁹, and the fixed-point requantiser matches the real multiplier
it replaces to 5×10⁻¹⁰. `run_q4.py` runs both gates before it reports anything.

**The honest headline:** on this target, **memory was never the binding constraint**.
The model is 8.3 KB in int8 and 28.6 KB in float32, against a budget of a few
hundred KB; peak activation RAM is 2.8 KB. Quantisation's value here is not fitting
in flash — it is that the M4's FPU is scalar while its DSP extension packs two int8
MACs per instruction, which roughly halves inference time and energy. That is a real
benefit, but not the one the question's framing implies, and saying so is more useful
than manufacturing a memory crisis. Widening the model 4× makes it *worse* (11.28 vs
10.85) at 3.6× the flash: the binding constraint is information, not capacity.

**The two answers agree without being asked to.** Q3's detector knows nothing about
Q4's model, and Q4's model is never gated by it — yet Q3's per-subject usable-window
rate predicts Q4's per-subject error at **Spearman −0.88, p = 0.0003**. S10, Q4's
30 bpm catastrophe, has the lowest usable rate of any subject at 9%. That model is
not failing because it is a bad model; it is failing because that subject's PPG
largely does not contain a recoverable heart rate. Reproduce with
`python cross_link.py`.

---

## Why no PyTorch or TensorFlow

Two reasons, one practical and one that matters more.

The practical one: this has to clone and run on your machine. NumPy, SciPy and
scikit-learn are a small, stable dependency set; a deep-learning framework is a
large, version-sensitive one, and *"if the code does not run, we will not assess it"*
is a strong argument for fewer moving parts. (I did attempt a TensorFlow install
for TFLite's int8 converter; it was still building after forty minutes, which rather
made the point.)

The one that matters: Q4 asks what happens when a model is squeezed into integer
arithmetic. If the quantisation step is a converter call, the interesting part of
the answer — where the precision goes, which tensor's range is the problem, what the
accumulator has to be — is hidden inside a tool I would then have to explain anyway.
Writing it out means the operation counts in the budget are *counted*, not estimated,
and that I can defend the arithmetic line by line.

---

## Data

Both datasets are public, redistributed on GitHub under research-use terms.
`download_data.py` fetches them into `data/` (git-ignored), skipping anything
already present.

| Dataset | Content | Used for |
|---|---|---|
| **TROIKA** — Zhang, Pi & Liu, *IEEE TBME* 62(2), 2015 | 12 recordings, 11 subjects. Wrist PPG ×2, ECG, 3-axis accelerometer at 125 Hz during treadmill exercise; reference heart rate per 8 s window from the simultaneous ECG. | Q3 primary validation, Q4 entirely |
| **Welltory PPG** — Neshitov et al., *Sensors* 21(20), 2021 | 21 recordings, 13 subjects. Smartphone-camera fingertip PPG (~30 Hz) with Polar H10 RR intervals. | Q3 cross-sensor check and artefact injection |

TROIKA's exercise data was chosen over cleaner clinical PPG deliberately: a
signal-quality detector validated on resting ICU recordings proves nothing about a
wearable. Subject 04 contributed two sessions under two filenames; the loader maps
both to one subject identity, because splitting on filename would put the same
person on both sides of the boundary — the exact leak Q4 warns about, wearing a
disguise.

---

## Repository layout

```
├── download_data.py            fetch both datasets
├── cross_link.py               does Q3's quality score predict Q4's failures?
├── common/
│   ├── dsp.py                  filtering, autocorrelation period estimation,
│   │                           beat segmentation
│   └── datasets.py             loaders that keep subject identity attached
├── q3_signal_quality/
│   ├── README.md               method, alternatives rejected, results, limits
│   ├── quality.py              the detector
│   ├── baselines.py            skewness / autocorrelation / zero-crossing SQIs
│   ├── validate.py             risk-coverage, subject bootstrap, threshold
│   │                           transfer, artefact injection
│   ├── run_q3.py               entry point
│   └── outputs/                committed metrics and figures
└── q4_edge_ml/
    ├── README.md               architecture, split design, compression, budget
    ├── nn.py                   NumPy conv net, with a gradient check
    ├── data.py                 windowing, preprocessing, split construction
    ├── train.py                training loop and cross-validation
    ├── quantize.py             int8 PTQ and the integer-only inference engine
    ├── budget.py               MAC counts, RAM, flash, latency, energy
    ├── run_q4.py               entry point
    └── outputs/                committed metrics and figures
```

---

## On the use of AI assistants

I used Claude (Anthropic) throughout this exercise, so I want to be explicit about how it was used:

- **Scoping and background research.** Locating candidate public datasets, and
  reading around cuffless-BP and PPG signal-quality literature to work out which
  approaches were worth attempting in the available time.
- **Code.** Substantial parts of the implementation were drafted with assistance
  and then reviewed, corrected and in several places rewritten by me. The
  `quantize.py` integer inference path and the `validate.py` experiment design went
  through the most revision.
- **Prose.** The READMEs were drafted with assistance and edited by me.

What was *not* delegated: the method design for Q3, the decision to build the
validation around threshold transfer rather than AUC, the choice of datasets and the
reasoning about why TROIKA's exercise condition is the right stress test, the
decision to write the network in NumPy, and every judgement about which results to
report and how to characterise them.

Three of the bugs found during development are documented in the write-ups rather
than quietly fixed, because how they were found is more informative than the final
code: the autocorrelation `argmax` failure (period estimation returning 40 bpm for a
120 bpm signal), the heart-rate estimator that spanned gaps left by rejected beats
(which made the quality gate appear actively harmful), and the initial global
template that averaged waveforms across a 90 bpm range. All three produced
plausible-looking numbers before they were caught, which is the argument for the
falsification-oriented validation in Q3.

I can walk through any line of this repository and say why it is there.
