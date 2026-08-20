# Q4 — Fitting a heart-rate regressor into a Cortex-M budget

**Run it:** `python q4_edge_ml/run_q4.py` (about 40 minutes; `--quick` gives a
reduced version in ~4 minutes). All numbers below are in `outputs/metrics.json`, and
the figures in `outputs/` are committed so nothing needs re-running.

---

## Summary of what was found

1. **A random per-window split understates the error by 51%** — 5.09 bpm against the
   honest 10.47 bpm. The leak is not only subject identity; it is temporal overlap,
   and that is the larger half of it.
2. **int8 quantisation costs 0.35 bpm** (10.47 → 10.82), about 3%.
3. **Which subject you hold out costs up to 25 bpm.** The spread across people
   (sd 6.74, best 4.97, worst 30.28) is roughly twenty times the cost of every
   compression decision combined. Any report of a single averaged MAE hides this.
4. **Memory was never the binding constraint.** 8.3 KB int8, 28.6 KB float32,
   2.8 KB peak RAM, against a budget of a few hundred KB. Quantisation's real value
   on this part is that the M4's FPU is scalar while its DSP extension packs two
   int8 MACs per instruction — roughly halving latency and energy.
5. **50% pruning is free in accuracy and worthless in practice** — on a dense
   CMSIS-NN kernel a zeroed weight is still stored and still multiplied.
6. **Q3's quality detector predicts which subjects this model fails on**
   (Spearman −0.77, p = 0.005), without seeing the model at all.

---

## The task and the model

**Input:** 8 s of two-channel wrist PPG. Band-passed 0.5–8 Hz and decimated 125 → 25 Hz,
so the model sees 2×200 samples. This is the single largest memory saving available
and it costs nothing measurable: the information lives below 8 Hz, 25 Hz leaves
12.5 Hz of Nyquist headroom, and every downstream activation buffer shrinks tenfold.
Doing it *before* the model rather than making the model learn to ignore high
frequencies is the whole point.

Each window is z-scored per channel — a local operation computable on-device from the
buffer already in RAM. That keeps it out of the leakage discussion entirely: there is
no training-set statistic to accidentally apply to the test set. It also removes the
nuisance variables a wearable cannot control (LED current, photodiode gain, skin
tone, contact pressure), all of which scale the trace and none of which change the
heart rate.

**Target:** reference heart rate from the simultaneously recorded ECG.

**Why heart rate and not blood pressure.** The brief allows either. Blood pressure
would be closer to Metasense's actual problem, and I chose against it deliberately:
the public BP datasets either have no subject identifiers at all (the widely-used
UCI cuffless-BP set is MIMIC records sliced into anonymous rows) or need credentialed
access. Without subject identity there is no way to build a subject-disjoint split,
and Q4 names that split as the thing it is checking. Choosing a target where the
honest split is *constructible* seemed better than choosing the more impressive
target and quietly reporting a leaky number — which is exactly the failure the brief
says will weigh more heavily than any technical shortfall.

**Architecture** — three strided convolutions, global average pooling, two dense
layers; 7,145 parameters.

```
input 2×200
Conv1D(2→16, k=7, s=2) → ReLU      → 16×97
Conv1D(16→24, k=5, s=2) → ReLU     → 24×47
Conv1D(24→32, k=5, s=2) → ReLU     → 32×22
GlobalAvgPool                       → 32
Dense(32→32) → ReLU                 → 32
Dense(32→1)                         → 1
```

The receptive field after three layers spans ~1.5 s of the 8 s window — comfortably
more than one cardiac cycle at any plausible rate, which is the requirement: a model
that cannot see a whole beat cannot measure its period. Global pooling rather than
flattening because flattening 32×22 into the first dense layer would need 704 inputs
and ~22k extra parameters, three times the rest of the network, and because a pulse
eight seconds in is the same evidence as a pulse two seconds in.

**Rejected:** LSTM/GRU (poor integer kernel support on Cortex-M, no clean full
integer path, unbounded latency); Transformer (parameters and activation memory far
past the budget, and attention over 200 samples would be architecture name-dropping);
classical spectral features plus a small regressor — which would very likely *win* on
accuracy at a fraction of the size, but has no compressible object in it, so it
answers a different question than the one asked. That trade is stated here rather
than buried: if the goal were shipping the best heart-rate estimate under this
budget, a spectral peak-tracker with the accelerometer for cancellation would be the
engineering answer, not this network.

## Why no PyTorch or TensorFlow

The convolution's forward and backward passes, Adam, the int8 quantiser and the
integer-only inference engine are all written out in `nn.py` and `quantize.py`.

Practically: a small, stable dependency set is likelier to run on your machine than a
version-sensitive framework install. (I did try TensorFlow for TFLite's converter; it
was still building after forty minutes.)

More importantly: Q4 asks what happens when a model is squeezed into integer
arithmetic. If quantisation is a converter call, the interesting part of the answer —
where precision goes, which tensor's range is the problem, what the accumulator must
be — is hidden inside a tool I would have to explain anyway. Writing it out means the
MAC counts in `budget.py` are *counted* rather than estimated.

**Both hand-written pieces are checked, not asserted.** `run_q4.py` runs these gates
before reporting anything and exits if the first fails:

| Check | Result |
|---|---|
| Analytic gradients vs central finite differences, every layer | max relative error **6.09 × 10⁻⁹** |
| Fixed-point requantiser vs the real multiplier it replaces, 10,000 samples | max relative error **4.59 × 10⁻¹⁰** |

## The split

**Leave-one-subject-out over 11 subjects.** With eleven people, any single held-out
split says more about which two subjects landed in the test set than about the model.
Rotating through all of them uses every subject as test exactly once and produces a
per-subject error distribution — which, as the results show, is the number that
actually matters.

Three details that are easy to get wrong:

- **The validation subject is also held out**, and rotates, and is never the test
  subject. Selecting the early-stopping epoch on the test subject would be a subtler
  version of the same leak — the reported error would be a best-case over epochs.
- **Split on subject, not on file.** TROIKA's subject 04 contributed two sessions
  under two filenames. Splitting on filename would put the same person on both sides
  of the boundary — the leak Q4 warns about, wearing a disguise.
- **Target standardisation uses training-fold statistics only.**

---

## Results

### Per-subject, subject-disjoint (`outputs/accuracy.png`)

| Held-out subject | float32 | int8 PTQ | pruned 50% + int8 |
|---|---|---|---|
| S01 | 11.47 | 11.21 | 12.42 |
| S02 | 14.29 | 14.40 | 15.95 |
| S03 | 7.19 | 6.93 | 8.50 |
| S04 | 9.20 | 11.30 | 11.24 |
| S05 | 6.37 | 6.54 | 5.54 |
| S06 | 6.87 | 7.00 | 7.54 |
| S07 | **4.97** | 6.25 | 5.02 |
| S08 | 9.35 | 10.04 | 6.13 |
| **S10** | **30.28** | 29.53 | 28.86 |
| S11 | 6.81 | 6.90 | 7.10 |
| S12 | 8.43 | 8.91 | 9.04 |
| **mean ± sd** | **10.47 ± 6.74** | **10.82 ± 6.41** | **10.67 ± 6.54** |

*(MAE in bpm)*

**The compression columns are nearly indistinguishable; the rows are not.** int8 costs
0.35 bpm on average. Being subject S10 costs 25 bpm. Reporting "int8 MAE 10.82" as
this system's accuracy would be technically true and substantively misleading, which
is why the per-subject table is the headline and the mean is a footnote.

S10 is not a bug — it is a subject whose wrist PPG is dominated by motion throughout.
See the cross-link below.

### The leakage demonstration

Identical model, identical training, identical hyperparameters. Only the split
changes.

| Split | MAE |
|---|---|
| Random per-window | **5.09 bpm** |
| Subject-disjoint (honest) | **10.47 bpm** |

**The leaky split understates the error by 5.38 bpm — 51% of the true value.**

Two mechanisms, and the less obvious one is larger. The obvious leak is subject
identity: windows from one person appear in both train and test, so the model can
learn that person's resting rate and waveform. The bigger one is **temporal overlap**
— TROIKA windows advance 2 s but span 8 s, so any test window shares 75% of its
samples with a neighbour that a random split almost certainly placed in training. The
model is being asked to recall, not to generalise. A paper reporting 5.09 bpm here
would not be reporting a better model; it would be reporting a different and much
easier question.

### int8 quantisation

Per-output-channel symmetric weights, per-tensor asymmetric activations, int32
accumulators, requantisation by fixed-point multiply and arithmetic shift. After the
input conversion there is no floating point in the forward pass.

- **Mean cost: +0.35 bpm** (10.47 → 10.82), about 3%.
- **But individual predictions move more than the aggregate suggests:** the largest
  disagreement between the float and int8 model on a single window is several bpm.
  The MAE is stable because the errors are not systematically signed, not because
  each prediction is preserved. For a device reporting per-window values rather than
  trends, that distinction matters and the aggregate hides it.
- **One fold degrades clearly:** S04, 9.20 → 11.30. S04 is the two-session subject
  and has the widest input range; a per-tensor activation scale that must cover both
  sessions quantises each more coarsely. This is the mechanism per-channel activation
  scales would fix, at the cost of runtime float work.

Design choices, stated because they are where 8-bit PTQ succeeds or fails:

- **Weights symmetric (zero-point 0)** — a weight zero point introduces a cross term
  into every accumulation that must then be computed and subtracted. Forcing it to
  zero makes it vanish.
- **Weights per output channel** — trained conv channels routinely differ in
  magnitude by an order of magnitude; one shared scale quantises the small channels
  into a handful of levels. This is what makes post-training int8 viable at all.
- **Activations asymmetric** — a ReLU output is one-sided; symmetric quantisation
  would spend half the codes on impossible values.
- **Ranges from training data only.** Calibration is a fitted parameter of the
  deployed artefact; fitting it on test data is a leak of the same family as a leaky
  split.
- **No percentile clipping.** Standard practice for image networks, wrong here: the
  tail of an activation distribution is often exactly the motion artefact or the
  unusually fast beat we need the model to register.

### Pruning — a negative result worth keeping

50% magnitude pruning, layer-wise, with fine-tuning and the mask re-applied after
every optimiser step (Adam's momentum would otherwise drift pruned weights back off
zero and the sparsity would silently evaporate).

**Accuracy: 10.67 bpm — statistically indistinguishable from the 10.82 unpruned int8
model.** Half the weights are doing nothing.

**And it buys nothing.** This is *unstructured* sparsity, and CMSIS-NN's kernels are
dense: a zeroed weight is still stored and still multiplied. Realising the saving
needs either a sparse format, whose index overhead at these sizes can exceed what it
saves, or structured pruning that removes whole channels. What this measures is
therefore the **accuracy headroom** pruning offers — a real and necessary prerequisite
— not a saving. Reporting "50% pruned" as a compression result would be an overclaim.

The width sweep below is the honest version of the same experiment: removing whole
channels, which does shrink the model.

### Microcontroller budget

Assumed target: **Cortex-M4F at 80 MHz, 256 KB SRAM, 1 MB flash** — an STM32L4 or
nRF52840 class part, and the class the brief describes.

| Layer | Output | MACs | int8 bytes | float32 bytes |
|---|---|---|---|---|
| conv1d_0 | 16×97 | 21,728 | 416 | 960 |
| conv1d_2 | 24×47 | 90,240 | 2,208 | 7,776 |
| conv1d_4 | 32×22 | 84,480 | 4,224 | 15,488 |
| gap | 32 | 704 | 0 | 0 |
| dense_7 | 32 | 1,024 | 1,408 | 4,224 |
| dense_9 | 1 | 32 | 44 | 132 |
| **total** | | **198,208** | **8,300** | **28,580** |

*(int8 bytes include int32 biases and the per-channel requantisation multipliers —
840 B of the 8,300. These are routinely forgotten in "model size" figures and are 10%
of this one.)*

| Quantity | Value | Derivation |
|---|---|---|
| Flash, model only | 8.3 KB int8 / 28.6 KB float32 | counted above |
| Flash, with runtime | 20 – 108 KB | + 12 KB hand-rolled CMSIS-NN chain, or + 100 KB TFLite Micro interpreter |
| Peak activation RAM | 2,808 B | largest adjacent buffer pair under ping-pong (2,680 B) + im2col scratch (128 B) |
| Latency, int8 | 1.24 – 2.48 ms | 198,208 MACs ÷ (1–2 MAC/cycle) ÷ 80 MHz |
| Latency, float32 | 2.48 – 4.96 ms | same MACs at 0.5–1 MAC/cycle (scalar FPU, no SIMD packing) |
| CPU duty cycle | 0.124% | 2.48 ms every 2 s |
| Energy per inference | ~74 µJ | 10 mA × 3.0 V × 2.48 ms |

Ranges rather than single numbers where the value depends on something not measurable
without the hardware: the 2 MAC/cycle int8 ceiling comes from SMLAD packing two 16-bit
MACs, which real CMSIS-NN convolution does not reach once address generation, the
im2col copy and loop overhead are counted. Headline figures use the pessimistic end.

**The honest conclusion, which is not the one the question's framing implies.** At
8.3 KB of weights and 2.8 KB of RAM against a few hundred KB of budget, this model
never had a memory problem — the float32 version fits comfortably too. Quantisation
still earns its place, for a different reason: the M4's FPU is *scalar*, so float
inference gets none of the two-way packing int8 enjoys, and int8 roughly halves both
latency and energy per inference. At a 0.124% duty cycle even that is not the
binding constraint on battery life; the optical front end and the radio dominate by
orders of magnitude.

If the memory budget were genuinely tight, the sweep below shows where.

### Width sweep — where does the budget bind?

| Width | Params | Flash (int8) | RAM | Latency | MAE |
|---|---|---|---|---|---|
| 0.5 | 1,877 | 2.5 KB | 1.4 KB | 0.69 ms | 12.57 |
| **1.0** | **7,145** | **8.3 KB** | **2.8 KB** | **2.48 ms** | **10.85** |
| 2.0 | 27,857 | 30.2 KB | 5.6 KB | 9.35 ms | 11.28 |

*(3 folds each, not 11 — the point is the shape of the curve, and resolving it more
precisely would not change the conclusion. Labelled as reduced rather than presented
as comparable to the main table.)*

**The curve turns back up.** Quadrupling the model from width 1.0 to 2.0 makes it
*worse* — 11.28 against 10.85 — while costing 3.6× the flash and 3.8× the latency.
With roughly 430 effectively independent training windows, 28k parameters is past the
point where more capacity is usable, and the model starts fitting the training
subjects rather than the physiology.

So on this data the binding constraint is neither flash nor RAM nor cycles: it is
**information**. There is only so much heart rate recoverable from single-modality
PPG during vigorous motion, and no amount of capacity manufactures more of it. The
route to a better number runs through the accelerometer, not through a bigger
network — which is the same conclusion Q3 reaches from the opposite direction.

### Cross-link to Q3 — an unplanned check that worked

The Q3 detector was built with no knowledge of this model, and this model is never
gated by the quality score. If both measure something real, the subjects Q3 judges to
have unusable signal should be the subjects Q4 gets wrong. Reproduce with
`python cross_link.py`:

| Q3 statistic vs Q4 per-subject MAE | Spearman | Pearson |
|---|---|---|
| Mean quality score | −0.773 (p = 0.005) | −0.470 (p = 0.14) |
| **Usable-window rate** | **−0.882 (p = 0.0003)** | −0.678 (p = 0.02) |

Spearman is the headline — the relationship is monotonic but not linear, and S10 is an
extreme MAE outlier that dominates a Pearson correlation. Both are reported.

S10, the 30 bpm failure, has by far the lowest usable-window rate of any subject
(9%, against 97% for the best). **This model is not failing on S10 because it is a bad
model; it is failing because S10's PPG largely does not contain a recoverable heart
rate.** A deployed system would gate this regressor's output on the quality score and
report nothing for those windows, which is precisely the architecture Q3 argues for.
That the two independently-built components agree at p = 0.005 is the strongest
evidence in this submission that either of them measures something real.

---

## What this misses — stated plainly

1. **The accelerometer is right there and we did not use it.** TROIKA provides
   3-axis accelerometry, and the published methods that reach ~2 bpm on this dataset
   all use it for motion cancellation. Our 10.47 bpm is a PPG-only number and should
   be read as such — it is **not comparable** to headline TROIKA results, and quoting
   it against them would be exactly the "benchmark comparison against differently
   validated results" the brief warns about.
2. **Eleven subjects, all healthy adults on a treadmill.** No older subjects, no
   cardiovascular disease, no documented skin-tone range, no daily-living motion.
   Every interval here is wide and the population is narrow.
3. **1,726 windows, but far fewer independent ones.** 75% overlap means the effective
   sample size is closer to 430. The subject-disjoint split handles the leakage
   correctly, but the *statistical* power is smaller than the window count suggests.
4. **No on-device measurement.** Latency and energy are derived from MAC counts and
   published throughput figures, not measured on silicon. The MAC counts are exact;
   the cycles-per-MAC assumption is where the uncertainty lives, and it is given as a
   range for that reason.
5. **Quantisation-aware training was not attempted.** PTQ costs only 0.35 bpm here so
   the headroom is small, but S04's 2.1 bpm degradation is the kind of per-subject
   failure QAT addresses, and we have not shown it would not help.
6. **Heart rate is the easy target.** It is periodic, and the network can plausibly
   learn something close to a matched filter. Blood pressure — Metasense's actual
   problem — depends on pulse morphology and transit time, is far more sensitive to
   quantisation of the waveform's fine structure, and would very likely show a much
   larger int8 cost. The 3% figure here should not be extrapolated to it.

---

## Files

| File | Contents |
|---|---|
| `nn.py` | Conv1D/Dense/ReLU/GlobalAvgPool with forward and backward passes, Adam, Huber loss, gradient check |
| `data.py` | Windowing, band-pass and decimation, per-window normalisation, subject-disjoint and deliberately-leaky split construction |
| `train.py` | Training loop with early stopping, cross-validation, aggregation |
| `quantize.py` | Per-channel int8 PTQ, fixed-point requantisation, integer-only inference, magnitude pruning |
| `budget.py` | Exact MAC counts, RAM/flash/latency/energy accounting, fixed-point verification |
| `run_q4.py` | Runs all five stages, writes `outputs/` |
