"""
Window extraction and splitting for Q4.

Two decisions in this file carry most of the weight of the whole answer.

The first is the input representation. The raw window is 8 s of two-channel PPG
at 125 Hz -- 2000 samples, 8 KB in float32, and more than a small Cortex-M
wants to hold, let alone convolve over. We band-pass and decimate to 25 Hz, so
the model's input is 2x200. This is not compression for its own sake: the
information we need lives between 0.5 and 8 Hz, a 25 Hz sample rate has 12.5 Hz
of Nyquist headroom over that, and the tenfold reduction in input size shrinks
every downstream activation buffer by the same factor. Doing this *before* the
model, rather than making the model learn to ignore high frequencies, is the
single largest memory saving available and it costs nothing we can measure.

The second is the split, which is the part Q4 warns about explicitly. See
`subject_folds` below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from common.datasets import Recording, load_troika, troika_windows

TARGET_FS = 25.0
WINDOW_S = 8.0
INPUT_LENGTH = int(WINDOW_S * TARGET_FS)   # 200


@dataclass
class Dataset:
    x: np.ndarray          # (n_windows, n_channels, INPUT_LENGTH), normalised
    y: np.ndarray          # (n_windows,) reference heart rate in bpm
    subject: np.ndarray    # (n_windows,) subject identifier
    record: np.ndarray     # (n_windows,) recording identifier

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.x[mask], self.y[mask], self.subject[mask],
                       self.record[mask])


def _preprocess(windows: np.ndarray, fs: float) -> np.ndarray:
    """Band-pass, decimate to TARGET_FS, then normalise each window.

    Normalisation is per window and per channel: subtract the mean, divide by
    the standard deviation. This is deliberately a *local* operation, computable
    on-device from the buffer already in RAM, with no dataset statistics
    involved. That property is what keeps it out of the leakage discussion
    entirely -- there is no training-set mean to accidentally apply to the test
    set, because there is no training-set mean.

    It also removes exactly the nuisance variables we cannot control in a
    wearable: LED drive current, photodiode gain, skin tone, contact pressure.
    All of them scale the trace; none of them changes its heart rate.
    """
    decimation = int(round(fs / TARGET_FS))
    n, channels, _ = windows.shape

    # Zero-phase band-pass before decimating. Filtering first matters: content
    # above 12.5 Hz would otherwise alias down into the cardiac band, where no
    # later stage could tell it apart from signal.
    nyq = 0.5 * fs
    b, a = sps.butter(4, [0.5 / nyq, 8.0 / nyq], btype="band")

    out = np.empty((n, channels, INPUT_LENGTH))
    for i in range(n):
        for c in range(channels):
            filtered = sps.filtfilt(b, a, windows[i, c])
            reduced = filtered[::decimation][:INPUT_LENGTH]
            if len(reduced) < INPUT_LENGTH:            # short final window
                reduced = np.pad(reduced, (0, INPUT_LENGTH - len(reduced)))
            sd = reduced.std()
            out[i, c] = (reduced - reduced.mean()) / sd if sd > 0 else 0.0
    return out


def build_dataset(recordings: list[Recording] | None = None) -> Dataset:
    """Assemble every reference window from TROIKA into one array."""
    recordings = recordings or load_troika()
    xs, ys, subjects, records = [], [], [], []

    for rec in recordings:
        windows, bpm = troika_windows(rec)
        if len(windows) == 0:
            continue
        xs.append(_preprocess(windows, rec.fs))
        ys.append(bpm)
        subjects.append(np.full(len(bpm), rec.subject))
        records.append(np.full(len(bpm), rec.record_id))

    return Dataset(
        x=np.concatenate(xs), y=np.concatenate(ys),
        subject=np.concatenate(subjects), record=np.concatenate(records),
    )


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def subject_folds(data: Dataset) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Leave-one-subject-out folds. Returns (train, val, test) index arrays.

    Why leave-one-subject-out rather than a single held-out split: with eleven
    subjects, any single split puts one or two people in the test set, and the
    result then says more about which two people they were than about the
    model. Rotating through every subject uses all the data for testing exactly
    once and gives a per-subject error distribution, which is the number that
    actually matters for a wearable -- a device that works well on average and
    fails on one person in ten is not a device you can ship.

    The validation subject, used for early stopping, is also held out of
    training and is never the test subject. Selecting the stopping epoch on the
    test subject would be a subtler form of the same leak Q4 warns about: the
    test error would then be a best-case over epochs rather than an honest
    estimate.

    Note `subject`, not `record`: TROIKA's subject 04 appears in two files, and
    splitting on filename would put the same person's two sessions on opposite
    sides of the boundary. That is the leak this whole function exists to
    prevent, wearing a disguise.
    """
    subjects = np.unique(data.subject)
    folds = []
    for i, held_out in enumerate(subjects):
        test = data.subject == held_out
        # Rotate the validation subject so it is not always the same person.
        val_subject = subjects[(i + 1) % len(subjects)]
        val = data.subject == val_subject
        train = ~test & ~val
        folds.append((np.flatnonzero(train), np.flatnonzero(val),
                      np.flatnonzero(test)))
    return folds


def random_window_folds(data: Dataset, n_folds: int = 11,
                        seed: int = 0) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Randomly split *windows*, ignoring subject identity. Deliberately wrong.

    This exists to be measured, not used. Q4 says a random per-window split
    "leaks information and will read here as a red flag", and the most useful
    thing we can do with that warning is quantify it on this exact data rather
    than nod at it.

    The leak has two channels here and both are worth naming. The obvious one
    is subject identity: windows from one person appear in train and test, so
    the model can learn that person's resting rate and waveform rather than
    anything general. The less obvious and larger one is temporal overlap --
    consecutive TROIKA windows advance by 2 s but span 8 s, so a test window
    shares 75% of its samples with its neighbour, and a random split all but
    guarantees that neighbour is in the training set. The model is being asked
    to recall, not to generalise.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))
    chunks = np.array_split(order, n_folds)

    folds = []
    for i in range(n_folds):
        test = chunks[i]
        val = chunks[(i + 1) % n_folds]
        train = np.concatenate([chunks[j] for j in range(n_folds)
                                if j != i and j != (i + 1) % n_folds])
        folds.append((train, val, test))
    return folds


def standardise_target(y_train: np.ndarray) -> tuple[float, float]:
    """Mean and standard deviation of the training targets only.

    Returned rather than applied so the caller has to be explicit about which
    split produced them. Heart rate in bpm has a mean near 120 and a spread
    near 25 on this data; regressing it raw would make the network spend its
    first epochs learning the offset, and would put the output layer's weights
    on a completely different scale from every other layer -- which later
    becomes a quantisation problem, since a per-tensor scale has to cover
    whatever range the weights actually occupy.
    """
    return float(np.mean(y_train)), float(np.std(y_train) + 1e-8)
