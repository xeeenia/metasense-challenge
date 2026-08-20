"""
Loaders for the two public datasets, with subject identity kept attached.

The one invariant this module enforces is that every record carries the
identifier of the *person* it came from, not just of the file. TROIKA's subject
04 contributed two sessions under two filenames; treating those as two subjects
would let the same person appear on both sides of a train/test split, which is
precisely the leak Q4 is asked to avoid. The mapping is therefore done here,
once, rather than in each experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TROIKA_FS = 125.0

# Row layout of TROIKA's `sig` matrix, from the dataset's own documentation.
TROIKA_ECG_ROW = 0
TROIKA_PPG_ROWS = (1, 2)          # two wrist sensors, ~2 cm apart
TROIKA_ACC_ROWS = (3, 4, 5)       # x, y, z

# Reference heart rate is provided per 8 s window advancing in 2 s steps.
TROIKA_WINDOW_S = 8.0
TROIKA_HOP_S = 2.0


@dataclass
class Recording:
    """One continuous recording from one subject."""

    subject: str          # person identity -- the unit that splits must respect
    record_id: str        # file-level identity (a subject may have several)
    fs: float
    ppg: np.ndarray       # (n_channels, n_samples)
    ecg: np.ndarray | None = None
    acc: np.ndarray | None = None
    ref_bpm: np.ndarray | None = None   # one value per analysis window
    ref_rr_ms: np.ndarray | None = None  # beat-to-beat intervals, if available
    channel: str = ""     # which colour channel was used (Welltory only)

    @property
    def duration_s(self) -> float:
        return self.ppg.shape[-1] / self.fs


# --------------------------------------------------------------------------
# TROIKA
# --------------------------------------------------------------------------

def _troika_records(root: Path) -> list[str]:
    files = sorted(p.name for p in (root / "troika").glob("DATA_*.mat")
                   if not p.name.startswith("._"))
    return [f[len("DATA_"):-len(".mat")] for f in files]


def load_troika(data_dir: Path | None = None) -> list[Recording]:
    """Load all TROIKA recordings.

    Reference heart rate comes from the simultaneously recorded ECG, which is
    what makes this dataset usable as ground truth: the target is not another
    PPG algorithm's opinion, it is the electrical activity of the same heart.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    out: list[Recording] = []

    for rec in _troika_records(root):
        sig = sio.loadmat(root / "troika" / f"DATA_{rec}.mat")["sig"]
        ref = sio.loadmat(root / "troika" / f"REF_{rec}.mat")["BPM0"].ravel()

        # "04_TYPE01" and "04_TYPE02" are two sessions from the same person.
        subject = f"S{rec.split('_')[0]}"

        out.append(
            Recording(
                subject=subject,
                record_id=rec,
                fs=TROIKA_FS,
                ppg=np.asarray(sig[list(TROIKA_PPG_ROWS)], dtype=float),
                ecg=np.asarray(sig[TROIKA_ECG_ROW], dtype=float),
                acc=np.asarray(sig[list(TROIKA_ACC_ROWS)], dtype=float),
                ref_bpm=np.asarray(ref, dtype=float),
            )
        )

    if not out:
        raise FileNotFoundError(
            f"no TROIKA recordings under {root / 'troika'}; run download_data.py"
        )
    return out


def troika_windows(rec: Recording,
                   window_s: float = TROIKA_WINDOW_S,
                   hop_s: float = TROIKA_HOP_S) -> tuple[np.ndarray, np.ndarray]:
    """Cut a recording into the windows its reference heart rates describe.

    Returns (windows, bpm) with windows shaped (n_windows, n_channels, n_samples).

    Consecutive windows overlap by 75%. That is the dataset's own convention and
    we keep it, but it has a consequence worth stating early: neighbouring
    windows are *not* independent samples. Any evaluation that splits them at
    random is scoring a model on windows that share six of their eight seconds
    with the training set. Q4 measures exactly how much that flatters a model.
    """
    win = int(round(window_s * rec.fs))
    hop = int(round(hop_s * rec.fs))
    n_ref = len(rec.ref_bpm) if rec.ref_bpm is not None else 0

    windows, bpms = [], []
    for i in range(n_ref):
        start = i * hop
        stop = start + win
        if stop > rec.ppg.shape[-1]:
            break
        windows.append(rec.ppg[:, start:stop])
        bpms.append(rec.ref_bpm[i])

    return np.asarray(windows), np.asarray(bpms)


# --------------------------------------------------------------------------
# Welltory
# --------------------------------------------------------------------------

def _cardiac_evidence(x: np.ndarray, fs: float) -> float:
    """How strongly a trace repeats at a plausible heart rate, in [0, 1].

    Used to choose between colour channels. Deliberately *not* variance: the
    noisiest channel usually has the largest variance, and picking it would be
    exactly backwards. Autocorrelation strength at a physiological lag asks the
    right question -- does this trace contain something beating? -- and is
    scale-free, so channels with different gains stay comparable.
    """
    from common.dsp import bandpass, dominant_period  # local import: avoids cycle

    if not np.isfinite(x).all() or x.std() <= 0:
        return 0.0
    _, strength = dominant_period(bandpass(x, fs), fs)
    return float(strength)


def load_welltory(data_dir: Path | None = None,
                  channel: str = "auto") -> list[Recording]:
    """Load the Welltory smartphone-camera recordings.

    Channel selection
    -----------------
    The obvious default is green: haemoglobin absorbs green strongly while the
    surrounding tissue does not, which is why wrist wearables use green LEDs.
    On this dataset that default is wrong for a third of the recordings. The
    traces come from whatever camera pipeline each participant's own Android
    phone exposed, and in 7 of 21 recordings the green channel is flat to
    three decimal places while red carries a clean pulse; in two others the
    reverse. Hard-coding a channel would silently discard those subjects, and
    discarding the subjects whose sensor behaved unusually is precisely the
    kind of quiet selection bias that makes a quality detector look better than
    it is.

    So `channel="auto"` picks, per recording, the channel with the most
    cardiac-looking content by autocorrelation strength. This is not a
    convenience: choosing among wavelengths by evidence of pulsatility, rather
    than by assumption, is the same problem a multi-wavelength front end faces
    when one LED's return is swamped by ambient light or poor contact.

    Timing
    ------
    A phone camera samples at its frame rate, nominally 30 fps but not exactly
    uniform. We resample onto a regular 30 Hz grid using the per-frame
    timestamps the dataset provides. Assuming uniform spacing instead would
    inject timing jitter that a beat-interval analysis would then read as
    arrhythmia -- an artefact of our own making, blamed on the subject.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    base = root / "welltory"
    out: list[Recording] = []
    fs = 30.0

    for folder in sorted(base.glob("subject_*")):
        ppg_path, rr_path = folder / "PPG.csv", folder / "RR.txt"
        if not (ppg_path.exists() and rr_path.exists()):
            continue

        raw = np.genfromtxt(ppg_path, delimiter=",", names=True)
        t_ms = np.asarray(raw["time"], dtype=float)
        if len(t_ms) < 100:
            continue

        grid = np.arange(t_ms[0], t_ms[-1], 1000.0 / fs)
        resampled: dict[str, np.ndarray] = {}
        for name in ("R", "G", "B"):
            values = np.asarray(raw[name], dtype=float)
            good = np.isfinite(t_ms) & np.isfinite(values)
            if good.sum() < 100:
                continue
            resampled[name] = np.interp(grid, t_ms[good], values[good])

        if not resampled:
            continue

        if channel == "auto":
            chosen = max(resampled, key=lambda c: _cardiac_evidence(resampled[c], fs))
        else:
            if channel not in resampled:
                continue
            chosen = channel

        rr = np.fromstring(rr_path.read_text().strip(), sep=" ")

        out.append(
            Recording(
                subject=folder.name,
                record_id=folder.name,
                fs=fs,
                # A camera PPG is inverted relative to a reflectance PPG: more
                # blood in the fingertip means more absorption, so *less* light
                # reaches the sensor. We flip it so a systolic peak is a maximum
                # in both datasets and one detector serves both without a flag.
                ppg=-resampled[chosen][None, :],
                ref_rr_ms=rr if rr.size else None,
            )
        )
        out[-1].channel = chosen  # type: ignore[attr-defined]

    if not out:
        raise FileNotFoundError(
            f"no Welltory recordings under {base}; run download_data.py"
        )
    return out


def welltory_reference_bpm(rec: Recording) -> float:
    """Mean heart rate over a Welltory recording, from the Polar H10 intervals.

    The chest strap gives us beat-to-beat intervals but not their absolute
    alignment to the video timeline, so we can compare average rate over the
    recording but not beat-by-beat timing. The Q3 validation is built around
    that limitation rather than pretending it away.
    """
    if rec.ref_rr_ms is None or rec.ref_rr_ms.size == 0:
        return float("nan")
    # Discard physiologically impossible intervals before averaging: the strap
    # occasionally drops or doubles a beat, and one 2 s "interval" would drag
    # the mean down by several bpm.
    rr = rec.ref_rr_ms[(rec.ref_rr_ms > 300) & (rec.ref_rr_ms < 2000)]
    if rr.size == 0:
        return float("nan")
    return float(60_000.0 / np.mean(rr))
