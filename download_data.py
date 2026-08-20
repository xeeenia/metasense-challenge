"""
Fetch the two public datasets used in this submission.

Both are redistributed on GitHub under research-use terms, which is why we pull
them from there rather than from their original hosts: it makes this repository
reproducible with nothing but `git` and an internet connection, no registration
step and no click-through licence in the middle of a clean-machine clone.

    TROIKA  (IEEE Signal Processing Cup 2015)
        12 recordings from 12 subjects running on a treadmill.
        Per recording: 1 ECG channel, 2 wrist-worn reflectance PPG channels,
        3 accelerometer axes, all at 125 Hz, plus a reference heart rate per
        8 s window (2 s hop) derived from the simultaneous ECG.
        Zhang, Pi & Liu (2015), IEEE Trans. Biomed. Eng. 62(2):522-531.

    Welltory PPG
        21 recordings from 13 subjects, PPG captured through a smartphone
        camera (fingertip, transmission mode, ~30 Hz) with simultaneous
        RR intervals from a Polar H10 chest strap.
        Neshitov et al. (2021), Sensors 21(20):6798.

Usage:
    python download_data.py            # fetch both into ./data
    python download_data.py --check    # verify what is already present

Everything lands in ./data, which is git-ignored. Re-running skips files that
are already on disk, so an interrupted download can simply be repeated.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

TROIKA_BASE = (
    "https://raw.githubusercontent.com/ElliotY-ML/Heart_Rate_Estimation_PPG_Acc"
    "/master/data/datasets/troika/training_data"
)

# The 12 TROIKA recordings. Subject 09 is absent from the public release and
# subject 04 contributed two sessions; we treat "04" as one subject in every
# split so that no subject can straddle a train/test boundary (see q4 README).
TROIKA_RECORDINGS = [
    "01_TYPE01",
    "02_TYPE02",
    "03_TYPE02",
    "04_TYPE01",
    "04_TYPE02",
    "05_TYPE02",
    "06_TYPE02",
    "07_TYPE02",
    "08_TYPE02",
    "10_TYPE02",
    "11_TYPE02",
    "12_TYPE02",
]

WELLTORY_BASE = (
    "https://raw.githubusercontent.com/Welltory/welltory-ppg-dataset/master/data"
)
WELLTORY_SUBJECTS = [f"subject_{i:02d}" for i in range(1, 22)]


def _fetch(url: str, dest: Path, timeout: int = 60) -> str:
    """Download `url` to `dest` unless it is already there. Returns a status word."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            tmp.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {exc}") from exc
    # Rename only after a complete read, so an interrupted run never leaves a
    # truncated file that a later run would happily "skip".
    tmp.rename(dest)
    return "ok"


def download_troika(root: Path) -> None:
    out = root / "troika"
    print(f"TROIKA -> {out}")
    for rec in TROIKA_RECORDINGS:
        for prefix in ("DATA", "REF"):
            name = f"{prefix}_{rec}.mat"
            status = _fetch(f"{TROIKA_BASE}/{name}", out / name)
            print(f"  [{status:4s}] {name}")


def download_welltory(root: Path) -> None:
    out = root / "welltory"
    print(f"Welltory -> {out}")
    for subject in WELLTORY_SUBJECTS:
        for name in ("PPG.csv", "RR.txt"):
            status = _fetch(f"{WELLTORY_BASE}/{subject}/{name}", out / subject / name)
            print(f"  [{status:4s}] {subject}/{name}")


def check(root: Path) -> bool:
    """Report which files are present. Returns True if everything is there."""
    missing: list[str] = []

    for rec in TROIKA_RECORDINGS:
        for prefix in ("DATA", "REF"):
            path = root / "troika" / f"{prefix}_{rec}.mat"
            if not path.exists():
                missing.append(str(path.relative_to(root)))

    for subject in WELLTORY_SUBJECTS:
        for name in ("PPG.csv", "RR.txt"):
            path = root / "welltory" / subject / name
            if not path.exists():
                missing.append(str(path.relative_to(root)))

    n_troika = 2 * len(TROIKA_RECORDINGS)
    n_welltory = 2 * len(WELLTORY_SUBJECTS)
    print(f"data root: {root}")
    print(f"  expected {n_troika + n_welltory} files, missing {len(missing)}")
    for name in missing[:10]:
        print(f"    missing: {name}")
    if len(missing) > 10:
        print(f"    ... and {len(missing) - 10} more")
    return not missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="only report what is already downloaded"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="destination (default: ./data)"
    )
    args = parser.parse_args()

    if args.check:
        return 0 if check(args.data_dir) else 1

    args.data_dir.mkdir(parents=True, exist_ok=True)
    download_troika(args.data_dir)
    download_welltory(args.data_dir)
    print()
    ok = check(args.data_dir)
    print("\ndone." if ok else "\nsome files are missing; re-run to retry.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
