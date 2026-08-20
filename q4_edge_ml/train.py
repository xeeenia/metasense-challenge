"""
Training loop and cross-validated evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q4_edge_ml.data import Dataset, standardise_target
from q4_edge_ml.nn import Adam, Network, build_model, huber_loss


@dataclass
class FoldResult:
    fold: int
    test_subject: str
    mae: float
    rmse: float
    n_test: int
    best_epoch: int
    state: list
    target_mean: float
    target_std: float


def train_one(data: Dataset, train_idx: np.ndarray, val_idx: np.ndarray,
              epochs: int = 120, batch_size: int = 64, lr: float = 3e-3,
              width: float = 1.0, patience: int = 25,
              seed: int = 0, verbose: bool = False) -> tuple[Network, dict]:
    """Train until the validation error stops improving, then rewind.

    Early stopping keeps the parameters from the best validation epoch rather
    than the last one. With ~1400 training windows and 7k parameters the model
    reaches its best validation error well before the training loss flattens,
    and taking the final epoch would report a model measurably worse than the
    one we actually found.
    """
    rng = np.random.default_rng(seed)
    x_train, y_train = data.x[train_idx], data.y[train_idx]
    x_val, y_val = data.x[val_idx], data.y[val_idx]

    mean, std = standardise_target(y_train)
    y_train_z = (y_train - mean) / std
    y_val_z = (y_val - mean) / std

    net = build_model(in_ch=data.x.shape[1], width=width, seed=seed)
    optimiser = Adam(net, lr=lr)

    best = {"val_mae": np.inf, "epoch": -1, "state": net.state()}
    history = []

    for epoch in range(epochs):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            pred = net.forward(x_train[batch], training=True)
            _, grad = huber_loss(pred, y_train_z[batch])
            net.backward(grad)
            optimiser.step()

        val_pred = net.forward(x_val, training=False).ravel() * std + mean
        val_mae = float(np.mean(np.abs(val_pred - y_val)))
        history.append(val_mae)

        if val_mae < best["val_mae"] - 1e-4:
            best = {"val_mae": val_mae, "epoch": epoch, "state": net.state()}
        elif epoch - best["epoch"] >= patience:
            break

        if verbose and epoch % 20 == 0:
            print(f"      epoch {epoch:3d}  val MAE {val_mae:6.2f}")

    net.load_state(best["state"])
    return net, {"best_epoch": best["epoch"], "val_mae": best["val_mae"],
                 "target_mean": mean, "target_std": std, "history": history}


def evaluate(net: Network, x: np.ndarray, y: np.ndarray,
             mean: float, std: float) -> dict[str, float]:
    pred = net.forward(x, training=False).ravel() * std + mean
    error = pred - y
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
        "n": int(len(y)),
    }


def cross_validate(data: Dataset, folds: list, epochs: int = 120,
                   width: float = 1.0, seed: int = 0,
                   label: str = "", verbose: bool = True) -> list[FoldResult]:
    """Train and evaluate one model per fold."""
    results: list[FoldResult] = []

    for i, (train_idx, val_idx, test_idx) in enumerate(folds):
        net, info = train_one(data, train_idx, val_idx, epochs=epochs,
                              width=width, seed=seed + i)
        metrics = evaluate(net, data.x[test_idx], data.y[test_idx],
                           info["target_mean"], info["target_std"])
        held_out = np.unique(data.subject[test_idx])
        results.append(FoldResult(
            fold=i,
            test_subject=held_out[0] if len(held_out) == 1 else "mixed",
            mae=metrics["mae"], rmse=metrics["rmse"], n_test=metrics["n"],
            best_epoch=info["best_epoch"], state=net.state(),
            target_mean=info["target_mean"], target_std=info["target_std"],
        ))
        if verbose:
            print(f"    {label}fold {i:2d}  test={results[-1].test_subject:6s} "
                  f"n={metrics['n']:4d}  MAE={metrics['mae']:6.2f}  "
                  f"RMSE={metrics['rmse']:6.2f}  (stopped at epoch "
                  f"{info['best_epoch']})")

    return results


def summarise(results: list[FoldResult]) -> dict[str, float]:
    """Aggregate folds, weighting each equally rather than by window count.

    Per-fold rather than pooled: pooling would let the subjects with longer
    recordings dominate, and the question a wearable has to answer is "how well
    does this work on a person", not "how well does this work on a window".
    The spread across folds is reported alongside the mean because it is the
    more useful number -- it is what tells you whether the device fails on
    somebody.
    """
    maes = np.array([r.mae for r in results])
    return {
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std()),
        "mae_median": float(np.median(maes)),
        "mae_worst": float(maes.max()),
        "mae_best": float(maes.min()),
        "rmse_mean": float(np.mean([r.rmse for r in results])),
        "n_folds": len(results),
    }
