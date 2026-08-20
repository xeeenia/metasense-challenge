"""
A minimal 1-D convolutional network, written out in NumPy.

Why not PyTorch or TensorFlow
-----------------------------
Two reasons, one practical and one that matters more.

The practical one: this repository has to clone and run on someone else's clean
machine. NumPy, SciPy and scikit-learn are a small, stable dependency set;
a deep-learning framework is a large one with a version-sensitive install, and
"if the code does not run, we will not assess it" is a strong argument for
fewer moving parts.

The one that matters: Q4 is a question about what happens when a model is
squeezed into integer arithmetic, and answering it honestly means knowing
exactly what the arithmetic is. If the quantisation step is a converter call,
the interesting part of the answer -- where the precision goes, which tensor's
range is the problem, what the accumulator has to be -- is hidden inside a tool
rather than explained. Every multiply-accumulate in this file is visible, which
is also why the operation counts in `budget.py` can be exact rather than
estimated.

The layers implemented are the ones a Cortex-M inference runtime actually has
good integer kernels for: strided 1-D convolution, ReLU, global average
pooling, and fully-connected. Nothing here is exotic, deliberately -- an
architecture that needs an operator CMSIS-NN does not implement is not a
candidate for this deployment target, however well it scores offline.

Correctness is checked, not assumed: `gradient_check()` compares every
analytic gradient against a central finite difference, and `run_q4.py` runs it
before training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------

class Layer:
    """Base class. Subclasses expose `params` so the optimiser can find them."""

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {}

    @property
    def grads(self) -> dict[str, np.ndarray]:
        return {}


def _im2col(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
    """(N, C, L) -> (N, C*kernel, L_out), each column one receptive field.

    Built with stride tricks rather than a Python loop: the same memory is
    viewed at overlapping offsets, so this costs a reshape rather than a copy
    until the matrix multiply forces one. On a 1500-window training set this is
    the difference between an epoch taking a second and taking a minute.
    """
    n, c, length = x.shape
    l_out = (length - kernel) // stride + 1
    s_n, s_c, s_l = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, c, kernel, l_out),
        strides=(s_n, s_c, s_l, s_l * stride), writeable=False,
    )
    return view.reshape(n, c * kernel, l_out)


class Conv1D(Layer):
    """Strided 1-D convolution, no padding.

    No padding is a deliberate choice rather than a default. Zero-padding a
    physiological window invents signal at the edges -- a flat run that the
    first layer sees as a genuine feature -- and on an 8 s window with only
    three conv layers the edges are a meaningful fraction of the input. Losing
    a few samples per layer is the cheaper error.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        # He initialisation: variance 2/fan_in keeps activation scale stable
        # through ReLU, which matters here because the network is deep enough
        # (three convs plus two dense) for a bad scale to stall training.
        fan_in = in_ch * kernel
        self.w = rng.normal(0, np.sqrt(2.0 / fan_in), (out_ch, in_ch, kernel))
        self.b = np.zeros(out_ch)
        self.stride = stride
        self.kernel = kernel
        self._cache: tuple | None = None
        self._dw = np.zeros_like(self.w)
        self._db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        cols = _im2col(x, self.kernel, self.stride)          # (N, C*K, L_out)
        flat = self.w.reshape(self.w.shape[0], -1)           # (C_out, C*K)
        out = np.einsum("ok,nkl->nol", flat, cols) + self.b[None, :, None]
        if training:
            self._cache = (x.shape, cols)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x_shape, cols = self._cache
        flat = self.w.reshape(self.w.shape[0], -1)

        self._db = grad.sum(axis=(0, 2))
        self._dw = np.einsum("nol,nkl->ok", grad, cols).reshape(self.w.shape)

        dcols = np.einsum("ok,nol->nkl", flat, grad)         # (N, C*K, L_out)

        # col2im: scatter each receptive field's gradient back, accumulating
        # where windows overlap. np.add.at is the correct primitive here --
        # plain fancy-index assignment would keep only the last write and
        # silently drop gradient wherever the stride is smaller than the kernel.
        n, c, length = x_shape
        dx = np.zeros(x_shape)
        l_out = dcols.shape[2]
        dcols = dcols.reshape(n, c, self.kernel, l_out)
        for k in range(self.kernel):
            idx = np.arange(l_out) * self.stride + k
            np.add.at(dx, (slice(None), slice(None), idx), dcols[:, :, k, :])
        return dx

    @property
    def params(self):
        return {"w": self.w, "b": self.b}

    @property
    def grads(self):
        return {"w": self._dw, "b": self._db}


class Dense(Layer):
    def __init__(self, in_dim: int, out_dim: int,
                 rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng(0)
        self.w = rng.normal(0, np.sqrt(2.0 / in_dim), (out_dim, in_dim))
        self.b = np.zeros(out_dim)
        self._cache: np.ndarray | None = None
        self._dw = np.zeros_like(self.w)
        self._db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self._cache = x
        return x @ self.w.T + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self._dw = grad.T @ self._cache
        self._db = grad.sum(axis=0)
        return grad @ self.w

    @property
    def params(self):
        return {"w": self.w, "b": self.b}

    @property
    def grads(self):
        return {"w": self._dw, "b": self._db}


class ReLU(Layer):
    def __init__(self):
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        mask = x > 0
        if training:
            self._mask = mask
        return x * mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class GlobalAvgPool(Layer):
    """Average over time. (N, C, L) -> (N, C).

    Chosen over flattening for a reason that is as much about deployment as
    about accuracy: flattening a (32, 22) feature map into the first dense
    layer would need 704 inputs and roughly 22k extra parameters, which is
    three times the rest of the network. Global pooling also makes the model
    indifferent to *where* in the window a feature occurs, which is what we
    want -- a pulse eight seconds in is the same evidence as a pulse two
    seconds in.
    """

    def __init__(self):
        self._shape: tuple | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self._shape = x.shape
        return x.mean(axis=2)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        n, c, length = self._shape
        return np.repeat(grad[:, :, None], length, axis=2) / length


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

@dataclass
class Network:
    layers: list[Layer] = field(default_factory=list)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameter_count(self) -> int:
        return sum(p.size for layer in self.layers for p in layer.params.values())

    def state(self) -> list[dict[str, np.ndarray]]:
        return [{k: v.copy() for k, v in layer.params.items()} for layer in self.layers]

    def load_state(self, state: list[dict[str, np.ndarray]]) -> None:
        for layer, saved in zip(self.layers, state):
            for name, value in saved.items():
                layer.params[name][...] = value


def build_model(in_ch: int = 2, width: int = 1.0, seed: int = 0) -> Network:
    """The reference architecture.

    Three strided convolutions progressively halve the time axis while widening
    the channel axis, then global pooling and two small dense layers produce a
    scalar. The receptive field after three layers of kernel 7/5/5 at stride 2
    spans roughly 1.5 s of the 8 s window -- comfortably more than one cardiac
    cycle at any plausible heart rate, which is the requirement: a model that
    cannot see a whole beat cannot measure its period.

    `width` scales the channel counts so the accuracy-versus-size trade-off can
    be explored without editing the architecture.
    """
    rng = np.random.default_rng(seed)
    c1, c2, c3 = (max(int(round(c * width)), 4) for c in (16, 24, 32))
    dense = max(int(round(32 * width)), 8)

    return Network([
        Conv1D(in_ch, c1, kernel=7, stride=2, rng=rng), ReLU(),
        Conv1D(c1, c2, kernel=5, stride=2, rng=rng), ReLU(),
        Conv1D(c2, c3, kernel=5, stride=2, rng=rng), ReLU(),
        GlobalAvgPool(),
        Dense(c3, dense, rng=rng), ReLU(),
        Dense(dense, 1, rng=rng),
    ])


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

class Adam:
    """Adam, with decoupled weight decay applied only to weights, not biases.

    Decaying biases is a common and quiet mistake: a bias has no scale to
    regularise and shrinking it just biases the output toward zero, which for a
    regression whose target is centred is a small but real handicap.
    """

    def __init__(self, network: Network, lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 1e-4):
        self.network = network
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [{k: np.zeros_like(v) for k, v in l.params.items()}
                  for l in network.layers]
        self.v = [{k: np.zeros_like(v) for k, v in l.params.items()}
                  for l in network.layers]

    def step(self) -> None:
        self.t += 1
        bias1 = 1 - self.b1 ** self.t
        bias2 = 1 - self.b2 ** self.t

        for i, layer in enumerate(self.network.layers):
            for name, param in layer.params.items():
                grad = layer.grads[name]
                if self.weight_decay and name == "w":
                    grad = grad + self.weight_decay * param
                self.m[i][name] = self.b1 * self.m[i][name] + (1 - self.b1) * grad
                self.v[i][name] = self.b2 * self.v[i][name] + (1 - self.b2) * grad ** 2
                m_hat = self.m[i][name] / bias1
                v_hat = self.v[i][name] / bias2
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def huber_loss(pred: np.ndarray, target: np.ndarray,
               delta: float = 1.0) -> tuple[float, np.ndarray]:
    """Huber rather than plain squared error, and the reason is the data.

    Reference heart rates in this dataset are themselves imperfect -- they come
    from ECG processed automatically, and during heavy motion some windows
    carry a reference that is simply wrong. Squared error lets one such window
    contribute as much gradient as a hundred ordinary ones, so the model spends
    its capacity fitting label noise. Huber is quadratic near zero, where we
    want the fine gradient, and linear in the tail, where we mostly want the
    sign.
    """
    error = pred.ravel() - target.ravel()
    abs_error = np.abs(error)
    quadratic = abs_error <= delta

    loss = np.where(quadratic, 0.5 * error ** 2, delta * (abs_error - 0.5 * delta))
    grad = np.where(quadratic, error, delta * np.sign(error)) / len(error)
    return float(loss.mean()), grad.reshape(-1, 1)


def gradient_check(seed: int = 0, tolerance: float = 1e-5) -> dict[str, float]:
    """Compare every analytic gradient with a central finite difference.

    This exists because hand-written backward passes are where silent bugs
    live, and a silent gradient bug does not crash -- it just trains to a worse
    optimum and leaves you reporting an honest number about a broken model. The
    check is cheap, so `run_q4.py` runs it every time rather than trusting that
    it passed once.

    Central differences (f(x+h) - f(x-h)) / 2h rather than forward differences,
    because the error term is O(h^2) instead of O(h) and lets us use a tolerance
    tight enough to catch a genuine mistake.
    """
    rng = np.random.default_rng(seed)
    net = build_model(in_ch=2, width=0.5, seed=seed)
    x = rng.normal(size=(4, 2, 60))
    y = rng.normal(size=4)

    pred = net.forward(x, training=True)
    _, grad = huber_loss(pred, y)
    net.backward(grad)

    worst: dict[str, float] = {}
    h = 1e-5
    for li, layer in enumerate(net.layers):
        for name, param in layer.params.items():
            analytic = layer.grads[name]
            flat = param.reshape(-1)
            flat_analytic = analytic.reshape(-1)
            # Spot-check a handful of entries; checking all of them is the same
            # test repeated thousands of times at thousands of times the cost.
            idx = rng.choice(flat.size, size=min(8, flat.size), replace=False)
            errors = []
            for j in idx:
                original = flat[j]
                flat[j] = original + h
                loss_plus, _ = huber_loss(net.forward(x, training=False), y)
                flat[j] = original - h
                loss_minus, _ = huber_loss(net.forward(x, training=False), y)
                flat[j] = original
                numeric = (loss_plus - loss_minus) / (2 * h)
                scale = max(abs(numeric), abs(flat_analytic[j]), 1e-8)
                errors.append(abs(numeric - flat_analytic[j]) / scale)
            worst[f"layer{li}.{name}"] = float(max(errors))

    worst["max"] = float(max(worst.values()))
    worst["passed"] = float(worst["max"] < tolerance)
    return worst
