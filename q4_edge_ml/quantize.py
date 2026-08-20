"""
Post-training int8 quantisation, and a genuinely integer-only inference engine.

What "integer-only" has to mean
-------------------------------
It is easy to write something that stores int8 weights, converts them back to
float to multiply, and calls itself quantised. That model would not run any
faster on a Cortex-M without an FPU, and would not answer Q4's question. So the
forward pass in `IntegerNetwork` uses no floating point at all after the input
is quantised: weights and activations are int8, accumulators are int32, and the
rescaling between layers is a fixed-point multiply and an arithmetic shift --
the same construction TensorFlow Lite emits and CMSIS-NN executes.

The scheme, and why each piece is the way it is
-----------------------------------------------
Affine quantisation maps a real value to an integer by ``real = S * (q - Z)``.

* Weights: symmetric (Z = 0), per output channel. Symmetric because a zero
  point on the weights would introduce a cross term into every accumulation
  that has to be computed and subtracted at inference; forcing Z = 0 makes it
  vanish. Per channel because output channels of a trained conv layer routinely
  differ in magnitude by an order of magnitude, and one shared scale would
  quantise the small channels into a handful of distinct levels. Per-channel
  weight scales cost one int32 multiplier per channel and are what makes 8-bit
  post-training quantisation viable at all.

* Activations: asymmetric (Z != 0), per tensor. Asymmetric because a ReLU
  output is one-sided -- forcing it symmetric would throw away half the
  available codes on values that cannot occur. Per tensor rather than per
  channel because a per-channel activation scale would have to be recomputed at
  runtime, which is exactly the floating-point work we are trying to remove.

* Ranges come from a calibration pass over training data only. Using test data
  to set quantisation ranges is a leak of the same family as a leaky split: it
  is a parameter of the deployed model fitted on data the deployed model will
  not have seen.

The residual honesty
--------------------
The input is quantised in float, and the final output is dequantised in float.
Both are unavoidable at the boundary and both are single scalar operations per
inference, not per multiply-accumulate. The ADC on a real device delivers
integers directly, so the input conversion would not exist there at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q4_edge_ml.nn import Conv1D, Dense, GlobalAvgPool, Network, ReLU


@dataclass
class QuantSpec:
    """An affine mapping between real values and int8 codes."""

    scale: float
    zero_point: int

    def quantise(self, x: np.ndarray) -> np.ndarray:
        q = np.round(x / self.scale) + self.zero_point
        return np.clip(q, -128, 127).astype(np.int8)

    def dequantise(self, q: np.ndarray) -> np.ndarray:
        return (q.astype(np.float64) - self.zero_point) * self.scale


def _activation_spec(values: np.ndarray, symmetric: bool = False) -> QuantSpec:
    """Choose scale and zero point covering the observed range of a tensor.

    We use the full observed min/max rather than a percentile. Clipping
    outliers usually helps image networks, where a saturated activation is
    harmless; here the tail of an activation distribution is often exactly the
    motion artefact or the unusually fast heartbeat we need the model to
    register, and clipping it would quietly change what the model does on the
    inputs that matter most. The cost of not clipping is a coarser scale, and
    we measure that cost rather than trading it away by assumption.
    """
    lo = float(np.min(values))
    hi = float(np.max(values))
    if symmetric:
        bound = max(abs(lo), abs(hi), 1e-12)
        return QuantSpec(scale=bound / 127.0, zero_point=0)

    # Always include zero in the range: padding, ReLU outputs and the zero
    # point itself all assume the real value 0 is exactly representable.
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-12:
        return QuantSpec(scale=1e-12, zero_point=0)

    scale = (hi - lo) / 255.0
    zero_point = int(np.clip(round(-128 - lo / scale), -128, 127))
    return QuantSpec(scale=scale, zero_point=zero_point)


def _quantise_weights(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric int8 weights. Returns (int8 weights, scales).

    127 rather than 128 as the positive bound, so the mapping is exactly
    symmetric and -128 never occurs. On some MCUs the int8 multiply of -128 by
    -128 is the one case that overflows an int16 intermediate; avoiding the
    code entirely is free and removes the question.
    """
    out_channels = w.shape[0]
    flat = w.reshape(out_channels, -1)
    scales = np.maximum(np.abs(flat).max(axis=1), 1e-12) / 127.0
    q = np.round(flat / scales[:, None])
    q = np.clip(q, -127, 127).astype(np.int8).reshape(w.shape)
    return q, scales


def _fixed_point_multiplier(real_multiplier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express a real multiplier M in (0, 1) as M0 * 2^-shift with M0 an int32.

    This is the step that removes the last floating-point operation from the
    inner loop. After an int32 accumulation we need to rescale by
    M = S_weight * S_input / S_output, a real number that is always small.
    Writing M = M0 * 2^-shift with M0 in [2^30, 2^31) lets the rescale become an
    int64 multiply followed by a rounding right shift -- two integer
    instructions, no FPU.

    This is exactly what `tflite::QuantizeMultiplier` does, and reproducing it
    here rather than describing it is what makes the latency estimate in
    budget.py an operation count rather than a guess.
    """
    m = np.asarray(real_multiplier, dtype=np.float64)
    shift = np.zeros(m.shape, dtype=np.int32)
    mantissa = m.copy()

    nonzero = mantissa > 0
    # Normalise into [0.5, 1) by repeated doubling/halving, tracked as a shift.
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = np.where(nonzero, np.ceil(np.log2(np.maximum(mantissa, 1e-300))), 0)
    mantissa = np.where(nonzero, mantissa / np.power(2.0, exponent), 0.0)
    shift = (-exponent).astype(np.int32)

    m0 = np.round(mantissa * (1 << 31)).astype(np.int64)
    # A mantissa of exactly 1.0 would overflow int32; renormalise that case.
    overflow = m0 >= (1 << 31)
    m0 = np.where(overflow, m0 >> 1, m0)
    shift = np.where(overflow, shift - 1, shift)
    return m0.astype(np.int64), shift.astype(np.int32)


def _requantise(acc: np.ndarray, m0: np.ndarray, shift: np.ndarray,
                zero_point: int) -> np.ndarray:
    """int32 accumulator -> int8 output, using only integer operations.

    Rounding is round-half-away-from-zero, implemented by adding half of the
    divisor before shifting. Truncating instead would introduce a systematic
    negative bias that compounds across layers -- small per layer, clearly
    visible by the output of a five-layer network.
    """
    acc = acc.astype(np.int64)
    # Multiply by the normalised mantissa, keeping the high 32 bits.
    product = (acc * m0 + (1 << 30)) >> 31

    total_shift = shift.astype(np.int64)
    rounding = np.where(total_shift > 0, np.int64(1) << (total_shift - 1), 0)
    shifted = np.where(total_shift > 0,
                       (product + rounding) >> np.maximum(total_shift, 0),
                       product << np.maximum(-total_shift, 0))

    return np.clip(shifted + zero_point, -128, 127).astype(np.int8)


# --------------------------------------------------------------------------

@dataclass
class QuantConv1D:
    w_q: np.ndarray            # int8 (C_out, C_in, K)
    b_q: np.ndarray            # int32, pre-scaled into accumulator units
    m0: np.ndarray             # int32 fixed-point multiplier per channel
    shift: np.ndarray
    stride: int
    input_spec: QuantSpec
    output_spec: QuantSpec
    relu: bool


@dataclass
class QuantDense:
    w_q: np.ndarray            # int8 (out, in)
    b_q: np.ndarray
    m0: np.ndarray
    shift: np.ndarray
    input_spec: QuantSpec
    output_spec: QuantSpec
    relu: bool


@dataclass
class QuantPool:
    input_spec: QuantSpec
    output_spec: QuantSpec


class IntegerNetwork:
    """An int8 model that runs its forward pass in integer arithmetic."""

    def __init__(self, layers: list, input_spec: QuantSpec,
                 target_mean: float, target_std: float):
        self.layers = layers
        self.input_spec = input_spec
        self.target_mean = target_mean
        self.target_std = target_std

    # -- inference ---------------------------------------------------------

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Full pipeline: quantise input, run in integers, dequantise output."""
        q = self.input_spec.quantise(x)
        for layer in self.layers:
            q = self._run_layer(layer, q)
        # Final layer is linear; recover the real value and undo target scaling.
        real = self.layers[-1].output_spec.dequantise(q).ravel()
        return real * self.target_std + self.target_mean

    def _run_layer(self, layer, q: np.ndarray) -> np.ndarray:
        if isinstance(layer, QuantConv1D):
            return self._conv(layer, q)
        if isinstance(layer, QuantDense):
            return self._dense(layer, q)
        if isinstance(layer, QuantPool):
            return self._pool(layer, q)
        raise TypeError(f"unknown quantised layer: {type(layer)}")

    @staticmethod
    def _conv(layer: QuantConv1D, q: np.ndarray) -> np.ndarray:
        n, c_in, length = q.shape
        c_out, _, kernel = layer.w_q.shape
        l_out = (length - kernel) // layer.stride + 1

        # Subtracting the input zero point here, in int32, is the whole reason
        # weights are kept symmetric: with a weight zero point there would be a
        # second cross term to correct as well.
        shifted = q.astype(np.int32) - layer.input_spec.zero_point

        # Gather the receptive fields, ordered (tap, in-channel) so the weight
        # matrix can be flattened to match with a single transpose.
        cols = np.empty((n, kernel * c_in, l_out), dtype=np.int32)
        for k in range(kernel):
            idx = np.arange(l_out) * layer.stride + k
            cols[:, k * c_in:(k + 1) * c_in, :] = shifted[:, :, idx]
        w_flat = layer.w_q.astype(np.int32).transpose(0, 2, 1).reshape(c_out, -1)

        # int32 accumulation: with int8 operands and at most 5*32 = 160 terms,
        # the worst-case magnitude is 160 * 127 * 255, four orders of magnitude
        # below int32's range, so no intermediate saturation is possible here.
        acc = np.einsum("ok,nkl->nol", w_flat, cols) + layer.b_q[None, :, None]
        out = _requantise(acc, layer.m0[None, :, None], layer.shift[None, :, None],
                          layer.output_spec.zero_point)
        if layer.relu:
            out = np.maximum(out, layer.output_spec.zero_point).astype(np.int8)
        return out

    @staticmethod
    def _dense(layer: QuantDense, q: np.ndarray) -> np.ndarray:
        shifted = q.astype(np.int32) - layer.input_spec.zero_point
        acc = shifted @ layer.w_q.astype(np.int32).T + layer.b_q[None, :]
        out = _requantise(acc, layer.m0[None, :], layer.shift[None, :],
                          layer.output_spec.zero_point)
        if layer.relu:
            out = np.maximum(out, layer.output_spec.zero_point).astype(np.int8)
        return out

    @staticmethod
    def _pool(layer: QuantPool, q: np.ndarray) -> np.ndarray:
        """Integer global average pool.

        Accumulate in int32 and divide with rounding. Input and output share a
        scale, so no rescale is needed -- averaging is the one operation that
        preserves the quantisation grid exactly.
        """
        acc = q.astype(np.int32).sum(axis=2)
        length = q.shape[2]
        averaged = (acc + np.sign(acc) * (length // 2)) // length
        return np.clip(averaged, -128, 127).astype(np.int8)

    # -- reporting ---------------------------------------------------------

    def size_bytes(self) -> dict[str, int]:
        """Storage the model itself occupies, broken down by what it is."""
        weights = biases = multipliers = 0
        for layer in self.layers:
            if isinstance(layer, (QuantConv1D, QuantDense)):
                weights += layer.w_q.size                 # int8: 1 byte each
                biases += layer.b_q.size * 4              # int32
                multipliers += layer.m0.size * 4 + layer.shift.size * 4
        return {"weights": weights, "biases": biases,
                "requant_params": multipliers,
                "total": weights + biases + multipliers}


def calibrate_and_quantise(net: Network, calibration_x: np.ndarray,
                           target_mean: float, target_std: float,
                           batch: int = 256) -> IntegerNetwork:
    """Observe activation ranges on calibration data, then build the int8 model.

    The calibration set must come from training data. It is a fitted parameter
    of the deployed artefact, and fitting it on test data would overstate the
    quantised model's accuracy for exactly the reason a leaky split does.
    """
    sample = calibration_x[:batch]

    # Record the real-valued range at every layer boundary.
    activations = [sample]
    x = sample
    for layer in net.layers:
        x = layer.forward(x, training=False)
        activations.append(x)

    input_spec = _activation_spec(activations[0])
    quantised: list = []
    current_spec = input_spec

    i = 0
    while i < len(net.layers):
        layer = net.layers[i]
        follows_relu = (i + 1 < len(net.layers)
                        and isinstance(net.layers[i + 1], ReLU))
        # The output range is measured after the ReLU when there is one, so the
        # scale covers what the next layer will actually receive rather than
        # wasting half its codes on values ReLU has already removed.
        output_values = activations[i + 2] if follows_relu else activations[i + 1]

        if isinstance(layer, Conv1D):
            out_spec = _activation_spec(output_values)
            w_q, w_scales = _quantise_weights(layer.w)
            acc_scale = w_scales * current_spec.scale
            b_q = np.round(layer.b / acc_scale).astype(np.int32)
            m0, shift = _fixed_point_multiplier(acc_scale / out_spec.scale)
            quantised.append(QuantConv1D(w_q, b_q, m0, shift, layer.stride,
                                         current_spec, out_spec, follows_relu))
            current_spec = out_spec
            i += 2 if follows_relu else 1

        elif isinstance(layer, Dense):
            out_spec = _activation_spec(output_values)
            w_q, w_scales = _quantise_weights(layer.w)
            acc_scale = w_scales * current_spec.scale
            b_q = np.round(layer.b / acc_scale).astype(np.int32)
            m0, shift = _fixed_point_multiplier(acc_scale / out_spec.scale)
            quantised.append(QuantDense(w_q, b_q, m0, shift,
                                        current_spec, out_spec, follows_relu))
            current_spec = out_spec
            i += 2 if follows_relu else 1

        elif isinstance(layer, GlobalAvgPool):
            quantised.append(QuantPool(current_spec, current_spec))
            i += 1

        elif isinstance(layer, ReLU):
            i += 1  # already folded into the preceding layer

        else:
            raise TypeError(f"cannot quantise {type(layer)}")

    return IntegerNetwork(quantised, input_spec, target_mean, target_std)


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------

def magnitude_prune(net: Network, sparsity: float) -> int:
    """Zero the smallest-magnitude weights, layer by layer. Returns count zeroed.

    Layer-wise rather than global, because a global threshold on this
    architecture removes almost the entire first convolution: its 240 weights
    are larger in magnitude than the 3872 in the third layer, but far fewer, so
    a global ranking is dominated by the big layer's distribution and the small
    layer gets cut to nothing. Layer-wise keeps each layer functional.

    A caveat stated plainly, because it changes what the numbers below mean:
    this is *unstructured* pruning, and unstructured sparsity does not by itself
    make anything smaller or faster on a Cortex-M. CMSIS-NN's kernels are dense;
    a zeroed weight is still stored and still multiplied. Realising the saving
    needs either a sparse format with its own index overhead -- which at these
    sizes can cost more than it saves -- or structured pruning that removes
    whole channels. What the experiment below measures is therefore the
    *accuracy headroom* pruning offers, which is the prerequisite question:
    there is no point engineering a sparse kernel if the accuracy is not there
    to begin with. This is reported in the README as a bound, not a saving.
    """
    zeroed = 0
    for layer in net.layers:
        if "w" not in layer.params:
            continue
        w = layer.params["w"]
        k = int(round(sparsity * w.size))
        if k <= 0:
            continue
        threshold = np.partition(np.abs(w).ravel(), k - 1)[k - 1]
        mask = np.abs(w) <= threshold
        w[mask] = 0.0
        zeroed += int(mask.sum())
    return zeroed
