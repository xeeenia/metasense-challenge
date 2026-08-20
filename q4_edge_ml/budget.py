"""
Memory, latency and energy accounting for a Cortex-M target.

The brief asks for accounting that is "realistic rather than aspirational", so
every number below is either counted from the actual model structure or
derived from a stated assumption with the arithmetic shown. Where a figure
depends on something we cannot measure without the hardware, it is given as a
range with both ends justified, not as a single confident value.

Assumed target
--------------
Cortex-M4F at 80 MHz, 256 KB SRAM, 1 MB flash. This is the class of part a
wearable of this kind is built around -- an STM32L4, nRF52840 or similar -- and
it is the class the brief describes ("a few hundred KB of RAM and flash"). The
M4 has a single-cycle 32-bit MAC and the DSP extension (SMLAD, SXTB16), which
is what CMSIS-NN's int8 kernels are written against. It has an FPU, but a
single-precision one whose throughput is roughly one operation per cycle with
no SIMD, so float inference gets none of the 2-way packing int8 enjoys -- which
is the real argument for quantisation on this part, and not, as it turns out,
memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q4_edge_ml.nn import Conv1D, Dense, GlobalAvgPool, Network, ReLU

# --- Assumptions, all stated so they can be argued with -------------------

CLOCK_HZ = 80e6

# Multiply-accumulates per cycle for int8 CMSIS-NN kernels on an M4. The DSP
# extension packs two 16-bit MACs per SMLAD, and int8 operands are unpacked to
# 16-bit first, so 2.0 is the theoretical ceiling. Real CMSIS-NN convolution
# achieves well under that once address generation, the im2col copy and loop
# overhead are counted, so we report a range and use the pessimistic end for
# headline figures.
MACS_PER_CYCLE_INT8 = (1.0, 2.0)

# The FPU is scalar: one MAC per cycle at best, and in practice less for
# convolution because operands must be loaded individually rather than packed.
MACS_PER_CYCLE_FLOAT = (0.5, 1.0)

# TFLite Micro's interpreter plus the op resolver and kernels for the four
# operators used here. The lower bound is a hand-rolled CMSIS-NN call chain with
# no interpreter at all; the upper is the interpreter with a modest op set.
# Both are widely reported figures rather than measurements of our own.
RUNTIME_FLASH_BYTES = (12_000, 100_000)

# Rough energy figure for an M4 class part in active mode, used only to turn
# a duty cycle into something comparable with the rest of a wearable's budget.
ACTIVE_CURRENT_MA = 10.0
SUPPLY_V = 3.0


@dataclass
class LayerCost:
    name: str
    macs: int
    weight_bytes_int8: int
    weight_bytes_float32: int
    output_bytes_int8: int
    output_shape: tuple


def analyse(net: Network, input_shape: tuple[int, int]) -> list[LayerCost]:
    """Count operations and buffer sizes layer by layer.

    These are exact counts derived from the layer geometry, not estimates: the
    convolution's inner loop runs exactly `out_length * out_channels *
    in_channels * kernel` times, and that is what is reported.
    """
    channels, length = input_shape
    costs: list[LayerCost] = []

    for i, layer in enumerate(net.layers):
        if isinstance(layer, Conv1D):
            out_ch, in_ch, kernel = layer.w.shape
            out_len = (length - kernel) // layer.stride + 1
            macs = out_len * out_ch * in_ch * kernel
            weights = layer.w.size + layer.b.size
            costs.append(LayerCost(
                name=f"conv1d_{i}",
                macs=macs,
                weight_bytes_int8=layer.w.size + layer.b.size * 4 + out_ch * 8,
                weight_bytes_float32=weights * 4,
                output_bytes_int8=out_ch * out_len,
                output_shape=(out_ch, out_len),
            ))
            channels, length = out_ch, out_len

        elif isinstance(layer, GlobalAvgPool):
            costs.append(LayerCost(
                name=f"gap_{i}", macs=channels * length,
                weight_bytes_int8=0, weight_bytes_float32=0,
                output_bytes_int8=channels, output_shape=(channels,),
            ))
            length = 1

        elif isinstance(layer, Dense):
            out_dim, in_dim = layer.w.shape
            costs.append(LayerCost(
                name=f"dense_{i}", macs=out_dim * in_dim,
                weight_bytes_int8=layer.w.size + layer.b.size * 4 + out_dim * 8,
                weight_bytes_float32=(layer.w.size + layer.b.size) * 4,
                output_bytes_int8=out_dim, output_shape=(out_dim,),
            ))
            channels = out_dim

        elif isinstance(layer, ReLU):
            continue  # fused into the preceding layer's requantisation

    return costs


def summarise(costs: list[LayerCost], input_bytes: int,
              inference_period_s: float = 2.0) -> dict:
    """Roll the per-layer counts into a deployment budget."""
    total_macs = sum(c.macs for c in costs)
    flash_int8 = sum(c.weight_bytes_int8 for c in costs)
    flash_float = sum(c.weight_bytes_float32 for c in costs)

    # Peak activation RAM under the usual ping-pong scheme: only the current
    # layer's input and output need to be resident, so the requirement is the
    # largest adjacent pair, not the sum of everything.
    buffers = [input_bytes] + [c.output_bytes_int8 for c in costs]
    peak_activation = max(a + b for a, b in zip(buffers[:-1], buffers[1:]))

    # The im2col scratch CMSIS-NN wants for convolution: two columns of the
    # patch matrix, in int16.
    scratch = max((c.output_shape[0] for c in costs if c.name.startswith("conv")),
                  default=0)
    im2col_scratch = 2 * scratch * 2

    latency_ms = {
        "int8_optimistic": 1000 * total_macs / (MACS_PER_CYCLE_INT8[1] * CLOCK_HZ),
        "int8_pessimistic": 1000 * total_macs / (MACS_PER_CYCLE_INT8[0] * CLOCK_HZ),
        "float32_optimistic": 1000 * total_macs / (MACS_PER_CYCLE_FLOAT[1] * CLOCK_HZ),
        "float32_pessimistic": 1000 * total_macs / (MACS_PER_CYCLE_FLOAT[0] * CLOCK_HZ),
    }

    duty = latency_ms["int8_pessimistic"] / (inference_period_s * 1000)
    energy_uj = (ACTIVE_CURRENT_MA * 1e-3 * SUPPLY_V
                 * latency_ms["int8_pessimistic"] * 1e-3 * 1e6)

    return {
        "total_macs": int(total_macs),
        "flash_model_int8_bytes": int(flash_int8),
        "flash_model_float32_bytes": int(flash_float),
        "flash_with_runtime_bytes": [int(flash_int8 + RUNTIME_FLASH_BYTES[0]),
                                     int(flash_int8 + RUNTIME_FLASH_BYTES[1])],
        "ram_peak_activation_bytes": int(peak_activation),
        "ram_im2col_scratch_bytes": int(im2col_scratch),
        "ram_total_bytes": int(peak_activation + im2col_scratch),
        "latency_ms": {k: round(v, 3) for k, v in latency_ms.items()},
        "cpu_duty_cycle": round(duty, 5),
        "energy_per_inference_uj": round(energy_uj, 1),
        "assumptions": {
            "clock_hz": CLOCK_HZ,
            "macs_per_cycle_int8": MACS_PER_CYCLE_INT8,
            "macs_per_cycle_float32": MACS_PER_CYCLE_FLOAT,
            "runtime_flash_bytes": RUNTIME_FLASH_BYTES,
            "inference_period_s": inference_period_s,
            "active_current_ma": ACTIVE_CURRENT_MA,
            "supply_v": SUPPLY_V,
        },
    }


def format_table(costs: list[LayerCost]) -> str:
    """A per-layer breakdown, so the totals can be checked rather than believed."""
    lines = [
        f"{'layer':12s} {'output':>12s} {'MACs':>10s} {'int8 B':>9s} {'fp32 B':>9s}",
        "-" * 56,
    ]
    for c in costs:
        shape = "x".join(str(v) for v in c.output_shape)
        lines.append(f"{c.name:12s} {shape:>12s} {c.macs:10,d} "
                     f"{c.weight_bytes_int8:9,d} {c.weight_bytes_float32:9,d}")
    lines.append("-" * 56)
    lines.append(f"{'total':12s} {'':>12s} {sum(c.macs for c in costs):10,d} "
                 f"{sum(c.weight_bytes_int8 for c in costs):9,d} "
                 f"{sum(c.weight_bytes_float32 for c in costs):9,d}")
    return "\n".join(lines)


def verify_fixed_point(n_trials: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Check that the fixed-point multiplier really reproduces the real one.

    The requantisation step is where an integer inference engine most often
    goes quietly wrong: an off-by-one in the shift changes every output by a
    factor of two, which is obvious, but a rounding error in the mantissa
    biases outputs by a fraction of a step, which is not. This samples the
    range of multipliers the model actually produces and reports the worst
    relative error, so the claim "the integer path matches the real arithmetic"
    is measured rather than asserted.
    """
    from q4_edge_ml.quantize import _fixed_point_multiplier

    rng = np.random.default_rng(seed)
    # Multipliers in a trained model span roughly 1e-6 to 1e-1.
    multipliers = 10 ** rng.uniform(-6, -0.5, n_trials)
    m0, shift = _fixed_point_multiplier(multipliers)
    reconstructed = m0.astype(np.float64) / (1 << 31) * 2.0 ** (-shift.astype(np.float64))
    relative = np.abs(reconstructed - multipliers) / multipliers
    return {
        "max_relative_error": float(relative.max()),
        "mean_relative_error": float(relative.mean()),
        "n_trials": n_trials,
    }
