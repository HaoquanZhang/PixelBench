"""Öztireli-Gross SSIM-optimal downscaling for integer scale factors.

This is Algorithm 1 from "Perceptually Based Downscaling of Images"
(SIGGRAPH 2015), specialized to the paper's published default ``n_p = 4``
(2x2 output patches).

The paper defines the input as channel values in [0, 1]. Integer arrays are
therefore normalized by their dtype maximum; floating-point arrays must already
be in [0, 1]. RGB channels are processed independently, as stated in section
3.3. No transfer-function conversion is performed: an sRGB PNG is processed in
its stored, gamma-encoded RGB coordinates. The paper does not prescribe
linear-light conversion.

Only complete, non-overlapping s-by-s source blocks and complete 2x2 output
patches are used. Consequently, input dimensions must be exactly divisible by
the integer factor and each output dimension must be at least two. At output
boundaries, the one, two, or four complete patches containing a pixel are
averaged (the ``convFull(...)/N`` operation in Algorithm 1).

The mathematical solution is not range constrained, so this module deliberately
does not clip its floating-point result to [0, 1].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_EPSILON = 1e-6


def _as_unit_float(image: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must have shape (height, width) or (height, width, channels)")
    if array.ndim == 3 and array.shape[2] == 0:
        raise ValueError("image must have at least one channel")
    if array.dtype.kind in "ui":
        maximum = np.iinfo(array.dtype).max
        result = array.astype(np.float64) / maximum
    elif array.dtype.kind == "f":
        result = array.astype(np.float64, copy=False)
    else:
        raise TypeError("image must have an unsigned/signed integer or floating dtype")
    if not np.all(np.isfinite(result)):
        raise ValueError("image values must be finite")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("image values must lie in [0, 1] after integer normalization")
    return result


def ssim_downscale(
    image: ArrayLike,
    factor: int,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> NDArray[np.float64]:
    """Downscale an image using the paper's SSIM-optimal closed-form filter.

    Args:
        image: ``(H, W)`` or ``(H, W, C)`` values. Integer inputs are normalized
            by the dtype maximum; floating inputs must be in ``[0, 1]``.
        factor: Integer spatial reduction factor. Both dimensions must be
            divisible by it.
        epsilon: Published constant-patch threshold, default ``1e-6``.

    Returns:
        A deterministic float64 array with shape ``(H/factor, W/factor[, C])``.
        Values can lie slightly outside ``[0, 1]``, as allowed by the paper.
    """
    if isinstance(factor, (bool, np.bool_)) or not isinstance(
        factor, (int, np.integer)
    ):
        raise TypeError("factor must be an integer")
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be positive")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be a finite non-negative number")

    source = _as_unit_float(image)
    original_was_grayscale = source.ndim == 2
    if original_was_grayscale:
        source = source[..., None]

    height, width, channels = source.shape
    if height % factor or width % factor:
        raise ValueError("image dimensions must be exactly divisible by factor")
    output_height = height // factor
    output_width = width // factor
    if output_height < 2 or output_width < 2:
        raise ValueError("each output dimension must be at least 2 for 2x2 patches")

    # Algorithm 1, lines 2-3: box means of H and H^2, then subsampling.
    blocks = source.reshape(
        output_height, factor, output_width, factor, channels
    )
    low = blocks.mean(axis=(1, 3))
    low_second_moment = np.square(blocks).mean(axis=(1, 3))

    # Lines 4-8 for all complete 2x2 output patches.
    patch_low = np.stack(
        (low[:-1, :-1], low[:-1, 1:], low[1:, :-1], low[1:, 1:]),
        axis=0,
    )
    patch_low2 = np.square(patch_low)
    patch_high2 = np.stack(
        (
            low_second_moment[:-1, :-1],
            low_second_moment[:-1, 1:],
            low_second_moment[1:, :-1],
            low_second_moment[1:, 1:],
        ),
        axis=0,
    )
    mean_high = patch_low.mean(axis=0)
    variance_low = patch_low2.mean(axis=0) - np.square(mean_high)
    variance_high = patch_high2.mean(axis=0) - np.square(mean_high)

    # Roundoff can make either analytically non-negative variance tiny-negative.
    variance_low = np.maximum(variance_low, 0.0)
    variance_high = np.maximum(variance_high, 0.0)
    ratio = np.zeros_like(mean_high)
    nonconstant = variance_low >= epsilon
    ratio[nonconstant] = np.sqrt(
        variance_high[nonconstant] / variance_low[nonconstant]
    )

    # Equation 8 / Algorithm 1, lines 9-13: average every complete patch
    # solution that contains each output pixel. This is explicit so that the
    # valid/full-convolution boundary convention cannot depend on a library.
    result = np.zeros_like(low)
    counts = np.zeros((output_height, output_width, 1), dtype=np.float64)
    patch_height, patch_width = mean_high.shape[:2]
    for offset_y in (0, 1):
        for offset_x in (0, 1):
            target = (
                slice(offset_y, offset_y + patch_height),
                slice(offset_x, offset_x + patch_width),
            )
            local_low = low[target]
            result[target] += mean_high + ratio * (local_low - mean_high)
            counts[target] += 1.0
    result /= counts

    if original_was_grayscale:
        return result[..., 0]
    return result


__all__ = ["DEFAULT_EPSILON", "ssim_downscale"]
