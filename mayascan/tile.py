"""Tile slicing and stitching for large raster processing.

Splits (C, H, W) arrays into fixed-size patches with configurable overlap,
and stitches them back together with averaging in overlapping regions.
"""

from __future__ import annotations

import numpy as np


def slice_tiles(
    image: np.ndarray,
    tile_size: int = 480,
    overlap: float = 0.0,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Slice a (C, H, W) image into (C, tile_size, tile_size) tiles.

    Parameters
    ----------
    image : np.ndarray
        Input array with shape (C, H, W).
    tile_size : int
        Side length of each square tile (default 480).
    overlap : float
        Fraction of overlap between adjacent tiles, in [0, 1).
        stride = int(tile_size * (1 - overlap)).

    Returns
    -------
    tiles : list[np.ndarray]
        List of (C, tile_size, tile_size) arrays.
    origins : list[tuple[int, int]]
        List of (row, col) pixel origins for each tile in the
        (possibly padded) image coordinate system.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3-D array (C, H, W), got shape {image.shape}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    C, H, W = image.shape
    stride = max(1, int(tile_size * (1 - overlap)))

    # Compute start positions along each axis.  We need enough positions
    # so that the last tile reaches or exceeds the image extent.
    def _starts(length: int) -> list[int]:
        positions: list[int] = []
        pos = 0
        while pos + tile_size <= length:
            positions.append(pos)
            pos += stride
        # If the image is larger than what we covered, add one more position.
        if not positions or positions[-1] + tile_size < length:
            positions.append(max(0, length - tile_size))
        return positions

    # Pad image so that the last tile in each axis fits exactly.
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)

    # Also ensure enough room for the grid of starts.
    row_starts = _starts(max(H, tile_size))
    col_starts = _starts(max(W, tile_size))

    need_h = max(row_starts[-1] + tile_size, H) if row_starts else tile_size
    need_w = max(col_starts[-1] + tile_size, W) if col_starts else tile_size
    pad_h = need_h - H
    pad_w = need_w - W

    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )

    # Re-derive starts on padded image to keep things consistent.
    _, H_pad, W_pad = image.shape
    row_starts = _starts(H_pad)
    col_starts = _starts(W_pad)

    tiles: list[np.ndarray] = []
    origins: list[tuple[int, int]] = []
    for r in row_starts:
        for c in col_starts:
            tiles.append(image[:, r : r + tile_size, c : c + tile_size].copy())
            origins.append((r, c))

    return tiles, origins


def stitch_tiles(
    tiles: list[np.ndarray],
    origins: list[tuple[int, int]],
    output_shape: tuple[int, int, int],
    overlap: float = 0.0,  # noqa: ARG001 – kept for API symmetry
) -> np.ndarray:
    """Stitch tiles back into a single (C, H, W) array.

    Overlapping regions are averaged.

    Parameters
    ----------
    tiles : list[np.ndarray]
        Each tile has shape (C, tile_h, tile_w).
    origins : list[tuple[int, int]]
        (row, col) origin of each tile.
    output_shape : tuple[int, int, int]
        Desired output shape (C, H, W).
    overlap : float
        Kept for API symmetry with *slice_tiles*; blending is always
        performed via averaging regardless of this value.

    Returns
    -------
    np.ndarray
        Stitched array of shape *output_shape*.
    """
    C, H, W = output_shape
    accum = np.zeros((C, H, W), dtype=np.float64)
    weight = np.zeros((C, H, W), dtype=np.float64)

    for tile, (r, c) in zip(tiles, origins):
        _, th, tw = tile.shape
        # Clip to output bounds (handles tiles that extend past the edge
        # due to padding during slicing).
        r_end = min(r + th, H)
        c_end = min(c + tw, W)
        tile_r = r_end - r
        tile_c = c_end - c
        accum[:, r:r_end, c:c_end] += tile[:, :tile_r, :tile_c]
        weight[:, r:r_end, c:c_end] += 1.0

    # Avoid division by zero for pixels not covered by any tile.
    weight = np.maximum(weight, 1.0)
    return (accum / weight).astype(tiles[0].dtype)
