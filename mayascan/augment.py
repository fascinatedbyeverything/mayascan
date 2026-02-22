"""Data augmentation for archaeological segmentation training.

Competition-grade augmentations based on ECML PKDD 2021 Maya Challenge
winning solutions. Includes geometric transforms, photometric augmentations,
and CutMix for improved generalization.
"""

from __future__ import annotations

import numpy as np


def random_rotate90(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random 90-degree rotation (k in {0, 1, 2, 3}).

    Parameters
    ----------
    image : np.ndarray
        (C, H, W) image array.
    mask : np.ndarray
        (H, W) mask array.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    tuple
        (rotated_image, rotated_mask)
    """
    rng = rng or np.random.default_rng()
    k = int(rng.integers(4))
    return np.rot90(image, k, axes=(1, 2)).copy(), np.rot90(mask, k).copy()


def random_flip(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random horizontal and/or vertical flip.

    Parameters
    ----------
    image : np.ndarray
        (C, H, W) image array.
    mask : np.ndarray
        (H, W) mask array.

    Returns
    -------
    tuple
        (flipped_image, flipped_mask)
    """
    rng = rng or np.random.default_rng()
    if rng.random() > 0.5:
        image = np.flip(image, axis=2).copy()
        mask = np.flip(mask, axis=1).copy()
    if rng.random() > 0.5:
        image = np.flip(image, axis=1).copy()
        mask = np.flip(mask, axis=0).copy()
    return image, mask


def random_brightness(
    image: np.ndarray,
    factor_range: tuple[float, float] = (0.8, 1.2),
    p: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random brightness/contrast adjustment.

    Parameters
    ----------
    image : np.ndarray
        (C, H, W) image array, values in [0, 1].
    factor_range : tuple
        (min_factor, max_factor) for brightness multiplier.
    p : float
        Probability of applying the augmentation.

    Returns
    -------
    np.ndarray
        Augmented image, clipped to [0, 1].
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p:
        return image
    factor = rng.uniform(*factor_range)
    return np.clip(image * factor, 0, 1).astype(image.dtype)


def random_noise(
    image: np.ndarray,
    sigma: float = 0.02,
    p: float = 0.3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add random Gaussian noise.

    Parameters
    ----------
    image : np.ndarray
        (C, H, W) image array, values in [0, 1].
    sigma : float
        Standard deviation of Gaussian noise.
    p : float
        Probability of applying noise.

    Returns
    -------
    np.ndarray
        Noisy image, clipped to [0, 1].
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p:
        return image
    noise = rng.normal(0, sigma, image.shape).astype(image.dtype)
    return np.clip(image + noise, 0, 1)


def random_channel_shuffle(
    image: np.ndarray,
    p: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly permute channels.

    SVF, openness, and slope channels are somewhat interchangeable
    under rotation, so shuffling provides valid augmentation.

    Parameters
    ----------
    image : np.ndarray
        (C, H, W) image array.
    p : float
        Probability of shuffling.

    Returns
    -------
    np.ndarray
        Channel-shuffled image.
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p:
        return image
    perm = rng.permutation(image.shape[0])
    return image[perm].copy()


def random_elastic(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 30.0,
    sigma: float = 4.0,
    p: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Elastic deformation (competition-grade augmentation).

    Applies smooth random displacement fields to both image and mask.
    Uses scipy for Gaussian smoothing of the displacement field.

    Parameters
    ----------
    image : np.ndarray
        (C, H, W) image array.
    mask : np.ndarray
        (H, W) mask array.
    alpha : float
        Displacement magnitude.
    sigma : float
        Smoothness of displacement field.
    p : float
        Probability of applying.

    Returns
    -------
    tuple
        (deformed_image, deformed_mask)
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p:
        return image, mask

    try:
        from scipy.ndimage import gaussian_filter, map_coordinates
    except ImportError:
        return image, mask

    h, w = mask.shape
    dx = gaussian_filter(rng.standard_normal((h, w)) * alpha, sigma)
    dy = gaussian_filter(rng.standard_normal((h, w)) * alpha, sigma)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords_y = np.clip(y + dy, 0, h - 1)
    coords_x = np.clip(x + dx, 0, w - 1)

    # Deform mask (nearest interpolation)
    deformed_mask = map_coordinates(
        mask, [coords_y, coords_x], order=0, mode="reflect"
    ).astype(mask.dtype)

    # Deform each image channel (bilinear)
    deformed_image = np.empty_like(image)
    for c in range(image.shape[0]):
        deformed_image[c] = map_coordinates(
            image[c], [coords_y, coords_x], order=1, mode="reflect"
        )

    return deformed_image, deformed_mask


def cutmix(
    image1: np.ndarray,
    mask1: np.ndarray,
    image2: np.ndarray,
    mask2: np.ndarray,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CutMix augmentation: paste a random crop from sample 2 into sample 1.

    Competition-winning technique that helps models learn local features
    regardless of surrounding context.

    Parameters
    ----------
    image1, mask1 : np.ndarray
        Primary sample (C, H, W) and (H, W).
    image2, mask2 : np.ndarray
        Secondary sample to cut from.
    alpha : float
        Beta distribution parameter controlling cut size.

    Returns
    -------
    tuple
        (mixed_image, mixed_mask)
    """
    rng = rng or np.random.default_rng()
    _, h, w = image1.shape

    # Random cut ratio from Beta distribution
    lam = rng.beta(alpha, alpha)
    cut_h = int(h * np.sqrt(1 - lam))
    cut_w = int(w * np.sqrt(1 - lam))

    # Random cut position
    cy = int(rng.integers(0, h))
    cx = int(rng.integers(0, w))

    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    mixed_image = image1.copy()
    mixed_mask = mask1.copy()
    mixed_image[:, y1:y2, x1:x2] = image2[:, y1:y2, x1:x2]
    mixed_mask[y1:y2, x1:x2] = mask2[y1:y2, x1:x2]

    return mixed_image, mixed_mask


def augment_sample(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
    use_elastic: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply full augmentation pipeline to an image-mask pair.

    Applies in order: rotate90, flip, elastic, brightness, noise,
    channel shuffle.

    Parameters
    ----------
    image : np.ndarray
        (C, H, W) image array, values in [0, 1].
    mask : np.ndarray
        (H, W) binary mask array.
    rng : np.random.Generator or None
        Random number generator for reproducibility.
    use_elastic : bool
        Whether to include elastic deformation.

    Returns
    -------
    tuple
        (augmented_image, augmented_mask)
    """
    rng = rng or np.random.default_rng()

    image, mask = random_rotate90(image, mask, rng)
    image, mask = random_flip(image, mask, rng)

    if use_elastic:
        image, mask = random_elastic(image, mask, rng=rng)

    image = random_brightness(image, rng=rng)
    image = random_noise(image, rng=rng)
    image = random_channel_shuffle(image, rng=rng)

    return image, mask
