"""Tests for mayascan.augment — data augmentation utilities."""

import numpy as np
import pytest

from mayascan.augment import (
    augment_sample,
    cutmix,
    random_brightness,
    random_channel_shuffle,
    random_elastic,
    random_flip,
    random_noise,
    random_rotate90,
)


@pytest.fixture
def sample():
    """Create a deterministic image-mask pair for testing."""
    rng = np.random.default_rng(42)
    image = rng.random((3, 64, 64), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 20:40] = 1.0
    return image, mask


class TestRandomRotate90:
    def test_preserves_shape(self, sample):
        img, mask = sample
        rot_img, rot_mask = random_rotate90(img, mask, rng=np.random.default_rng(0))
        assert rot_img.shape == img.shape
        assert rot_mask.shape == mask.shape

    def test_preserves_pixel_count(self, sample):
        img, mask = sample
        rot_img, rot_mask = random_rotate90(img, mask, rng=np.random.default_rng(0))
        assert rot_mask.sum() == mask.sum()


class TestRandomFlip:
    def test_preserves_shape(self, sample):
        img, mask = sample
        f_img, f_mask = random_flip(img, mask, rng=np.random.default_rng(0))
        assert f_img.shape == img.shape
        assert f_mask.shape == mask.shape

    def test_preserves_pixel_count(self, sample):
        img, mask = sample
        f_img, f_mask = random_flip(img, mask, rng=np.random.default_rng(0))
        assert f_mask.sum() == mask.sum()


class TestRandomBrightness:
    def test_stays_in_range(self, sample):
        img, _ = sample
        bright = random_brightness(img, p=1.0, rng=np.random.default_rng(0))
        assert bright.min() >= 0.0
        assert bright.max() <= 1.0

    def test_no_op_when_p_zero(self, sample):
        img, _ = sample
        result = random_brightness(img, p=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result, img)


class TestRandomNoise:
    def test_stays_in_range(self, sample):
        img, _ = sample
        noisy = random_noise(img, p=1.0, rng=np.random.default_rng(0))
        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0

    def test_changes_image(self, sample):
        img, _ = sample
        noisy = random_noise(img, p=1.0, rng=np.random.default_rng(0))
        assert not np.array_equal(noisy, img)


class TestRandomChannelShuffle:
    def test_preserves_shape(self, sample):
        img, _ = sample
        shuffled = random_channel_shuffle(img, p=1.0, rng=np.random.default_rng(0))
        assert shuffled.shape == img.shape

    def test_same_channel_set(self, sample):
        img, _ = sample
        shuffled = random_channel_shuffle(img, p=1.0, rng=np.random.default_rng(0))
        # Channel sums should be a permutation
        orig_sums = sorted([img[c].sum() for c in range(3)])
        shuf_sums = sorted([shuffled[c].sum() for c in range(3)])
        np.testing.assert_allclose(orig_sums, shuf_sums, atol=1e-5)


class TestRandomElastic:
    def test_preserves_shape(self, sample):
        img, mask = sample
        e_img, e_mask = random_elastic(img, mask, p=1.0, rng=np.random.default_rng(0))
        assert e_img.shape == img.shape
        assert e_mask.shape == mask.shape

    def test_mask_still_binary(self, sample):
        img, mask = sample
        _, e_mask = random_elastic(img, mask, p=1.0, rng=np.random.default_rng(0))
        unique = set(np.unique(e_mask))
        assert unique <= {0.0, 1.0}


class TestCutMix:
    def test_preserves_shape(self, sample):
        img1, mask1 = sample
        rng = np.random.default_rng(99)
        img2 = rng.random((3, 64, 64), dtype=np.float32)
        mask2 = np.ones((64, 64), dtype=np.float32)
        mixed_img, mixed_mask = cutmix(img1, mask1, img2, mask2, rng=rng)
        assert mixed_img.shape == img1.shape
        assert mixed_mask.shape == mask1.shape

    def test_mixed_content(self, sample):
        img1, mask1 = sample
        rng = np.random.default_rng(99)
        img2 = np.ones((3, 64, 64), dtype=np.float32)
        mask2 = np.ones((64, 64), dtype=np.float32)
        mixed_img, mixed_mask = cutmix(img1, mask1, img2, mask2, rng=rng)
        # Mixed mask should have more positive pixels than original
        assert mixed_mask.sum() >= mask1.sum()


class TestAugmentSample:
    def test_full_pipeline(self, sample):
        img, mask = sample
        aug_img, aug_mask = augment_sample(img, mask, rng=np.random.default_rng(0))
        assert aug_img.shape == img.shape
        assert aug_mask.shape == mask.shape
        assert aug_img.min() >= 0.0
        assert aug_img.max() <= 1.0

    def test_without_elastic(self, sample):
        img, mask = sample
        aug_img, aug_mask = augment_sample(
            img, mask, rng=np.random.default_rng(0), use_elastic=False
        )
        assert aug_img.shape == img.shape
        assert aug_mask.shape == mask.shape

    def test_deterministic_with_seed(self, sample):
        img, mask = sample
        aug1_img, aug1_mask = augment_sample(img, mask, rng=np.random.default_rng(42))
        aug2_img, aug2_mask = augment_sample(img, mask, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(aug1_img, aug2_img)
        np.testing.assert_array_equal(aug1_mask, aug2_mask)
