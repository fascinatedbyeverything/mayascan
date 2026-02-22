"""Configuration constants and defaults for MayaScan.

Centralizes all tunable parameters so they can be adjusted in one place
or overridden via environment variables.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------

#: Default tile size for inference (pixels)
TILE_SIZE: int = 480

#: Default overlap fraction between tiles
TILE_OVERLAP: float = 0.5

#: Default confidence threshold
CONFIDENCE_THRESHOLD: float = 0.5

#: Minimum blob size after post-processing (pixels)
MIN_BLOB_SIZE: int = 50

#: Default DEM resolution (metres per pixel)
DEFAULT_RESOLUTION: float = 0.5

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

#: Default v2 architecture
V2_ARCH: str = "deeplabv3plus"

#: Default v2 encoder
V2_ENCODER: str = "resnet101"

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

#: Mapping from class ID to human-readable name
CLASS_NAMES: dict[int, str] = {
    0: "background",
    1: "building",
    2: "platform",
    3: "aguada",
}

#: v2 per-class models (excludes background)
V2_CLASSES: dict[int, str] = {
    1: "building",
    2: "platform",
    3: "aguada",
}

#: Class colours (RGBA)
CLASS_COLORS: dict[int, tuple[int, int, int, int]] = {
    0: (0, 0, 0, 0),           # background — transparent
    1: (255, 60, 60, 180),     # building — red
    2: (60, 200, 60, 180),     # platform — green
    3: (50, 120, 255, 180),    # aguada — blue
}

# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

#: Default HuggingFace model repository
HF_REPO_ID: str = os.environ.get("MAYASCAN_HF_REPO", "fascinated23/mayascan")

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------

#: Default learning rate
LEARNING_RATE: float = 3e-4

#: Default batch size
BATCH_SIZE: int = 4

#: Default number of training epochs
EPOCHS: int = 80

#: Focal loss gamma
FOCAL_GAMMA: float = 2.0

#: Focal loss alpha
FOCAL_ALPHA: float = 0.75
