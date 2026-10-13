"""Central, leakage-safe configuration for the computer-vision data pipeline.

Unlike a tabular ColumnTransformer, YOLO consumes an image tensor and a YOLO label
file. This module records split and preprocessing decisions in one importable place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    """Fixed split proportions and seed for reproducible dataset manifests."""

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 42

    def __post_init__(self) -> None:
        if round(self.train_fraction + self.validation_fraction + self.test_fraction, 8) != 1.0:
            raise ValueError("Train, validation, and test fractions must sum to 1.")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Image transformations; augmentation is training-only by design."""

    image_size: int = 640
    horizontal_flip_probability: float = 0.0  
    use_hsv_augmentation: bool = True
    use_scale_translation_augmentation: bool = True


SPLIT_CONFIG = SplitConfig()
PREPROCESSING_CONFIG = PreprocessingConfig()
