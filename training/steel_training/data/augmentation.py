"""Augmentation pipelines for the steel surface defect classifier.

Two pipelines are provided:

- :func:`build_train_transform`: randomized augmentations (random resized
  crop 224, horizontal/vertical flips, +-10 degree rotation, color jitter)
  followed by ImageNet normalization.
- :func:`build_eval_transform`: deterministic resize to 224 (the 200x200
  NEU-DET images are upscaled, preserving the full defect region) followed
  by the same ImageNet normalization.

torchvision is imported inside the builder functions so that importing this
module stays cheap and does not hard-require torchvision until a pipeline
is actually constructed.
"""

from __future__ import annotations

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_train_transform(
    crop_size: int = 224,
    scale: tuple[float, float] = (0.8, 1.0),
    rotation_degrees: float = 10.0,
    brightness_jitter: float = 0.1,
    contrast_jitter: float = 0.1,
) -> "torchvision.transforms.Compose":  # noqa: F821
    """Build the randomized augmentation pipeline used for training."""
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=crop_size, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ColorJitter(
                brightness=brightness_jitter, contrast=contrast_jitter
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(resize_size: int = 224) -> "torchvision.transforms.Compose":  # noqa: F821
    """Build the deterministic transform used for validation and testing."""
    import torchvision.transforms as transforms

    return transforms.Compose(
        [
            transforms.Resize(size=resize_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
