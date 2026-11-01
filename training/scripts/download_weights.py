"""Download ImageNet-pretrained ResNet weights into a local cache directory.

Strategy, in order of preference:

1. ModelScope Hub (``modelscope`` SDK) - preferred for China-based hosts.
   A curated list of candidate model ids is tried; the first snapshot that
   contains a loadable state dict wins.
2. Torchvision direct download - instantiating the model with its
   ``IMAGENET1K_V1`` weights downloads the file from ``download.pytorch.org``
   into the torch hub cache; the cached file is then copied to the output
   directory.

Usage:
    python training/scripts/download_weights.py --model resnet18
    python training/scripts/download_weights.py --model resnet50 --output-dir D:/weights

The downloaded file is verified with ``torch.load`` before success is
reported; exit code is 0 on success and 1 on failure.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import NamedTuple, Optional

WEIGHT_SUFFIXES = (".pth", ".pt", ".bin")
MIN_WEIGHT_BYTES = 1_000_000  # ignore small non-weight files in snapshots

MODEL_SPECS = {
    "resnet18": {"filename": "resnet18-f37072fd.pth"},
    "resnet50": {"filename": "resnet50-0676ba61.pth"},
}

# Best-effort ModelScope ids. These were probed against the ModelScope API on
# 2026-11-02 and currently return 404; they are kept so that community
# mirrors are picked up automatically if they appear. The torchvision direct
# download below is the effective fallback path.
MODELSCOPE_CANDIDATES = {
    "resnet18": ["AI-ModelScope/resnet18-imagenet", "tiansz/resnet18-imagenet"],
    "resnet50": ["AI-ModelScope/resnet50-imagenet", "tiansz/resnet50-imagenet"],
}


class DownloadResult(NamedTuple):
    path: Optional[Path]
    errors: list[str]


def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    default_output = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "pretrained"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"cache directory (default: {default_output})",
    )
    return parser.parse_args(argv)


def _is_loadable(path: Path) -> bool:
    """Return True if ``path`` can be torch.load-ed as a non-empty dict."""
    try:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=True)
        return isinstance(state, dict) and len(state) > 0
    except Exception:
        return False


def _pick_weight_file(directory: Path) -> Optional[Path]:
    """Return the largest weight-like file inside a snapshot, if any."""
    candidates = [
        entry
        for entry in directory.rglob("*")
        if entry.is_file()
        and entry.suffix.lower() in WEIGHT_SUFFIXES
        and entry.stat().st_size >= MIN_WEIGHT_BYTES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry.stat().st_size)


def _copy_weight(source: Path, output_dir: Path, model_name: str) -> Path:
    """Copy ``source`` to ``output_dir/<model>.pth`` unless already cached."""
    destination = output_dir / f"{model_name}.pth"
    if destination.is_file() and _is_loadable(destination):
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def download_via_modelscope(model_name: str, output_dir: Path) -> DownloadResult:
    """Try every curated ModelScope id for ``model_name``; first hit wins."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        return DownloadResult(None, [f"modelscope SDK is not installed: {exc}"])

    errors: list[str] = []
    for model_id in MODELSCOPE_CANDIDATES.get(model_name, []):
        try:
            snapshot_dir = Path(snapshot_download(model_id))
            weight = _pick_weight_file(snapshot_dir)
            if weight is None:
                errors.append(f"{model_id}: snapshot contains no weight file")
                continue
            return DownloadResult(_copy_weight(weight, output_dir, model_name), [])
        except Exception as exc:
            errors.append(f"{model_id}: {type(exc).__name__}: {exc}")
    return DownloadResult(None, errors)


def download_via_torchvision(model_name: str, output_dir: Path) -> DownloadResult:
    """Download via torchvision's own mechanism into the torch hub cache."""
    try:
        import torch
        import torchvision.models as models

        if model_name == "resnet18":
            builder, weights = models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1
        elif model_name == "resnet50":
            builder, weights = models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"unsupported model: {model_name}")

        builder(weights=weights)  # triggers the download into the hub cache
        cached = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / MODEL_SPECS[model_name]["filename"]
        )
        if not cached.is_file():
            return DownloadResult(
                None, [f"torchvision download finished but {cached} is missing"]
            )
        return DownloadResult(_copy_weight(cached, output_dir, model_name), [])
    except Exception as exc:
        return DownloadResult(
            None, [f"torchvision direct download failed: {type(exc).__name__}: {exc}"]
        )


def verify_weight(path: Path) -> bool:
    """Verify ``path`` can be torch.load-ed and contains tensors."""
    try:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[ERROR] torch.load failed for {path}: {exc}")
        return False
    tensor_count = sum(1 for value in state.values() if torch.is_tensor(value))
    print(f"[OK] torch.load verified {tensor_count} tensors in {path.name}")
    return tensor_count > 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cached = output_dir / f"{args.model}.pth"
    if cached.is_file() and _is_loadable(cached):
        print(f"[OK] weights already cached: {cached}")
        print(f"     size: {cached.stat().st_size / 1e6:.1f} MB")
        return 0

    result = None
    for strategy in (download_via_modelscope, download_via_torchvision):
        result = strategy(args.model, output_dir)
        for error in result.errors:
            print(f"[WARN] {error}")
        if result.path is not None:
            break
    if result is None or result.path is None:
        print(
            f"[ERROR] could not download '{args.model}' weights from "
            "ModelScope or torchvision"
        )
        return 1

    weight_path = result.path
    if not verify_weight(weight_path):
        return 1
    print(f"[OK] {args.model} weights saved to: {weight_path}")
    print(f"     size: {weight_path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
