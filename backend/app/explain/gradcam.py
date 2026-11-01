"""Hand-written Grad-CAM (Gradient-weighted Class Activation Mapping).

Implements the Selvaraju et al. (2017) algorithm without the
``pytorch-grad-cam`` library:

1. register a forward hook on the target layer to capture activations and a
   full backward hook to capture gradients;
2. run one forward pass and backpropagate the predicted-class score;
3. gradient global-average-pooling (spatial dims) yields one weight per
   channel;
4. heatmap = ReLU(sum_c(weight_c * activations_c));
5. bilinear-upsample to the input resolution and min-max normalize to [0, 1].

Hook handles are removed in a ``finally`` block on every call (registration
and removal are paired), so no hook state leaks across calls. Hooks write
into instance attributes rather than closures to avoid reference cycles.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

__all__ = ["GradCAM", "overlay_heatmap", "heatmap_to_base64_png"]

# Approximated MATLAB "jet" colormap anchors: (position, (r, g, b)).
_JET_ANCHORS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (0, 0, 128)),
    (0.125, (0, 0, 255)),
    (0.375, (0, 255, 255)),
    (0.625, (255, 255, 0)),
    (0.875, (255, 0, 0)),
    (1.0, (128, 0, 0)),
)


def _jet(values: np.ndarray) -> np.ndarray:
    """Map an array of [0, 1] floats to an ``[..., 3]`` uint8 jet-RGB array."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    rgb = np.zeros((flat.size, 3), dtype=np.float64)
    for channel in range(3):
        for (start, start_color), (end, end_color) in zip(_JET_ANCHORS[:-1], _JET_ANCHORS[1:]):
            mask = (flat >= start) & (flat <= end)
            if mask.any():
                fraction = (flat[mask] - start) / (end - start)
                rgb[mask, channel] = start_color[channel] + fraction * (
                    end_color[channel] - start_color[channel]
                )
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8).reshape(values.shape + (3,))


class GradCAM:
    """Grad-CAM heatmap extractor for a single ``nn.Module`` target layer.

    Args:
        model: the network to explain. Its output must be a scalar tensor or
            a ``[1, num_classes]`` logits tensor for a single image.
        target_layer: the layer whose activations are mapped to the input.

    Raises:
        ValueError: if ``model`` is not an ``nn.Module`` or ``target_layer``
            is not a submodule of ``model``.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self._validate(model, target_layer)
        self._model = model
        self._target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _validate(model: nn.Module, target_layer: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise ValueError(f"model must be an nn.Module, got {type(model).__name__}")
        if not isinstance(target_layer, nn.Module):
            raise ValueError(
                f"target_layer must be an nn.Module, got {type(target_layer).__name__}"
            )
        if target_layer not in list(model.modules()):
            raise ValueError("target_layer is not a submodule of the given model")

    # -- hook plumbing ------------------------------------------------------

    def _forward_hook(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module, inputs
        self._activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        del module, grad_input
        gradient = grad_output[0]
        # A layer disconnected from the model output receives no gradient
        # (None); keep the sentinel so compute() can fail with a clear error.
        self._gradients = None if gradient is None else gradient.detach()

    def _register_hooks(self) -> None:
        if self._handles:
            raise RuntimeError(
                "GradCAM.compute() is re-entrant: hooks are already registered "
                "for an in-flight computation"
            )
        self._handles.append(self._target_layer.register_forward_hook(self._forward_hook))
        self._handles.append(self._target_layer.register_full_backward_hook(self._backward_hook))

    def _remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._activations = None
        self._gradients = None

    # -- core algorithm -----------------------------------------------------

    def compute(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute the normalized Grad-CAM heatmap for a single image.

        Args:
            input_tensor: a ``[1, channels, height, width]`` floating-point
                tensor on the same device as the model.

        Returns:
            A ``[height, width]`` float32 numpy array with values in [0, 1].

        Raises:
            ValueError: for invalid input shapes/dtypes or for model outputs
                that are neither a scalar nor ``[1, num_classes]`` logits.
            RuntimeError: if backpropagation produces no gradient at the
                target layer.
        """
        if not torch.is_tensor(input_tensor):
            raise ValueError(
                f"input_tensor must be a torch.Tensor, got {type(input_tensor).__name__}"
            )
        if input_tensor.ndim != 4:
            raise ValueError(
                f"input_tensor must be 4D [1, C, H, W], got shape {tuple(input_tensor.shape)}"
            )
        if not input_tensor.is_floating_point():
            raise ValueError(
                f"input_tensor must be a floating-point tensor, got {input_tensor.dtype}"
            )

        self._register_hooks()
        try:
            x = input_tensor.detach()
            x.requires_grad_(True)
            output = self._model(x)
            class_score = self._predicted_class_score(output)
            self._model.zero_grad(set_to_none=True)
            class_score.backward()
            if self._gradients is None:
                raise RuntimeError(
                    "backward pass produced no gradient at the target layer; "
                    "the layer may be disconnected from the model output"
                )
            weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # gradient GAP
            cam = (weights * self._activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam,
                size=(input_tensor.shape[2], input_tensor.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam.squeeze()
            cam = cam - cam.min()
            span = cam.max()
            if span > 0:
                cam = cam / span
        finally:
            self._remove_hooks()
        return cam.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _predicted_class_score(output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 0:
            return output
        if output.ndim == 2 and output.shape[0] == 1:
            return output[0, output.argmax(dim=1)[0]]
        raise ValueError(
            "model output must be a scalar or [1, num_classes] logits, "
            f"got shape {tuple(output.shape)}"
        )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def overlay_heatmap(image_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a jet-colored heatmap over an RGB PIL image.

    The heatmap is min-max normalized defensively (so any float array works,
    not only compute() output) and resized to the image size with bilinear
    interpolation. ``alpha=0`` returns the original pixels, ``alpha=1`` the
    pure colormap.

    Args:
        image_pil: the base image (any mode; converted to RGB).
        heatmap: 2D float array.
        alpha: blend weight of the colormap, in [0, 1].

    Returns:
        A new RGB image with the same size as ``image_pil``.
    """
    if not isinstance(image_pil, Image.Image):
        raise TypeError(f"image_pil must be a PIL Image, got {type(image_pil).__name__}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"heatmap must be a 2D array, got shape {values.shape}")

    low, high = float(values.min()), float(values.max())
    if high > low:
        normalized = (values - low) / (high - low)
    else:
        normalized = np.zeros_like(values)
    mask = Image.fromarray(_jet(normalized)).resize(
        image_pil.size, resample=Image.Resampling.BILINEAR
    )
    return Image.blend(image_pil.convert("RGB"), mask, alpha)


def heatmap_to_base64_png(heatmap: np.ndarray, image_pil: Image.Image, alpha: float = 0.5) -> str:
    """Return an alpha-blended heatmap overlay encoded as a base64 PNG string."""
    overlay = overlay_heatmap(image_pil, heatmap, alpha)
    buffer = io.BytesIO()
    overlay.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
