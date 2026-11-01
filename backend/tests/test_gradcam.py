"""TDD tests for the hand-written Grad-CAM implementation (T3.3).

Covers the hook lifecycle (register/remove pairing, no state leakage), the
normalized heatmap contract, closed-form numerical correctness on a
hand-computed 1-layer model, the overlay/PNG helpers, and a slow regression
against the real trained resnet18 checkpoint on a NEU-DET crazing image.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from app.explain.gradcam import GradCAM, heatmap_to_base64_png, overlay_heatmap

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SumOfConvModel(nn.Module):
    """1x1 conv (weight 2, no bias) on a 2x2 single-channel image, summed.

    Closed form for Grad-CAM on ``conv``: the activations are ``A = 2 * x``
    and the gradient of the scalar sum w.r.t. ``A`` is all ones, so the
    gradient-GAP weight is 1. Min-max normalization is affine-invariant, so
    the normalized heatmap equals the min-max normalized ``x`` itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).sum()


class ExplodingModel(nn.Module):
    """Model that raises in forward, for the hook-cleanup-on-error test."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")


class DisconnectedModel(nn.Module):
    """Model whose target conv never feeds the output (no backward gradient)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.conv(x)  # computed and discarded: no gradient path to it
        return x.mean()


# ---------------------------------------------------------------------------
# Heatmap contract
# ---------------------------------------------------------------------------


def test_heatmap_shape_matches_input_spatial_size(tiny_model, sample_input):
    cam = GradCAM(tiny_model, tiny_model.conv2)
    heatmap = cam.compute(sample_input)
    assert heatmap.shape == (8, 8)
    assert np.issubdtype(heatmap.dtype, np.floating)


def test_heatmap_values_lie_in_unit_interval(tiny_model, sample_input):
    heatmap = GradCAM(tiny_model, tiny_model.conv2).compute(sample_input)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0


def test_all_black_input_produces_zero_heatmap(bias_free_model):
    # Zero input -> zero activations at the (bias-free) target layer ->
    # the ReLU'd weighted sum is zero -> the normalized map stays all zeros.
    black = torch.zeros(1, 3, 8, 8)
    heatmap = GradCAM(bias_free_model, bias_free_model.conv2).compute(black)
    assert np.all(heatmap == 0.0)


def test_consecutive_computes_are_identical(tiny_model, sample_input):
    cam = GradCAM(tiny_model, tiny_model.conv2)
    first = cam.compute(sample_input)
    second = cam.compute(sample_input)
    assert np.array_equal(first, second)


def test_different_inputs_produce_different_heatmaps(tiny_model, sample_input):
    cam = GradCAM(tiny_model, tiny_model.conv2)
    other = sample_input * 0.5 + 0.25
    first = cam.compute(sample_input)
    second = cam.compute(other)
    assert not np.allclose(first, second)


def test_hand_computed_gradcam_matches_closed_form():
    model = SumOfConvModel()
    model.eval()
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    heatmap = GradCAM(model, model.conv).compute(x)
    # (x - min(x)) / (max(x) - min(x)) for x = [[1, 2], [3, 4]]
    expected = np.array([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]], dtype=np.float32)
    assert np.allclose(heatmap, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Hook lifecycle
# ---------------------------------------------------------------------------


def test_hooks_removed_after_compute(tiny_model, sample_input):
    layer = tiny_model.conv2
    cam = GradCAM(tiny_model, layer)
    assert len(layer._forward_hooks) == 0
    assert len(layer._backward_hooks) == 0
    cam.compute(sample_input)
    assert len(layer._forward_hooks) == 0
    assert len(layer._backward_hooks) == 0
    assert cam._activations is None
    assert cam._gradients is None
    assert cam._handles == []


def test_hooks_removed_when_forward_raises():
    model = ExplodingModel()
    cam = GradCAM(model, model.conv)
    with pytest.raises(RuntimeError, match="boom"):
        cam.compute(torch.zeros(1, 3, 4, 4))
    assert len(model.conv._forward_hooks) == 0
    assert len(model.conv._backward_hooks) == 0


def test_compute_raises_when_layer_has_no_gradient_path():
    model = DisconnectedModel()
    cam = GradCAM(model, model.conv)
    with pytest.raises(RuntimeError, match="gradient"):
        cam.compute(torch.rand(1, 3, 4, 4))
    assert len(model.conv._forward_hooks) == 0
    assert len(model.conv._backward_hooks) == 0


def test_compute_is_not_reentrant(tiny_model, sample_input):
    cam = GradCAM(tiny_model, tiny_model.conv2)
    # A second registration while hooks are already live must fail loudly
    # rather than silently stacking duplicate hooks.
    cam._register_hooks()
    try:
        with pytest.raises(RuntimeError, match="re-entrant"):
            cam._register_hooks()
    finally:
        cam._remove_hooks()
    assert len(tiny_model.conv2._forward_hooks) == 0
    assert len(tiny_model.conv2._backward_hooks) == 0


# ---------------------------------------------------------------------------
# Constructor / input validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_target_layer_not_in_model(tiny_model):
    stranger = nn.Conv2d(1, 1, kernel_size=1)
    with pytest.raises(ValueError, match="target_layer"):
        GradCAM(tiny_model, stranger)


def test_constructor_rejects_missing_layer_attribute(tiny_model):
    layer = getattr(tiny_model, "layer4", None)
    assert layer is None
    with pytest.raises(ValueError, match="target_layer"):
        GradCAM(tiny_model, layer)  # type: ignore[arg-type]


def test_constructor_rejects_non_module_model(tiny_model):
    with pytest.raises(ValueError, match="model"):
        GradCAM("not a module", tiny_model.conv2)  # type: ignore[arg-type]


def test_compute_rejects_non_4d_input(tiny_model):
    cam = GradCAM(tiny_model, tiny_model.conv2)
    with pytest.raises(ValueError, match="4D"):
        cam.compute(torch.rand(3, 8, 8))
    with pytest.raises(ValueError, match="Tensor"):
        cam.compute([1, 2, 3])  # type: ignore[arg-type]


def test_compute_rejects_integer_input(tiny_model):
    cam = GradCAM(tiny_model, tiny_model.conv2)
    with pytest.raises(ValueError, match="floating"):
        cam.compute(torch.zeros(1, 3, 8, 8, dtype=torch.long))


def test_compute_rejects_non_scalar_or_logits_output(tiny_model):
    cam = GradCAM(tiny_model, tiny_model.conv2)
    with pytest.raises(ValueError, match="logits"):
        cam.compute(torch.rand(2, 3, 8, 8))  # batch of 2 -> output [2, 6]


# ---------------------------------------------------------------------------
# Overlay and base64 helpers
# ---------------------------------------------------------------------------


def test_overlay_preserves_image_size():
    image = Image.new("RGB", (64, 32), (10, 20, 30))
    heatmap = np.random.default_rng(0).random((8, 16))
    out = overlay_heatmap(image, heatmap, alpha=0.5)
    assert isinstance(out, Image.Image)
    assert out.size == image.size
    assert out.mode == "RGB"


def test_overlay_alpha_zero_returns_original_pixels():
    image = Image.new("RGB", (40, 40), (11, 22, 33))
    out = overlay_heatmap(image, np.ones((5, 5)), alpha=0.0)
    assert np.array_equal(np.asarray(out), np.asarray(image))


def test_overlay_alpha_one_returns_pure_colormap():
    image = Image.new("RGB", (40, 40), (255, 255, 255))
    heatmap = np.zeros((10, 10))  # jet(0.0) == (0, 0, 128)
    out = overlay_heatmap(image, heatmap, alpha=1.0)
    arr = np.asarray(out)
    assert (arr[:, :, 0] == 0).all()
    assert (arr[:, :, 1] == 0).all()
    assert (arr[:, :, 2] == 128).all()


def test_overlay_midpoint_blend_is_exact():
    image = Image.new("RGB", (16, 16), (100, 100, 100))
    # A constant heatmap is defensively normalized to all zeros, so the
    # colormap is jet(0.0) == (0, 0, 128).
    heatmap = np.ones((4, 4))
    out = overlay_heatmap(image, heatmap, alpha=0.5)
    arr = np.asarray(out)
    assert (arr[:, :, 0] == 50).all()  # round(0.5*0 + 0.5*100)
    assert (arr[:, :, 1] == 50).all()
    assert (arr[:, :, 2] == 114).all()  # round(0.5*128 + 0.5*100)


def test_overlay_alpha_one_applies_jet_gradient():
    image = Image.new("RGB", (24, 24), (255, 255, 255))
    heatmap = np.linspace(0.0, 1.0, 6 * 6).reshape(6, 6)
    out = overlay_heatmap(image, heatmap, alpha=1.0)
    arr = np.asarray(out)
    # jet is blue at 0.0 and red at 1.0; corner pixels survive the bilinear
    # resize up to interpolation rounding.
    assert np.allclose(arr[0, 0], [0, 0, 128], atol=2)
    assert np.allclose(arr[-1, -1], [128, 0, 0], atol=2)


def test_overlay_rejects_bad_arguments():
    image = Image.new("RGB", (8, 8))
    heatmap = np.zeros((4, 4))
    with pytest.raises(TypeError, match="PIL"):
        overlay_heatmap("not an image", heatmap)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(image, heatmap, alpha=-0.1)
    with pytest.raises(ValueError, match="alpha"):
        overlay_heatmap(image, heatmap, alpha=1.5)
    with pytest.raises(ValueError, match="2D"):
        overlay_heatmap(image, np.zeros(4))


def test_heatmap_to_base64_png_roundtrip_matches_overlay():
    image = Image.new("RGB", (32, 24), (5, 6, 7))
    heatmap = np.linspace(0, 1, 8 * 8).reshape(8, 8)
    encoded = heatmap_to_base64_png(heatmap, image, alpha=0.4)
    assert isinstance(encoded, str)
    decoded = base64.b64decode(encoded)
    roundtrip = Image.open(io.BytesIO(decoded))
    assert roundtrip.format == "PNG"
    assert roundtrip.size == image.size
    expected = overlay_heatmap(image, heatmap, alpha=0.4)
    assert np.array_equal(np.asarray(roundtrip), np.asarray(expected))


# ---------------------------------------------------------------------------
# Slow regression: real resnet18 checkpoint on a real crazing image
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_resnet18_gradcam_on_crazing_image(data_junction, resnet18_checkpoint):
    import torchvision.models as models
    import torchvision.transforms as transforms

    image_path = data_junction / "crazing" / "crazing_1.jpg"
    if not image_path.is_file():
        pytest.skip(f"crazing sample image is missing: {image_path}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 6)
    checkpoint = torch.load(str(resnet18_checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        predicted = int(model(x).argmax(dim=1).item())
    assert predicted == 0, "trained model must predict crazing (class 0) on a crazing image"

    cam = GradCAM(model, model.layer4)
    heatmap = cam.compute(x)
    black_heatmap = cam.compute(torch.zeros_like(x))

    # Note: a real resnet keeps nonzero activations on an all-black input
    # (conv/batchnorm biases), so the black baseline is not all-zero here;
    # the zero-map property is covered by the bias-free unit test instead.
    # The real-image heatmap must be non-trivial and clearly different from
    # the black-input baseline.
    assert heatmap.shape == (224, 224)
    assert black_heatmap.shape == (224, 224)
    assert float(heatmap.max()) == 1.0
    assert float(heatmap.std()) > 0.02, "real-image heatmap must carry spatial structure"
    mean_abs_diff = float(np.abs(heatmap - black_heatmap).mean())
    assert mean_abs_diff > 0.05, "real-image heatmap must differ from the all-black baseline"

    # The crazing defect occupies the center of the NEU-DET image, so the
    # heatmap's center of mass must fall inside the central 60% box.
    yy, xx = np.mgrid[0:224, 0:224]
    center_y = float((heatmap * yy).sum() / heatmap.sum())
    center_x = float((heatmap * xx).sum() / heatmap.sum())
    assert 0.2 * 224 <= center_x <= 0.8 * 224
    assert 0.2 * 224 <= center_y <= 0.8 * 224

    assert len(model.layer4._forward_hooks) == 0
    assert len(model.layer4._backward_hooks) == 0
