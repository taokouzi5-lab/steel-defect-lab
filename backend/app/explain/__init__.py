"""Explainability package: hand-written Grad-CAM heatmaps and overlays."""

from app.explain.gradcam import GradCAM, heatmap_to_base64_png, overlay_heatmap

__all__ = ["GradCAM", "overlay_heatmap", "heatmap_to_base64_png"]
