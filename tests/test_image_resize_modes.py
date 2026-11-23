"""Regression tests for the Image Resize node's device and pad-mode branches.

Run with: python -m pytest tests/test_image_resize_modes.py
"""
import pytest
import torch

from AILab_ImageMaskTools import AILab_ImageResize

OUTPUT_MODES = ["stretch", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur"]


@pytest.fixture
def node():
    return AILab_ImageResize()


@pytest.fixture
def image_and_mask():
    image = torch.zeros((1, 100, 200, 3), dtype=torch.float32)
    image[:, :50, :100, :] = 1.0
    mask = torch.zeros((1, 100, 200), dtype=torch.float32)
    mask[:, :50, :100] = 1.0
    return image, mask


def resize(node, image, mask, output_mode, device="cpu"):
    return node.resize(
        image, 128, 128, 0.0, 1.0, "longest_side", 0, "bilinear",
        device, 1, output_mode, "center", "#FFFFFF", mask=mask,
    )


@pytest.mark.parametrize("output_mode", OUTPUT_MODES)
def test_cpu_all_output_modes_produce_target_size(node, image_and_mask, output_mode):
    image, mask = image_and_mask
    out, out_mask, width, height = resize(node, image, mask, output_mode)
    assert (width, height) == (128, 128)
    assert tuple(out.shape) == (1, 128, 128, 3)
    assert tuple(out_mask.shape) == (1, 128, 128)


@pytest.mark.parametrize("output_mode", OUTPUT_MODES)
def test_gpu_device_all_output_modes_produce_target_size(node, image_and_mask, output_mode):
    """`device="gpu"` must resolve through comfy.model_management, not an unbound name."""
    image, mask = image_and_mask
    out, out_mask, width, height = resize(node, image, mask, output_mode, device="gpu")
    assert (width, height) == (128, 128)
    assert tuple(out.shape) == (1, 128, 128, 3)


def test_pillarbox_blur_fills_bands_with_dimmed_blur(node):
    """Bands outside the fitted image are the source blurred and dimmed by 0.35."""
    image = torch.full((1, 100, 200, 3), 0.8)
    mask = torch.ones((1, 100, 200))

    out, _, _, _ = resize(node, image, mask, "pillarbox_blur")
    row_mean = out[0].mean(dim=(1, 2))

    # 200x100 fitted into 128x128 gives a 128x64 image, centred: 32 band rows each side.
    assert row_mean[:32].max().item() == pytest.approx(0.8 * 0.35, abs=1e-4)
    assert row_mean[-32:].max().item() == pytest.approx(0.8 * 0.35, abs=1e-4)
    assert row_mean[32:96].min().item() == pytest.approx(0.8, abs=1e-4)
