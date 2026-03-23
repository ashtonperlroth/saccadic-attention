"""Tests for FovealProcessor."""

import torch
import pytest
from src.foveal_processor import FovealProcessor


@pytest.fixture
def processor():
    return FovealProcessor(hidden_dim=64, num_heads=4, window_size=16)


def test_output_shape(processor):
    """Output should be (batch, hidden_dim)."""
    x = torch.randn(2, 128, 64)
    state = torch.randn(2, 64)
    fixation = torch.tensor([64, 32])
    out = processor(x, fixation, state)
    assert out.shape == (2, 64)


def test_fixation_at_start(processor):
    """Window should clamp correctly when fixation is near the start."""
    x = torch.randn(1, 128, 64)
    state = torch.randn(1, 64)
    fixation = torch.tensor([0])
    out = processor(x, fixation, state)
    assert out.shape == (1, 64)
    assert not torch.isnan(out).any()


def test_fixation_at_end(processor):
    """Window should clamp correctly when fixation is near the end."""
    x = torch.randn(1, 128, 64)
    state = torch.randn(1, 64)
    fixation = torch.tensor([127])
    out = processor(x, fixation, state)
    assert out.shape == (1, 64)
    assert not torch.isnan(out).any()


def test_short_sequence(processor):
    """Works when sequence is shorter than window_size."""
    x = torch.randn(1, 8, 64)  # 8 < 16 (window_size)
    state = torch.randn(1, 64)
    fixation = torch.tensor([4])
    out = processor(x, fixation, state)
    assert out.shape == (1, 64)


def test_gradient_flow(processor):
    """Gradients should flow back through both x and state."""
    x = torch.randn(2, 64, 64, requires_grad=True)
    state = torch.randn(2, 64, requires_grad=True)
    fixation = torch.tensor([32, 16])
    out = processor(x, fixation, state)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert state.grad is not None
    assert (state.grad != 0).any()


def test_different_fixations_give_different_outputs(processor):
    """Different fixation points on the same input should produce different outputs."""
    torch.manual_seed(42)
    x = torch.randn(1, 128, 64)
    state = torch.randn(1, 64)
    out1 = processor(x, torch.tensor([10]), state)
    out2 = processor(x, torch.tensor([100]), state)
    # Outputs should differ since they attend to different windows
    assert not torch.allclose(out1, out2)
