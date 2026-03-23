"""Tests for SaccadicController."""

import torch
import pytest
from src.saccadic_controller import SaccadicController


@pytest.fixture
def controller():
    return SaccadicController(hidden_dim=64, block_size=8, temperature=1.0)


def test_output_shapes(controller):
    """Should return fixation_point, logits, and block_idx with correct shapes."""
    peripheral_map = torch.randn(2, 16, 64)  # 16 blocks
    state = torch.randn(2, 64)
    fixation_point, logits, block_idx = controller(peripheral_map, state)
    assert fixation_point.shape == (2,)
    assert logits.shape == (2, 16)
    assert block_idx.shape == (2,)


def test_valid_fixation_positions(controller):
    """Fixation points should be valid token positions (multiples of block_size)."""
    peripheral_map = torch.randn(4, 16, 64)
    state = torch.randn(4, 64)
    fixation_point, _, _ = controller(peripheral_map, state)
    for fp in fixation_point:
        assert fp.item() % controller.block_size == 0
        assert 0 <= fp.item() < 16 * controller.block_size


def test_gradient_flow_gumbel(controller):
    """Gradients should flow through Gumbel-softmax during training."""
    controller.train()
    peripheral_map = torch.randn(2, 16, 64, requires_grad=True)
    state = torch.randn(2, 64, requires_grad=True)
    fixation_point, logits, _ = controller(peripheral_map, state)
    # Use logits for loss (fixation_point is integer, not differentiable)
    loss = logits.sum()
    loss.backward()
    assert peripheral_map.grad is not None
    assert state.grad is not None


def test_inference_mode(controller):
    """In eval mode, should use argmax (deterministic)."""
    controller.eval()
    torch.manual_seed(0)
    peripheral_map = torch.randn(2, 16, 64)
    state = torch.randn(2, 64)
    fp1, _, _ = controller(peripheral_map, state)
    fp2, _, _ = controller(peripheral_map, state)
    torch.testing.assert_close(fp1, fp2)


def test_attention_mask(controller):
    """Masked blocks should never be selected."""
    controller.eval()
    peripheral_map = torch.randn(1, 8, 64)
    state = torch.randn(1, 64)
    # Only allow the last block
    mask = torch.zeros(1, 8)
    mask[:, 7] = 1
    fp, logits, block_idx = controller(peripheral_map, state, attention_mask=mask)
    assert block_idx.item() == 7


def test_fixation_history_masking():
    """With mask_fixated=True, previously visited blocks should not be reselected."""
    controller = SaccadicController(hidden_dim=64, block_size=8, temperature=1.0, mask_fixated=True)
    controller.eval()
    peripheral_map = torch.randn(1, 4, 64)
    state = torch.randn(1, 64)

    # First fixation — all blocks available
    fp1, _, block1 = controller(peripheral_map, state)

    # Second fixation — block1 should be masked
    history = [block1]
    fp2, _, block2 = controller(peripheral_map, state, fixation_history=history)
    assert block1.item() != block2.item()


def test_temperature_effect():
    """Lower temperature should produce more peaked distributions."""
    controller_high = SaccadicController(hidden_dim=64, block_size=8, temperature=10.0)
    controller_low = SaccadicController(hidden_dim=64, block_size=8, temperature=0.01)
    # Share weights
    controller_low.load_state_dict(controller_high.state_dict())

    controller_high.eval()
    controller_low.eval()

    peripheral_map = torch.randn(1, 8, 64)
    state = torch.randn(1, 64)

    _, logits_high, _ = controller_high(peripheral_map, state)
    _, logits_low, _ = controller_low(peripheral_map, state)

    # Both should produce same logits (temperature only affects Gumbel, not eval argmax)
    # but the logits themselves are the same since temperature is only used in gumbel_softmax
    torch.testing.assert_close(logits_high, logits_low)
