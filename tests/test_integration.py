"""Integration tests for the full SaccadicAttention module."""

import torch
import pytest
from src.saccadic_attention import SaccadicAttention


@pytest.fixture
def saccadic():
    return SaccadicAttention(
        hidden_dim=64,
        num_heads=4,
        num_saccades=3,
        window_size=16,
        block_size=8,
        gumbel_temperature=1.0,
    )


def test_output_shape(saccadic):
    """Output should match input shape (batch, seq_len, hidden_dim)."""
    x = torch.randn(2, 64, 64)
    output, info = saccadic(x)
    assert output.shape == (2, 64, 64)


def test_fixation_info(saccadic):
    """Info dict should contain fixation points and logits for each saccade."""
    x = torch.randn(2, 64, 64)
    _, info = saccadic(x)
    assert len(info['fixation_points']) == 3  # num_saccades
    assert len(info['fixation_logits']) == 3
    for fp in info['fixation_points']:
        assert fp.shape == (2,)
    for logits in info['fixation_logits']:
        assert logits.shape[0] == 2
        assert logits.shape[1] == 64 // 8  # num_blocks


def test_end_to_end_gradient_flow(saccadic):
    """Gradients should flow from output back through all components."""
    x = torch.randn(2, 64, 64, requires_grad=True)
    output, info = saccadic(x)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None
    assert (x.grad != 0).any()


def test_gradient_flow_through_logits(saccadic):
    """Logits should support gradient computation for entropy bonus."""
    x = torch.randn(2, 64, 64)
    saccadic.train()
    _, info = saccadic(x)
    # Entropy bonus on logits
    logits = torch.stack(info['fixation_logits'], dim=0)  # (num_saccades, batch, num_blocks)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    entropy.backward()
    # Controller params should have gradients
    assert saccadic.saccadic_controller.query_proj.weight.grad is not None


def test_with_attention_mask(saccadic):
    """Should work correctly with attention masking."""
    x = torch.randn(2, 64, 64)
    mask = torch.ones(2, 64)
    mask[0, 32:] = 0  # First example: only first 32 tokens are real
    output, info = saccadic(x, attention_mask=mask)
    assert output.shape == (2, 64, 64)
    assert not torch.isnan(output).any()


def test_non_divisible_sequence_length():
    """Works when seq_len is not divisible by block_size."""
    saccadic = SaccadicAttention(
        hidden_dim=64, num_heads=4, num_saccades=2,
        window_size=8, block_size=8,
    )
    x = torch.randn(1, 50, 64)  # 50 not divisible by 8
    output, info = saccadic(x)
    assert output.shape == (1, 50, 64)


def test_eval_mode_deterministic(saccadic):
    """In eval mode, outputs should be deterministic."""
    saccadic.eval()
    x = torch.randn(1, 64, 64)
    with torch.no_grad():
        out1, info1 = saccadic(x)
        out2, info2 = saccadic(x)
    torch.testing.assert_close(out1, out2)
    for fp1, fp2 in zip(info1['fixation_points'], info2['fixation_points']):
        torch.testing.assert_close(fp1, fp2)


def test_different_num_saccades():
    """Module should work with different numbers of saccades."""
    for n in [1, 5, 10]:
        saccadic = SaccadicAttention(
            hidden_dim=64, num_heads=4, num_saccades=n,
            window_size=8, block_size=8,
        )
        x = torch.randn(1, 32, 64)
        output, info = saccadic(x)
        assert output.shape == (1, 32, 64)
        assert len(info['fixation_points']) == n
