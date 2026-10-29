"""
CPU unit tests for BAP2.

No models are downloaded: the tests use a diffusers `Attention` layer and a
tiny UNet with the same structure as the Stable Diffusion 1.5 UNet.

Run from the repository root:
    pytest -q
"""

import pytest
import torch

diffusers = pytest.importorskip("diffusers")
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention

from src.att2 import attach_bap2
from src.bap2 import BAP2

HIDDEN = 320
CROSS = 768
TOKENS = 4
TEXT_LEN = 77


def make_attn():
    return Attention(query_dim=HIDDEN, cross_attention_dim=CROSS, heads=8, dim_head=HIDDEN // 8)


def make_inputs(batch=2, pixels=64):
    torch.manual_seed(0)
    hidden_states = torch.randn(batch, pixels, HIDDEN)
    text = torch.randn(batch, TEXT_LEN, CROSS)
    child = torch.randn(batch, TOKENS, CROSS)
    father = torch.randn(batch, TOKENS, CROSS)
    mother = torch.randn(batch, TOKENS, CROSS)
    return hidden_states, text, child, father, mother


def run(attn, proc, hidden_states, text, child, father, mother):
    encoder_hidden_states = torch.cat([text, child, father, mother], dim=1)
    return proc(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)


def make_proc(**kwargs):
    proc = BAP2(hidden_size=HIDDEN, cross_attention_dim=CROSS, num_tokens=TOKENS, **kwargs)
    # Same as copying the IP-Adapter weights: the three branches share weights.
    with torch.no_grad():
        for branch in ("father", "mother"):
            getattr(proc, f"to_k_{branch}").weight.copy_(proc.to_k_child.weight)
            getattr(proc, f"to_v_{branch}").weight.copy_(proc.to_v_child.weight)
    return proc


@torch.no_grad()
def test_output_shape_and_initial_alpha():
    attn, proc = make_attn(), make_proc()
    hidden_states, *conds = make_inputs()

    out = run(attn, proc, hidden_states, *conds)

    assert out.shape == hidden_states.shape
    assert proc.last_alpha.shape == (hidden_states.shape[0], proc.gate_groups)
    # Zero-initialized gate -> sigmoid(0) = 0.5 (equal father/mother mix).
    assert torch.allclose(proc.last_alpha, torch.full_like(proc.last_alpha, 0.5))


@torch.no_grad()
def test_parent_scale_zero_ignores_parents():
    attn, proc = make_attn(), make_proc(parent_scale=0.0)
    hidden_states, text, child, father, mother = make_inputs()

    out_a = run(attn, proc, hidden_states, text, child, father, mother)
    out_b = run(attn, proc, hidden_states, text, child, torch.randn_like(father), torch.randn_like(mother))

    assert torch.allclose(out_a, out_b, atol=1e-6)


@torch.no_grad()
def test_parents_change_output_when_parent_scale_positive():
    attn, proc = make_attn(), make_proc(parent_scale=0.1)
    hidden_states, text, child, father, mother = make_inputs()

    out_a = run(attn, proc, hidden_states, text, child, father, mother)
    out_b = run(attn, proc, hidden_states, text, child, torch.randn_like(father), mother)

    assert not torch.allclose(out_a, out_b, atol=1e-6)


@torch.no_grad()
def test_initial_gate_is_symmetric_between_parents():
    attn, proc = make_attn(), make_proc(parent_scale=0.1)
    hidden_states, text, child, father, mother = make_inputs()

    out_a = run(attn, proc, hidden_states, text, child, father, mother)
    out_b = run(attn, proc, hidden_states, text, child, mother, father)

    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_rejects_encoder_states_without_text():
    attn, proc = make_attn(), make_proc()
    hidden_states, _, child, father, mother = make_inputs()

    with pytest.raises(ValueError):
        proc(attn, hidden_states, encoder_hidden_states=torch.cat([child, father, mother], dim=1))


def test_only_gate_receives_gradients_when_branches_frozen():
    attn, proc = make_attn(), make_proc(parent_scale=0.1)
    for p in list(attn.parameters()) + list(proc.parameters()):
        p.requires_grad_(False)
    for p in proc.gate.parameters():
        p.requires_grad_(True)

    hidden_states, *conds = make_inputs()
    run(attn, proc, hidden_states, *conds).pow(2).mean().backward()

    assert proc.gate.weight.grad is not None
    assert proc.gate.weight.grad.abs().sum() > 0


def tiny_unet():
    return UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=32,
        attention_head_dim=8,
        norm_num_groups=8,
    )


@torch.no_grad()
def test_attach_bap2_replaces_only_cross_attention():
    unet = tiny_unet()
    names = list(unet.attn_processors)
    n_cross = sum(n.endswith("attn2.processor") for n in names)

    modules = attach_bap2(unet, num_tokens=TOKENS, device="cpu", dtype=torch.float32, copy_ip_weights=False)

    procs = unet.attn_processors
    assert len(modules) == n_cross
    assert all(isinstance(procs[n], BAP2) for n in names if n.endswith("attn2.processor"))
    assert not any(isinstance(procs[n], BAP2) for n in names if n.endswith("attn1.processor"))
    assert all(m.gate.weight.dtype == torch.float32 for m in modules.values())

    sample = torch.randn(1, 4, 8, 8)
    encoder_hidden_states = torch.randn(1, 5 + 3 * TOKENS, 32)
    out = unet(sample, 0, encoder_hidden_states=encoder_hidden_states).sample
    assert out.shape == sample.shape


def test_checkpoint_keys_match_processor_names():
    """Checkpoint keys (`bap2_gate_state`) are derived from the processor names."""
    unet = tiny_unet()
    modules = attach_bap2(unet, num_tokens=TOKENS, device="cpu", dtype=torch.float32, copy_ip_weights=False)

    for name, proc in unet.attn_processors.items():
        if isinstance(proc, BAP2):
            safe_name = name.replace(".", "_").replace("processor", "").strip("_")
            assert modules[safe_name] is proc
