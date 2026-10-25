import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from mamba3.mamba3_torch import Mamba3, Mamba3LMHeadModel, Mamba3State, heavy_tail_activation
from tests._mamba3_reference import mamba3_siso_step_ref


def _random_mixer(ngroups=1, chunk_size=16, seed=0):
    torch.manual_seed(seed)
    m = Mamba3(d_model=64, d_state=32, headdim=16, ngroups=ngroups, chunk_size=chunk_size)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
        m.dt_bias.copy_(torch.empty_like(m.dt_bias).uniform_(-4, -1))
        m.D.uniform_(0.5, 1.5)
        m.B_bias.add_(1)
        m.C_bias.add_(1)
        m.B_norm.weight.add_(1)
        m.C_norm.weight.add_(1)
    return m


def _reference(m, u, input_states=None):
    """Mamba3.forward from mamba_ssm, with the Triton kernel swapped for the upstream recurrence."""
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(m.in_proj(u), m.split_sizes, dim=-1)
    z = rearrange(z, "b l (h p) -> b l h p", p=m.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=m.headdim)
    B = rearrange(B, "b l (g n) -> b l g n", g=m.ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=m.ngroups)
    trap = rearrange(trap, "b l h -> b h l")
    A = torch.clamp(-heavy_tail_activation(dd_A.float()), max=-m.A_floor)
    DT = F.softplus(dd_dt + m.dt_bias)
    ADT = rearrange(A * DT, "b l n -> b n l")
    DT = rearrange(DT, "b l n -> b n l")
    angles = angles.unsqueeze(-2).expand(-1, -1, m.nheads, -1).float()
    y, states = mamba3_siso_step_ref(
        m.C_norm(C), m.B_norm(B), x, ADT, DT, trap,
        m.C_bias.squeeze(1), m.B_bias.squeeze(1), angles, m.D, z, input_states,
    )
    return m.out_proj(rearrange(y, "b l h p -> b l (h p)")), Mamba3State(*states)


def _assert_states_close(a: Mamba3State, b: Mamba3State):
    # Angles are compared on the circle so wrap-around at 2*pi doesn't matter.
    torch.testing.assert_close(a.angle.cos(), b.angle.cos(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(a.angle.sin(), b.angle.sin(), atol=1e-4, rtol=1e-4)
    for name in ("ssm", "k", "v"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name), atol=1e-4, rtol=1e-4, msg=name)


@pytest.mark.parametrize("ngroups", [1, 2])
@pytest.mark.parametrize("seqlen", [1, 16, 37])
@torch.no_grad()
def test_forward_matches_reference(ngroups, seqlen):
    m = _random_mixer(ngroups=ngroups)
    u = torch.randn(2, seqlen, m.d_model)
    out, state = m(u)
    ref_out, ref_state = _reference(m, u)
    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    _assert_states_close(state, ref_state)


@torch.no_grad()
def test_forward_continues_from_state():
    m = _random_mixer()
    u = torch.randn(2, 45, m.d_model)
    out1, state = m(u[:, :21])
    out2, state = m(u[:, 21:], state)
    ref_out, ref_state = _reference(m, u)
    torch.testing.assert_close(torch.cat([out1, out2], dim=1), ref_out, atol=1e-4, rtol=1e-4)
    _assert_states_close(state, ref_state)


@torch.no_grad()
def test_step_matches_forward():
    m = _random_mixer()
    u = torch.randn(2, 23, m.d_model)
    ref_out, ref_state = m(u)
    state, outs = None, []
    for t in range(u.shape[1]):
        out, state = m.step(u[:, t], state)
        outs.append(out)
    torch.testing.assert_close(torch.stack(outs, dim=1), ref_out, atol=1e-4, rtol=1e-4)
    _assert_states_close(state, ref_state)


def test_backward_runs():
    m = _random_mixer()
    u = torch.randn(2, 37, m.d_model, requires_grad=True)
    m(u)[0].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


TINY_CONFIG = {
    "d_model": 64,
    "d_intermediate": 128,
    "n_layer": 2,
    "vocab_size": 100,
    "ssm_cfg": {"layer": "Mamba3", "d_state": 32, "headdim": 16, "chunk_size": 16},
    "pad_vocab_size_multiple": 16,
    "tie_embeddings": True,
}


@torch.no_grad()
def test_lm_prefill_then_decode_matches_full_forward():
    torch.manual_seed(0)
    model = Mamba3LMHeadModel(TINY_CONFIG).eval()
    ids = torch.randint(0, 100, (2, 30))
    full_logits, _ = model(ids)
    logits, states = model(ids[:, :19])
    step_logits = []
    for t in range(19, 30):
        step, states = model.step(ids[:, t], states)
        step_logits.append(step)
    torch.testing.assert_close(logits, full_logits[:, :19], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(torch.stack(step_logits, dim=1), full_logits[:, 19:], atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_lm_generate_greedy_shape():
    torch.manual_seed(0)
    model = Mamba3LMHeadModel(TINY_CONFIG).eval()
    out = model.generate(torch.randint(0, 100, (2, 5)), max_length=12)
    assert out.shape == (2, 12)
    assert out.max() < 100  # never samples padded vocab entries


def test_lm_rejects_mimo():
    cfg = {**TINY_CONFIG, "ssm_cfg": {**TINY_CONFIG["ssm_cfg"], "is_mimo": True}}
    with pytest.raises(NotImplementedError):
        Mamba3LMHeadModel(cfg)


@pytest.mark.slow
@torch.no_grad()
def test_pretrained_187m():
    """Downloads ~370MB. Run with `uv run pytest -m slow`."""
    from transformers import AutoTokenizer

    from mamba3 import TOKENIZER_ID

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = Mamba3LMHeadModel.from_pretrained("state-spaces/mamba3-siso-187m", device="cpu", dtype=torch.float32)
    text = (
        "The mitochondria is the powerhouse of the cell. It produces energy in the form of ATP "
        "through a process called cellular respiration, which requires oxygen and glucose."
    )
    ids = tokenizer(text, return_tensors="pt").input_ids
    logits, _ = model(ids)
    ppl = math.exp(F.cross_entropy(logits[0, :-1], ids[0, 1:]).item())
    assert ppl < 30, ppl

    prefix_logits, states = model(ids[:, :-4])
    for t in range(ids.shape[1] - 4, ids.shape[1]):
        step_logits, states = model.step(ids[:, t], states)
    torch.testing.assert_close(step_logits, logits[:, -1], atol=2e-3, rtol=2e-3)
