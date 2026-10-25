"""Pure-PyTorch Mamba-3 (SISO) language model for machines without the official CUDA kernels.

Loads the official ``state-spaces/mamba3-siso-*`` checkpoints unchanged. The math follows
``mamba3_siso_step_ref`` in state-spaces/mamba ``tests/ops/triton/test_mamba3_siso.py``:

    h_t = alpha_t * h_{t-1} + beta_t * v_{t-1} k_{t-1}^T + gamma_t * v_t k_t^T
    y_t = h_t q_t + D * v_t,            then gated by silu(z_t)

with alpha_t = exp(A_t dt_t), beta_t = (1 - trap_t) dt_t alpha_t, gamma_t = trap_t dt_t,
and q/k rotated by a data-dependent cumulative angle. Prefill uses a chunked parallel scan.
"""

import json
import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

TWO_PI = 2 * math.pi


class Mamba3State(NamedTuple):
    angle: torch.Tensor  # (batch, nheads, num_rope_angles) cumulative rotary angle
    ssm: torch.Tensor  # (batch, nheads, headdim, d_state)
    k: torch.Tensor  # (batch, nheads, d_state) last rotated k, for the trapezoid term
    v: torch.Tensor  # (batch, nheads, headdim) last v, for the trapezoid term


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """f(x) = 1 + x for x >= 0, 1 / (1 - x) for x < 0."""
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """out[..., i, j] = x[..., j+1] + ... + x[..., i] for j <= i, -inf above the diagonal."""
    T = x.size(-1)
    x = x[..., None].expand(*x.shape, T)
    x = x.masked_fill(~torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), -1), 0)
    x = torch.cumsum(x, dim=-2)
    return x.masked_fill(~torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)), -torch.inf)


def _rotate_pairwise(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs (x0, x1), (x2, x3), ... of the first 2 * cos.shape[-1] dims of x."""
    rot_dim = 2 * cos.shape[-1]
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    x0, x1 = x_rot.unflatten(-1, (-1, 2)).unbind(-1)
    x_rot = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)
    return torch.cat([x_rot, x_pass], dim=-1)


def _chunked_scan(q, k, v, log_decay, h0, chunk_size):
    """Causal linear attention with per-step decay, computed chunk by chunk.

    y_t = sum_{s<=t} exp(a_{s+1} + ... + a_t) (q_t . k_s) v_s + exp(a_1 + ... + a_t) h0 q_t

    q, k: (b, l, h, n); v: (b, l, h, p); log_decay: (b, l, h); h0: (b, h, p, n) or None.
    Returns y: (b, l, h, p) and the final state (b, h, p, n).
    """
    b, l, h, n = q.shape
    p = v.shape[-1]
    pad = (-l) % chunk_size
    if pad:
        # Zero k/v and zero log-decay leave both outputs and the final state unchanged.
        q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
        log_decay = F.pad(log_decay, (0, 0, 0, pad))
    q, k, v = (rearrange(t, "b (c s) h d -> b c s h d", s=chunk_size) for t in (q, k, v))
    a = rearrange(log_decay, "b (c s) h -> b h c s", s=chunk_size)
    a_cum = torch.cumsum(a, dim=-1)

    scores = torch.einsum("bclhn,bcshn->bhcls", q, k) * torch.exp(_segsum(a))
    y = torch.einsum("bhcls,bcshp->bclhp", scores, v)

    decay_to_end = torch.exp(a_cum[..., -1:] - a_cum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", k, decay_to_end, v)
    if h0 is None:
        h0 = q.new_zeros(b, h, p, n)
    states = torch.cat([h0[:, None], states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(a_cum[..., -1], (1, 0))))
    states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = states[:, :-1], states[:, -1]

    y = y + torch.einsum("bclhn,bchpn,bhcl->bclhp", q, states, torch.exp(a_cum))
    return rearrange(y, "b c s h p -> b (c s) h p")[:, :l], final_state


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return out.to(self.weight.dtype)


class GatedRMSNorm(RMSNorm):
    """Group-wise RMSNorm followed by a silu(z) gate (norm_before_gate=True)."""

    def __init__(self, dim, group_size, eps=1e-5, device=None, dtype=None):
        super().__init__(dim, eps=eps, device=device, dtype=dtype)
        self.group_size = group_size

    def forward(self, x, z):
        xg = x.float().unflatten(-1, (-1, self.group_size))
        xg = xg * torch.rsqrt(xg.square().mean(-1, keepdim=True) + self.eps)
        return xg.flatten(-2) * self.weight.float() * F.silu(z.float())


class Mamba3(nn.Module):
    """Mamba-3 SISO mixer. Parameter names match ``mamba_ssm.modules.mamba3.Mamba3``."""

    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        chunk_size=64,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,  # dt_min, dt_max, ...: init-only
    ):
        if is_mimo:
            raise NotImplementedError(
                "mamba3_torch only implements SISO. Run MIMO checkpoints with the official mamba_ssm kernels."
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.ngroups = ngroups
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        assert rope_fraction in (0.5, 1.0)
        split_tensor_size = int(d_state * rope_fraction)
        split_tensor_size -= split_tensor_size % 2
        self.num_rope_angles = split_tensor_size // 2

        # in_proj order: [z, x, B, C, dd_dt, dd_A, trap, angle]
        self.split_sizes = [
            self.d_inner,
            self.d_inner,
            d_state * ngroups,
            d_state * ngroups,
            self.nheads,
            self.nheads,
            self.nheads,
            self.num_rope_angles,
        ]
        self.in_proj = nn.Linear(d_model, sum(self.split_sizes), bias=False, **factory_kwargs)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, device=device))
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, device=device))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, d_state, device=device))
        self.B_norm = RMSNorm(d_state, **factory_kwargs)
        self.C_norm = RMSNorm(d_state, **factory_kwargs)
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        if is_outproj_norm:
            self.norm = GatedRMSNorm(self.d_inner, group_size=headdim, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    def _ssm_inputs(self, u):
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(self.in_proj(u), self.split_sizes, dim=-1)
        A = torch.clamp(-heavy_tail_activation(dd_A.float()), max=-self.A_floor)
        dt = F.softplus(dd_dt.float() + self.dt_bias.float())
        heads_per_group = self.nheads // self.ngroups
        B = self.B_norm(B.unflatten(-1, (self.ngroups, self.d_state))).repeat_interleave(heads_per_group, dim=-2)
        C = self.C_norm(C.unflatten(-1, (self.ngroups, self.d_state))).repeat_interleave(heads_per_group, dim=-2)
        return dict(
            q=C.float() + self.C_bias.squeeze(1).float(),
            k=B.float() + self.B_bias.squeeze(1).float(),
            v=x.unflatten(-1, (self.nheads, self.headdim)).float(),
            z=z.unflatten(-1, (self.nheads, self.headdim)),
            adt=A * dt,
            dt=dt,
            trap=torch.sigmoid(trap.float()),
            angles=torch.tanh(angles.float()) * math.pi,
        )

    def _output(self, y, z, dtype):
        if self.is_outproj_norm:
            y = self.norm(y.flatten(-2), z.flatten(-2))
        else:
            y = (y * F.silu(z.float())).flatten(-2)
        return self.out_proj(y.to(dtype))

    def forward(self, u, state: Mamba3State | None = None):
        """Parallel (chunked) forward. u: (batch, seqlen, d_model). Returns (out, final_state)."""
        s = self._ssm_inputs(u)
        q, k, v, dt, trap = s["q"], s["k"], s["v"], s["dt"], s["trap"]

        angle = torch.cumsum(s["angles"][:, :, None, :] * dt[..., None], dim=1)  # (b, l, h, S)
        if state is not None:
            angle = angle + state.angle[:, None]
        angle = torch.remainder(angle, TWO_PI)

        # k_s is weighted by gamma_s + beta_{s+1}; the beta part only applies from t = s + 1,
        # so qk_dot removes it from the diagonal.
        shifted_beta = F.pad((dt * (1 - trap))[:, 1:], (0, 0, 0, 1))
        qk_dot = (q * k).sum(-1)  # rotation-invariant, so computed before rotating
        cos, sin = angle.cos(), angle.sin()
        q, k = _rotate_pairwise(q, cos, sin), _rotate_pairwise(k, cos, sin)

        h0 = None
        if state is not None:
            beta0 = dt[:, 0] * (1 - trap[:, 0])
            h0 = state.ssm + beta0[..., None, None] * state.v[..., :, None] * state.k[..., None, :]
        scale = trap * dt + shifted_beta
        y, ssm = _chunked_scan(q, k * scale[..., None], v, s["adt"], h0, self.chunk_size)
        y = y + v * (self.D.float()[:, None] - (qk_dot * shifted_beta)[..., None])

        new_state = Mamba3State(angle=angle[:, -1], ssm=ssm, k=k[:, -1], v=v[:, -1])
        return self._output(y, s["z"], u.dtype), new_state

    def step(self, u, state: Mamba3State | None = None):
        """One recurrent decode step. u: (batch, d_model). Returns (out, new_state)."""
        s = self._ssm_inputs(u)
        q, k, v, dt, trap = s["q"], s["k"], s["v"], s["dt"], s["trap"]
        if state is None:
            state = self.allocate_state(u.shape[0], u.device)

        angle = torch.remainder(state.angle + s["angles"][:, None, :] * dt[..., None], TWO_PI)
        cos, sin = angle.cos(), angle.sin()
        q, k = _rotate_pairwise(q, cos, sin), _rotate_pairwise(k, cos, sin)

        alpha = torch.exp(s["adt"])
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt
        ssm = (
            alpha[..., None, None] * state.ssm
            + beta[..., None, None] * state.v[..., :, None] * state.k[..., None, :]
            + gamma[..., None, None] * v[..., :, None] * k[..., None, :]
        )
        y = torch.einsum("bhpn,bhn->bhp", ssm, q) + self.D.float()[:, None] * v

        new_state = Mamba3State(angle=angle, ssm=ssm, k=k, v=v)
        return self._output(y, s["z"], u.dtype), new_state

    def allocate_state(self, batch_size, device=None):
        device = self.in_proj.weight.device if device is None else device
        zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.float32)  # noqa: E731
        return Mamba3State(
            angle=zeros(batch_size, self.nheads, self.num_rope_angles),
            ssm=zeros(batch_size, self.nheads, self.headdim, self.d_state),
            k=zeros(batch_size, self.nheads, self.d_state),
            v=zeros(batch_size, self.nheads, self.headdim),
        )


class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features, multiple_of=128, device=None, dtype=None):
        super().__init__()
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=False, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


class Block(nn.Module):
    """Add -> RMSNorm -> Mamba3 -> Add -> RMSNorm -> GatedMLP, as in ``mamba_ssm.modules.block``."""

    def __init__(self, d_model, d_intermediate, ssm_cfg, norm_epsilon, residual_in_fp32, layer_idx, device, dtype):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.norm = RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)
        self.mixer = Mamba3(d_model, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        if d_intermediate > 0:
            self.norm2 = RMSNorm(d_model, eps=norm_epsilon, **factory_kwargs)
            self.mlp = GatedMLP(d_model, d_intermediate, **factory_kwargs)
        else:
            self.mlp = None

    def _add(self, hidden, residual):
        residual = hidden if residual is None else hidden + residual
        return residual.float() if self.residual_in_fp32 else residual

    def forward(self, hidden, residual, state=None, step=False):
        residual = self._add(hidden, residual)
        mixer = self.mixer.step if step else self.mixer
        hidden, state = mixer(self.norm(residual), state)
        if self.mlp is not None:
            residual = self._add(hidden, residual)
            hidden = self.mlp(self.norm2(residual))
        return hidden, residual, state


class Mamba3LMHeadModel(nn.Module):
    """Drop-in for ``mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel`` with Mamba3 SISO layers."""

    def __init__(self, config: dict, device=None, dtype=None):
        super().__init__()
        self.config = config
        ssm_cfg = dict(config.get("ssm_cfg", {}))
        layer = ssm_cfg.pop("layer", "Mamba3")
        if layer != "Mamba3" or config.get("attn_layer_idx"):
            raise NotImplementedError("mamba3_torch only supports pure Mamba-3 models (no attention layers).")
        if not config.get("rms_norm", True):
            raise NotImplementedError("mamba3_torch only supports rms_norm=True.")
        d_model = config["d_model"]
        vocab_size = config["vocab_size"]
        pad_multiple = config.get("pad_vocab_size_multiple", 8)
        vocab_size += -vocab_size % pad_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        self.backbone = nn.Module()
        self.backbone.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.backbone.layers = nn.ModuleList(
            Block(
                d_model,
                config.get("d_intermediate", 0),
                ssm_cfg,
                norm_epsilon=config.get("norm_epsilon", 1e-5),
                residual_in_fp32=config.get("residual_in_fp32", True),
                layer_idx=i,
                **factory_kwargs,
            )
            for i in range(config["n_layer"])
        )
        self.backbone.norm_f = RMSNorm(d_model, eps=config.get("norm_epsilon", 1e-5), **factory_kwargs)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)
        if config.get("tie_embeddings", True):
            self.lm_head.weight = self.backbone.embedding.weight

    def _run(self, input_ids, states, step):
        hidden, residual = self.backbone.embedding(input_ids), None
        states = [None] * len(self.backbone.layers) if states is None else states
        new_states = []
        for layer, state in zip(self.backbone.layers, states):
            hidden, residual, state = layer(hidden, residual, state, step=step)
            new_states.append(state)
        hidden = self.backbone.norm_f(hidden + residual)
        return hidden, new_states

    def forward(self, input_ids, states=None, num_last_tokens=0):
        """input_ids: (batch, seqlen). Returns (logits, per-layer states)."""
        hidden, states = self._run(input_ids, states, step=False)
        if num_last_tokens > 0:
            hidden = hidden[:, -num_last_tokens:]
        return self.lm_head(hidden), states

    def step(self, input_ids, states):
        """input_ids: (batch,). Returns (logits (batch, vocab), new states)."""
        hidden, states = self._run(input_ids, states, step=True)
        return self.lm_head(hidden), states

    @torch.inference_mode()
    def generate(self, input_ids, max_length, top_k=1, top_p=0.0, temperature=1.0, eos_token_id=None):
        """Same sampling semantics as ``mamba_ssm``'s ``generate``: top_k=1 is greedy,
        top_k=0 disables top-k, top_p=0 disables nucleus sampling. Returns (batch, <= max_length)."""
        logits, states = self(input_ids, num_last_tokens=1)
        logits = logits[:, -1]
        sequences = [input_ids]
        done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for _ in range(max_length - input_ids.shape[1]):
            token = _sample(logits[:, : self.config["vocab_size"]].float(), top_k, top_p, temperature)
            sequences.append(token[:, None])
            if eos_token_id is not None:
                done |= token == eos_token_id
                if done.all():
                    break
            logits, states = self.step(token, states)
        return torch.cat(sequences, dim=1)

    @classmethod
    def from_pretrained(cls, model_id, device=None, dtype=None):
        """Load an official ``state-spaces/mamba3-siso-*`` checkpoint (HF Hub id or local dir)."""
        config, state_dict = _load_checkpoint(model_id)
        model = cls(config, device="meta", dtype=dtype)
        model.load_state_dict(state_dict, strict=True, assign=True)
        if config.get("tie_embeddings", True):
            model.lm_head.weight = model.backbone.embedding.weight
        model.to(device=device, dtype=dtype)
        return model.eval()


def _sample(logits, top_k, top_p, temperature):
    if top_k == 1:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -torch.inf)
    if 0.0 < top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p
        logits = logits.scatter(-1, idx, sorted_logits.masked_fill(remove, -torch.inf))
    return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)


def _load_checkpoint(model_id):
    from pathlib import Path

    local = Path(model_id)
    if local.is_dir():
        config_path, weights_path = local / "config.json", local / "pytorch_model.bin"
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_id, "config.json")
        weights_path = hf_hub_download(model_id, "pytorch_model.bin")
    with open(config_path) as f:
        config = json.load(f)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    return config, state_dict
