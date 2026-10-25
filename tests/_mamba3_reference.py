"""Reference Mamba-3 SISO recurrence, copied verbatim from state-spaces/mamba
(tests/ops/triton/test_mamba3_siso.py @ e9594ce, Apache-2.0, Copyright (c) 2025 Dao AI Lab,
Goombalab). Used as the test oracle for mamba3/mamba3_torch.py.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import repeat


def mamba3_siso_step_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference implementation of Mamba-3 in recurrent (step) mode.
    
    Args:
        Input_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)
    
    Returns:
        out: Output tensor (batch, seqlen, nheads, headdim_v)
        Final_States: Tuple of (Angle_State, SSM_State, K_State, V_State)
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0
    Angles = torch.tanh(Angles) * math.pi

    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
        return rotated
    
    # Initialize states
    if Input_States is not None:
        Angle_State, SSM_State, K_State, V_State = Input_States
        Angle_State = Angle_State.clone()
        SSM_State = SSM_State.clone().to(torch.float32)
        K_State = K_State.clone()
        V_State = V_State.clone()
    else:
        Angle_State = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=device)
        SSM_State = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device)
        K_State = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
        V_State = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)
    
    TWO_PI = 2 * math.pi
    out_arr = []

    for idx in range(seqlen):
        q = Q[:, idx, :, :] + Q_bias.unsqueeze(0)
        k = K[:, idx, :, :] + K_bias.unsqueeze(0)
        v = V[:, idx, :, :]
        adt = ADT[:, :, idx]
        dt = DT[:, :, idx]
        trap = Trap[:, :, idx]
        z = Z[:, idx, :, :] if Z is not None else None
        angles = Angles[:, idx, :, :]

        # Update angle state with cumsum: Angle_State = (Angle_State + Angles * DT) mod 2π
        Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)

        # Apply rotary embeddings to Q and K using cumulative angles
        cos_angles = torch.cos(Angle_State)
        sin_angles = torch.sin(Angle_State)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

        trap = torch.sigmoid(trap)
        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        # Update SSM state using previous K_State and V_State
        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State 
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * (K_State.unsqueeze(-2) * V_State.unsqueeze(-1))
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * (k_rot.unsqueeze(-2) * v.unsqueeze(-1))

        # Compute output
        out = torch.einsum("bhdD, bhD -> bhd", SSM_State, q_rot.to(SSM_State.dtype))
        
        if D is not None:
            out = out + D[None, :, None] * v
        
        if Z is not None:
            out = out * z * torch.sigmoid(z)
        
        out_arr.append(out)
        
        # Update K and V states for next step
        K_State = k_rot
        V_State = v
    
    out = torch.stack(out_arr, dim=1)
    Final_States = (Angle_State, SSM_State, K_State, V_State)
    return out, Final_States
