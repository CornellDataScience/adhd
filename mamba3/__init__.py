"""Mamba-3 language models: official CUDA kernels on the cluster, pure PyTorch elsewhere."""

from mamba3.mamba3_torch import Mamba3, Mamba3LMHeadModel, Mamba3State

TOKENIZER_ID = "unsloth/Llama-3.1-8B"  # ungated copy of meta-llama/Llama-3.1-8B's tokenizer

CHECKPOINTS = {
    "siso-187m": "state-spaces/mamba3-siso-187m",
    "siso-443m": "state-spaces/mamba3-siso-443m",
    "siso-893m": "state-spaces/mamba3-siso-893m",
    "siso-1.5b": "state-spaces/mamba3-siso-1.5b",
    "mimo-187m": "state-spaces/mamba3-mimo-187m",
    "mimo-444m": "state-spaces/mamba3-mimo-444m",
    "mimo-894m": "state-spaces/mamba3-mimo-894m",
    "mimo-1.5b": "state-spaces/mamba3-mimo-1.5b",
}


def official_kernels_available() -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return True


def load_model(model_id, backend="auto", device=None, dtype=None):
    """backend: "official" (mamba_ssm CUDA kernels), "torch" (SISO only), or "auto"."""
    import torch

    model_id = CHECKPOINTS.get(model_id, model_id)
    if backend == "auto":
        backend = "official" if official_kernels_available() else "torch"
    if backend == "official":
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        return MambaLMHeadModel.from_pretrained(model_id, device=device or "cuda", dtype=dtype or torch.bfloat16).eval()
    if backend == "torch":
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        return Mamba3LMHeadModel.from_pretrained(model_id, device=device, dtype=dtype or torch.float32)
    raise ValueError(f"Unknown backend: {backend!r}")


__all__ = [
    "CHECKPOINTS",
    "TOKENIZER_ID",
    "Mamba3",
    "Mamba3LMHeadModel",
    "Mamba3State",
    "load_model",
    "official_kernels_available",
]
