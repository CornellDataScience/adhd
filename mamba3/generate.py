"""Generate text with a Mamba-3 checkpoint.

    uv run python -m mamba3.generate --model siso-187m --prompt "Cornell Data Science is"
"""

import argparse
import time

import torch
from transformers import AutoTokenizer

from mamba3 import CHECKPOINTS, TOKENIZER_ID, load_model

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="siso-187m", help=f"HF id, local dir, or one of: {', '.join(CHECKPOINTS)}")
    parser.add_argument("--prompt", default="Mamba-3 is a state space model that")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1, help="1 = greedy, 0 = no top-k limit")
    parser.add_argument("--top-p", type=float, default=0.0, help="0 = disabled")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--backend", choices=["auto", "official", "torch"], default="auto")
    parser.add_argument("--device", default=None, help="Defaults to cuda / mps / cpu")
    parser.add_argument("--dtype", choices=list(DTYPES), default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    t0 = time.perf_counter()
    model = load_model(args.model, args.backend, args.device, DTYPES.get(args.dtype))
    device = next(model.parameters()).device
    print(f"Loaded {args.model} ({type(model).__module__}) on {device} in {time.perf_counter() - t0:.1f}s")

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    t0 = time.perf_counter()
    out = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    elapsed = time.perf_counter() - t0
    new_tokens = out.shape[1] - input_ids.shape[1]
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print(f"\n{new_tokens} tokens in {elapsed:.2f}s ({new_tokens / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
