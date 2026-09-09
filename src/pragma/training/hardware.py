"""Hardware-aware precision resolution (ADR 0011).

Device and DDP selection are never done here or anywhere in `pragma.training`
— that's entirely `Accelerate`'s job (construct a plain `Accelerator()` and
read `accelerator.device`/`num_processes`/`process_index`). The one place
this project *does* need to check actual hardware support is mixed
precision: requesting `"fp16"`/`"bf16"` on hardware that can't use it should
degrade gracefully, not error.
"""

from __future__ import annotations

import torch


def resolve_mixed_precision(requested: str) -> str:
    """Downgrades `requested` ("no" | "fp16" | "bf16") to what this process's
    hardware can actually support, given whatever device `Accelerate` would
    select right now."""
    if requested == "no":
        return "no"

    if requested == "bf16":
        # torch supports bf16 autocast on both CPU and CUDA; on CUDA, only
        # some architectures have native bf16 — fall back to fp16 there.
        if not torch.cuda.is_available():
            return "bf16"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if requested == "fp16":
        # fp16 autocast requires a CUDA device; CPU fp16 autocast is not
        # supported by PyTorch.
        return "fp16" if torch.cuda.is_available() else "no"

    raise ValueError(f"unknown mixed_precision setting: {requested!r}")
