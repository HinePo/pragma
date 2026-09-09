"""`OptimizerFactory`: AdamW or hybrid Muon+AdamW with explicit parameter routing.

Implementation plan, section 11.4, 13.2; ADR 0011. Every 2D `Linear` weight
matrix (`q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2`/`PragmaMLMHead.proj`)
routes to Muon; the shared embedding table (also 2D, but explicitly excluded)
and every 1D parameter (`LayerNorm` weight/bias, every `Linear` bias) route to
AdamW — Muon is designed for hidden 2D weight matrices, not embeddings or
gain/bias parameters.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from pragma.config.training_config import TrainingConfig
from pragma.training.muon import Muon


class HybridOptimizer:
    """Composite optimizer forwarding to an independent Muon and AdamW instance.

    Exposes the minimal `torch.optim.Optimizer`-like surface (`step`,
    `zero_grad`, `param_groups`, `state_dict`/`load_state_dict`) that
    `Accelerate` and a standard LR scheduler both need — a scheduler that
    multiplies each `param_groups` entry's own base LR by a shared
    warmup/decay factor works correctly here even though Muon's and AdamW's
    absolute learning rates are on very different scales (ADR 0011).
    """

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.muon.param_groups + self.adamw.param_groups

    def step(self, closure: Any = None) -> None:
        self.muon.step()
        self.adamw.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def _is_muon_eligible(name: str, param: torch.nn.Parameter) -> bool:
    return param.ndim == 2 and "embedding" not in name


def build_optimizer(
    model: nn.Module, config: TrainingConfig
) -> torch.optim.AdamW | HybridOptimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
        )

    if config.optimizer == "muon_adamw":
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (muon_params if _is_muon_eligible(name, param) else adamw_params).append(param)

        muon = Muon(muon_params, lr=config.muon_learning_rate, momentum=config.muon_momentum)
        adamw = torch.optim.AdamW(
            adamw_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.adam_betas,
        )
        return HybridOptimizer(muon, adamw)

    raise ValueError(f"unknown optimizer: {config.optimizer!r}")
