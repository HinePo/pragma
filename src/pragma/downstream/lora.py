"""`build_lora_model`: wraps `PragmaForTask` with a PEFT LoRA adapter.

Implementation plan section 11.4's `LoRAAdapterFactory` row and section
15.2 ("Use Hugging Face PEFT rather than implementing LoRA mathematics
manually"). A plain function, not a class — the same "*Factory" -> function"
pattern this project already uses for `OptimizerFactory`/`SchedulerFactory`
(`pragma.training.optimizer.build_optimizer`, `.scheduler.build_scheduler`);
a class here would carry no state PEFT's own `LoraConfig` doesn't already
hold. See ADR 0013 for why plain LoRA (not QLoRA — no base-model
quantization) is the right choice for a ~10M-parameter PRAGMA-S backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from peft import LoraConfig, PeftModel, get_peft_model

from pragma.config import DownstreamConfig
from pragma.downstream.task_model import PragmaForTask
from pragma.modeling import PragmaConfig


def build_lora_model(model: PragmaForTask, config: DownstreamConfig) -> PeftModel:
    """Freezes every backbone parameter except the injected LoRA A/B matrices
    (in `config.target_modules`) and `config.modules_to_save` (the task
    head) — everything else, including every embedding, `LayerNorm`, and
    non-targeted `Linear`, stays frozen and untouched, matching section
    15.1's entry condition that the base checkpoint stays immutable."""
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        modules_to_save=list(config.modules_to_save),
        bias="none",
    )
    # `get_peft_model`'s return type also covers mixed-adapter models; a single
    # `LoraConfig` with no prior adapters always yields a plain `PeftModel`.
    return cast(PeftModel, get_peft_model(model, lora_config))


def load_lora_model(
    pragma_config: PragmaConfig,
    adapter_dir: Path,
    *,
    base_checkpoint_dir: Path,
    base_checkpoint_name: str,
    processor_dir: Path,
) -> PeftModel:
    """Reloads a saved adapter **independently** of the training run that
    produced it (section 15's exit gate: "adapters reload independently") -
    rebuilds a fresh frozen backbone from the base checkpoint (verifying its
    processor hash, same as `load_frozen_backbone`), then reattaches the
    saved LoRA weights and task head (`modules_to_save`) via PEFT's own
    `PeftModel.from_pretrained`. Nothing about this function depends on any
    in-memory state from whatever `DownstreamTrainer` instance saved
    `adapter_dir` - a fresh process calling this is exactly the reload path
    a real deployment would use.
    """
    # Imported here, not at module level: `load_frozen_backbone` lives in
    # `pragma.downstream.trainer`, which imports `build_lora_model` from this
    # module - a top-level import would be circular.
    from pragma.downstream.trainer import load_frozen_backbone

    base_model = load_frozen_backbone(
        pragma_config, base_checkpoint_dir, base_checkpoint_name, processor_dir=processor_dir
    )
    return PeftModel.from_pretrained(base_model, str(adapter_dir))
