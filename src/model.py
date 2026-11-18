"""模型加载 + QLoRA 注入封装（创新点 C2）。"""

from __future__ import annotations

from typing import Any


def load_model_and_processor(model_name_or_path: str, quantization_bit: int = 4, **kwargs: Any):
    """加载 Qwen2.5-VL 与处理器，并按位宽做 4bit 量化。

    TODO:
        - BitsAndBytesConfig(load_in_4bit=True, ...)
        - AutoModelForImageTextToText.from_pretrained(...)
        - AutoProcessor.from_pretrained(...)
    """
    raise NotImplementedError("load_model_and_processor 待实现")


def inject_qlora(model, lora_rank: int = 16, lora_alpha: int = 32,
                 target_modules: list[str] | None = None, **kwargs: Any):
    """注入 QLoRA adapter，可配置秩与目标模块。

    TODO: 用 peft.LoraConfig + get_peft_model 实现，支持投影层解冻消融。
    """
    raise NotImplementedError("inject_qlora 待实现")
