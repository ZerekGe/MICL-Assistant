"""模型加载 + QLoRA 注入封装（创新点 C2）。

MVP 阶段：4bit 量化加载 Qwen2.5-VL-3B，注入 LoRA，验证单步反向传播。
模型优先从本地缓存读取，缺失时用 ModelScope 自动下载（HF 在本环境不可达）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ModelScope 上的模型 id（与 HF id 相同）
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_CACHE_DIR = "D:/caches/modelscope"


def ensure_model(model_id: str = DEFAULT_MODEL_ID, cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """确保模型在本地；缺失则从 ModelScope 下载，返回本地绝对路径。"""
    # 若 model_id 本身就是本地已存在的路径，直接用
    if os.path.isdir(model_id):
        return model_id

    from modelscope import snapshot_download

    local_dir = snapshot_download(model_id, cache_dir=cache_dir)
    return local_dir


def load_model_and_processor(
    model_name_or_path: str = DEFAULT_MODEL_ID,
    quantization_bit: int = 4,
    device_map: str = "auto",
    **kwargs: Any,
):
    """加载 Qwen2.5-VL 与处理器，并按位宽做 4bit 量化。

    Returns:
        (model, processor)
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    local_path = ensure_model(model_name_or_path)

    quant_cfg = None
    if quantization_bit == 4:
        from transformers import BitsAndBytesConfig

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        local_path,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
    return model, processor


def inject_qlora(
    model,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    unfreeze_projector: bool = False,
    **kwargs: Any,
):
    """注入 QLoRA adapter，支持投影层解冻消融。"""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    if unfreeze_projector:
        # Qwen2.5-VL 视觉投影层
        for name, param in model.named_parameters():
            if "visual" in name and "proj" in name.lower():
                param.requires_grad = True

    return model
