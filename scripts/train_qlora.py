#!/usr/bin/env python3
"""QLoRA 指令微调主入口。

MVP 阶段：3B 模型 + 少量样本跑 1 个 step，验证链路与显存。
真实模式用 Qwen2.5-VL-3B（4bit）；--mock 用随机小模型走通训练循环。

用法：
    python scripts/train_qlora.py --config configs/default.yaml
    python scripts/train_qlora.py --max-steps 1 --lora-rank 8
    python scripts/train_qlora.py --mock              # 无依赖兜底
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-VL 的 QLoRA 指令微调")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    p.add_argument("--lora-rank", type=int, default=None, help="覆盖 LoRA 秩（消融用）")
    p.add_argument("--unfreeze-projector", action="store_true", help="解冻视觉投影层（消融用）")
    p.add_argument("--max-steps", type=int, default=1, help="训练步数（MVP=1）")
    p.add_argument("--num-samples", type=int, default=8, help="参与训练的样本数")
    p.add_argument("--data-dir", type=str, default="data/processed/mvp", help="样本目录")
    p.add_argument("--model", type=str, default=None, help="覆盖模型路径/id")
    p.add_argument("--mock", action="store_true", help="用随机小模型跑通训练循环（无 GPU/无模型也可用）")
    return p.parse_args()


def build_inputs(processor, rec: dict, max_pixels: int | None = None):
    """把一条 {image, question, answer} 样本转成 Qwen2.5-VL 训练张量。

    直接用 processor 处理静态图片（MVP 只做图像问答），绕开 qwen_vl_utils
    对 torchvision 的依赖。视频/多图扩展时再引入该工具。
    """
    from PIL import Image

    image = Image.open(rec["image_path"]).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": rec["image_path"]},
                {"type": "text", "text": rec["question"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": rec["answer"]}],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    kwargs = dict(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    if max_pixels:
        kwargs["max_pixels"] = max_pixels
    return processor(**kwargs)


def run_real(args, cfg, logger) -> None:
    """真实 3B QLoRA 单步训练。"""
    import torch
    from src.model import load_model_and_processor, inject_qlora
    from src.mvp_data import load_samples

    rank = args.lora_rank if args.lora_rank is not None else cfg["train"]["lora_rank"]
    unfreeze = args.unfreeze_projector or cfg["train"]["unfreeze_projector"]
    model_id = args.model or cfg["model"]["name_or_path"]

    logger.info("加载模型: %s (4bit)", model_id)
    model, processor = load_model_and_processor(
        model_id, quantization_bit=cfg["train"]["quantization_bit"]
    )
    model = inject_qlora(
        model,
        lora_rank=rank,
        lora_alpha=cfg["train"]["lora_alpha"],
        lora_dropout=cfg["train"]["lora_dropout"],
        target_modules=cfg["train"]["target_modules"],
        unfreeze_projector=unfreeze,
    )
    model.train()
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("可训练参数: %s (rank=%d, unfreeze_projector=%s)", f"{trainable:,}", rank, unfreeze)

    samples = load_samples(args.data_dir)[: args.num_samples]
    logger.info("训练样本: %d 条", len(samples))

    from torch.optim import AdamW

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["train"]["learning_rate"]))

    max_pixels = cfg["model"].get("max_pixels")
    for step in range(args.max_steps):
        rec = samples[step % len(samples)]
        batch = build_inputs(processor, rec, max_pixels)
        batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
        # 训练标签 = 输入（简单因果 LM；正式阶段会 mask 掉 prompt）
        batch["labels"] = batch["input_ids"].clone()

        t0 = time.time()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        dt = time.time() - t0

        mem = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(
            "step %d/%d loss=%.4f 耗时=%.2fs 峰值显存=%.2fGB",
            step + 1, args.max_steps, loss.item(), dt, mem,
        )

    logger.info("真实 QLoRA 单步训练验证通过 ✅")


def run_mock(args, cfg, logger) -> None:
    """无 GPU/无模型兜底：随机小模型跑通完整训练循环。"""
    import torch
    from torch import nn

    logger.info("[MOCK] 用随机小模型跑通训练循环（验证代码路径，非真实训练）")
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 16, device="cuda")
    y = torch.randint(0, 8, (4,), device="cuda")
    for step in range(args.max_steps):
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        logger.info("[MOCK] step %d/%d loss=%.4f", step + 1, args.max_steps, loss.item())
    logger.info("[MOCK] 训练循环验证通过（注意：这不是真实 Qwen2.5-VL 训练）✅")


def main() -> None:
    args = parse_args()
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.utils import get_logger, load_config

    logger = get_logger("train_qlora")
    cfg_path = root / args.config if (root / args.config).exists() else args.config
    cfg = load_config(cfg_path)

    if args.mock:
        run_mock(args, cfg, logger)
    else:
        run_real(args, cfg, logger)


if __name__ == "__main__":
    main()
