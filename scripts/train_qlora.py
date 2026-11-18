#!/usr/bin/env python3
"""QLoRA 指令微调主入口。

用法：
    python scripts/train_qlora.py --config configs/default.yaml
    python scripts/train_qlora.py --config configs/default.yaml --lora-rank 32 --unfreeze-projector
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qwen2.5-VL 的 QLoRA 指令微调")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    p.add_argument("--lora-rank", type=int, default=None, help="覆盖 LoRA 秩（消融用）")
    p.add_argument("--unfreeze-projector", action="store_true", help="解冻视觉投影层（消融用）")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import get_logger, load_config, set_seed

    logger = get_logger("train_qlora")
    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    rank = args.lora_rank if args.lora_rank is not None else cfg["train"]["lora_rank"]
    unfreeze = args.unfreeze_projector or cfg["train"]["unfreeze_projector"]
    logger.info("模型: %s", cfg["model"]["name_or_path"])
    logger.info("LoRA rank=%d, unfreeze_projector=%s", rank, unfreeze)

    # TODO:
    #   1) model, processor = load_model_and_processor(...)
    #   2) model = inject_qlora(model, rank, ...)
    #   3) dataset = load_dataset(...); 构造 Trainer
    #   4) trainer.train(); trainer.save_model()
    logger.info("训练流程待实现（骨架阶段）")


if __name__ == "__main__":
    main()
