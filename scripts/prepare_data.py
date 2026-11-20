#!/usr/bin/env python3
"""数据准备：生成/下载并清洗为统一 {image, question, answer} 格式。

MVP 阶段用 PIL 合成最小样本（32 条）跑通链路；正式阶段替换为 ChartQA/DocVQA。

用法：
    python scripts/prepare_data.py --config configs/default.yaml
    python scripts/prepare_data.py --num-samples 32 --output-dir data/processed/mvp
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="准备多模态数据集与 demo 检索池")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    p.add_argument("--num-samples", type=int, default=32, help="MVP 合成样本数量")
    p.add_argument("--output-dir", type=str, default="data/processed/mvp", help="样本输出目录")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.mvp_data import generate_samples
    from src.utils import get_logger, load_config

    logger = get_logger("prepare_data")
    cfg = load_config(root / args.config) if (root / args.config).exists() else load_config(args.config)

    out_dir = root / args.output_dir
    logger.info("生成 MVP 样本: num=%d, out=%s", args.num_samples, out_dir)
    records = generate_samples(out_dir, num_samples=args.num_samples, seed=args.seed)

    # 校验：每条样本图片存在、字段齐全
    for rec in records:
        img_path = out_dir / rec["image"]
        if not img_path.exists():
            raise FileNotFoundError(f"样本图片缺失: {img_path}")
        assert rec["question"] and rec["answer"], f"样本字段为空: {rec['id']}"

    logger.info("数据校验通过: %d 条样本，图片+问答对齐全", len(records))
    logger.info("样本清单: %s", out_dir / "samples.jsonl")
    logger.info("示例: Q=%r A=%r", records[0]["question"], records[0]["answer"])


if __name__ == "__main__":
    main()
