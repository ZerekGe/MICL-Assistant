#!/usr/bin/env python3
"""评测：ChartQA / DocVQA 精确率 + 可选 GPT-4o 裁判。

用法：
    python scripts/eval_chartqa.py --config configs/default.yaml --judge
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多模态助手评测")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    p.add_argument("--checkpoint", type=str, default=None, help="待评测的权重路径")
    p.add_argument("--judge", action="store_true", help="启用 GPT-4o-as-judge")
    p.add_argument("--output", type=str, default="outputs/eval_result.json", help="结果输出路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import get_logger, load_config

    logger = get_logger("eval_chartqa")
    cfg = load_config(args.config)
    logger.info("评测数据集: %s", cfg["data"]["eval_dataset"])
    if args.judge:
        logger.info("GPT-4o judge: 开启（需设置 %s）", cfg["eval"]["judge_api_key_env"])

    # TODO:
    #   1) 加载模型 + 检索器，逐样本生成答案
    #   2) 计算 ChartQA/DocVQA 精确率
    #   3) 若 --judge，调用 GPT-4o 对开放式问答打分
    #   4) 写出 args.output
    logger.info("评测流程待实现（骨架阶段）")


if __name__ == "__main__":
    main()
