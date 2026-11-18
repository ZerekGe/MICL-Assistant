#!/usr/bin/env python3
"""数据准备：下载 ChartQA / DocVQA，清洗对齐，构建 demo 检索池。

用法：
    python scripts/prepare_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="准备多模态数据集与 demo 检索池")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    p.add_argument("--datasets", nargs="+", default=None, help="覆盖配置中的数据集列表")
    p.add_argument("--cache-dir", type=str, default="data/raw", help="数据缓存目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 延迟导入：让 --help 在未安装依赖时也能工作
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.utils import get_logger, load_config, set_seed

    logger = get_logger("prepare_data")
    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])

    datasets = args.datasets or cfg["data"]["train_datasets"]
    demo_size = cfg["data"]["demo_pool_size"]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("准备数据集: %s", datasets)
    # TODO: 逐个下载并清洗数据集，统一为 {image, question, answer}
    for name in datasets:
        logger.info("[%s] 下载到 %s —— 待实现", name, cache_dir / name)

    logger.info("构建 demo 检索池, size=%d —— 待实现", demo_size)
    logger.info("数据准备完成（骨架阶段，具体逻辑待实现）")


if __name__ == "__main__":
    main()
