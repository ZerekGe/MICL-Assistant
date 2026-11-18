"""通用工具：配置加载、随机种子、日志。"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容不是字典: {path}")
    return cfg


def set_seed(seed: int) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """返回一个带统一格式的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
