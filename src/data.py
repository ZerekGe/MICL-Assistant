"""数据集加载与统一多模态预处理（创新点 C3）。"""

from __future__ import annotations

from typing import Any


def load_dataset(name: str, split: str = "train", **kwargs: Any):
    """加载 ChartQA / DocVQA 等数据集。

    TODO: 用 datasets.load_dataset 加载并统一字段为
          {image, question, answer}。
    """
    raise NotImplementedError(f"load_dataset({name!r}) 待实现")


def sample_video_frames(video_path: str, num_frames: int = 8, strategy: str = "uniform"):
    """从视频中均匀采样 num_frames 帧（C3）。

    TODO: 用 decord 读取视频，按 strategy 采样并返回 PIL.Image 列表。
    """
    raise NotImplementedError("sample_video_frames 待实现")


def build_demo_pool(dataset, size: int = 2000):
    """从训练集抽取 demo 池供检索器使用。

    TODO: 随机抽样 size 条，返回 list[Demo]。
    """
    raise NotImplementedError("build_demo_pool 待实现")
