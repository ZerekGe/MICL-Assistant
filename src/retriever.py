"""检索式多模态上下文学习（MICL）demo 选择器 —— 项目核心创新点 C1。

思路：给定当前查询（图像 + 问题），从标注 demo 池中检索 top-k 个最相关的
(图像, 问题, 答案) 三元组，拼入提示作为 in-context examples。

检索分两阶段（method="clip+reward"）：
    1) CLIP 图文相似度粗排；
    2) （可选）reward 模型对候选 demo 重排。

TODO: 接入真实 CLIP / reward 模型；当前先给出可运行的接口与骨架实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class Demo:
    """一条图文示范样本。"""

    image_path: str
    question: str
    answer: str
    meta: dict[str, Any] | None = None


class DemoRetriever:
    """基于图文相似度的 in-context demo 检索器。"""

    def __init__(
        self,
        method: str = "clip",
        top_k: int = 4,
        clip_model: str = "openai/clip-vit-base-patch32",
        reward_model: str | None = None,
    ) -> None:
        self.method = method
        self.top_k = top_k
        self.clip_model_name = clip_model
        self.reward_model_name = reward_model
        self._model = None
        self._processor = None
        self._pool: list[Demo] = []

    def build_index(self, demos: Sequence[Demo]) -> None:
        """构建 demo 池索引。

        TODO: 用 CLIP 提取每张 demo 图像 + 问题的向量，缓存为 numpy 矩阵。
        """
        self._pool = list(demos)

    def retrieve(self, image_path: str, question: str) -> list[Demo]:
        """检索与查询最相关的 top-k demos。

        TODO: 1) CLIP 编码查询与池内样本 -> 余弦相似度粗排；
              2) reward 模型重排（若配置）；
              3) 返回 top_k 条 Demo。
        当前占位实现：返回池中前 top_k 条，保证接口可跑通。
        """
        return self._pool[: self.top_k]

    def format_prompt(self, demos: Sequence[Demo], question: str) -> str:
        """把检索到的 demos 拼成多图 in-context 提示文本。

        TODO: 与 Qwen2.5-VL 的多图输入格式对齐（<image> 占位符顺序）。
        """
        parts = []
        for i, d in enumerate(demos, 1):
            parts.append(f"Example {i}:\n<image>\nQ: {d.question}\nA: {d.answer}")
        parts.append(f"Q: {question}\nA:")
        return "\n\n".join(parts)
