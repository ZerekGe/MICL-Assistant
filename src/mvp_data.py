"""MVP 冒烟测试数据生成器。

用途：在真实数据集（ChartQA/DocVQA）下载链路就绪前，用 PIL 合成最小的
「图表图片 + 问答对」样本，供数据→训练→评测→Demo 四步链路排雷。

注意：这是 MVP 专用的占位数据生成器，**不是**正式训练数据。正式实验阶段
（后续）会替换为真实的 ChartQA/DocVQA。生成逻辑确定性（固定 seed），
便于复现。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 图表主题词，让问答有语义（虽然答案是程序算出来的）
_LABELS = ["A", "B", "C", "D", "E", "F"]
_QUESTIONS = {
    "max": "Which category has the largest value?",
    "min": "Which category has the smallest value?",
    "sum": "What is the sum of all values?",
    "count": "How many bars are in the chart?",
}


def _get_font(size: int):
    """优先用等宽/系统字体，找不到就用默认字体。"""
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_barchart(values: list[int], labels: list[str], path: Path) -> None:
    """画一张简单柱状图并保存。"""
    w, h = 480, 320
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(18)
    title_font = _get_font(22)

    draw.text((10, 8), "Sales by Category", fill="black", font=title_font)

    n = len(values)
    margin_left, margin_bottom, margin_top = 40, 50, 45
    plot_h = h - margin_bottom - margin_top
    plot_w = w - margin_left - 20
    max_val = max(values) or 1
    bar_w = plot_w / n * 0.6
    gap = plot_w / n

    for i, (v, label) in enumerate(zip(values, labels)):
        x0 = margin_left + i * gap + (gap - bar_w) / 2
        bar_h = (v / max_val) * plot_h
        y0 = h - margin_bottom - bar_h
        y1 = h - margin_bottom
        draw.rectangle([x0, y0, x0 + bar_w, y1], fill="#4C78A8", outline="black")
        draw.text((x0 + bar_w / 2 - 6, y0 - 24), str(v), fill="black", font=font)
        draw.text((x0 + bar_w / 2 - 6, y1 + 6), label, fill="black", font=font)

    # 基线
    draw.line(
        [margin_left, h - margin_bottom, w - 20, h - margin_bottom],
        fill="black",
        width=2,
    )
    img.save(path)


def generate_samples(out_dir: str | Path, num_samples: int = 32, seed: int = 42) -> list[dict]:
    """生成 num_samples 条 {image, question, answer} 样本，图片落盘，返回记录列表。"""
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    records: list[dict] = []
    q_types = list(_QUESTIONS.keys())

    for idx in range(num_samples):
        n_bars = rng.randint(3, 6)
        values = [rng.randint(1, 50) for _ in range(n_bars)]
        labels = _LABELS[:n_bars]
        q_type = q_types[idx % len(q_types)]

        img_name = f"sample_{idx:04d}.png"
        img_path = img_dir / img_name
        _draw_barchart(values, labels, img_path)

        if q_type == "max":
            answer = labels[values.index(max(values))]
        elif q_type == "min":
            answer = labels[values.index(min(values))]
        elif q_type == "sum":
            answer = str(sum(values))
        else:  # count
            answer = str(n_bars)

        records.append(
            {
                "id": idx,
                "image": str(img_path.relative_to(out_dir)).replace("\\", "/"),
                "question": _QUESTIONS[q_type],
                "answer": answer,
                "question_type": q_type,
            }
        )

    # 落盘 jsonl
    jsonl_path = out_dir / "samples.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def load_samples(data_dir: str | Path) -> list[dict]:
    """从 jsonl 读回样本，并补全 image 为绝对路径。"""
    data_dir = Path(data_dir)
    records = []
    with open(data_dir / "samples.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["image_path"] = str(data_dir / rec["image"])
                records.append(rec)
    return records


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/processed/mvp"
    recs = generate_samples(out)
    print(f"生成 {len(recs)} 条样本到 {out}")
    for r in recs[:3]:
        print(r)
