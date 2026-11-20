#!/usr/bin/env python3
"""评测：ChartQA 精确率 + 可选 GPT-4o 裁判（MVP 阶段用合成样本）。

真实模式用 Qwen2.5-VL 生成答案并算准确率；--mock 用规则返回一个可复现的
准确率数字，验证评测→写文件的链路。

用法：
    python scripts/eval_chartqa.py --config configs/default.yaml
    python scripts/eval_chartqa.py --mock --output outputs/eval_result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多模态助手评测")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    p.add_argument("--checkpoint", type=str, default=None, help="待评测的权重路径")
    p.add_argument("--judge", action="store_true", help="启用 GPT-4o-as-judge（MVP 跳过）")
    p.add_argument("--output", type=str, default="outputs/eval_result.json", help="结果输出路径")
    p.add_argument("--data-dir", type=str, default="data/processed/mvp", help="样本目录")
    p.add_argument("--num-samples", type=int, default=32, help="评测样本数")
    p.add_argument("--mock", action="store_true", help="用规则生成结果（无模型兜底）")
    return p.parse_args()


def normalize_answer(s: str) -> str:
    return str(s).strip().lower().replace(",", "")


# 英文数词 → 数字，用于 count/sum 题型的答案抽取
_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def extract_answer(pred: str, question_type: str) -> str:
    """从模型的自然语言回答中抽取规范化答案，再与 gold 比对。

    MVP 的合成图表有 4 种题型：
      - max/min：答案是类别字母 A-F
      - sum/count：答案是数字（模型可能用英文数词或 「38 + 2」形式）
    抽取失败时回退到整句规范化，保证不崩。
    """
    import re

    text = str(pred).strip()
    q = (question_type or "").lower()

    if q in ("max", "min"):
        # 取回答中第一个独立的 A-F 大写字母（"largest value is B" → B）
        m = re.search(r"\b([A-F])\b", text)
        if m:
            return m.group(1)
    elif q in ("count", "sum"):
        # 先看英文数词
        for word, digit in _WORD_NUM.items():
            if re.search(rf"\b{word}\b", text, re.IGNORECASE):
                return digit
        # 再看阿拉伯数字，取第一个
        m = re.search(r"\d+", text.replace(",", ""))
        if m:
            return m.group(0)

    return normalize_answer(text)


def is_correct(pred: str, gold: str, question_type: str) -> bool:
    return normalize_answer(extract_answer(pred, question_type)) == normalize_answer(gold)


def run_real(args, cfg, logger, samples) -> list[dict]:
    import torch
    from src.model import load_model_and_processor

    model_id = args.checkpoint or cfg["model"]["name_or_path"]
    logger.info("加载模型: %s", model_id)
    model, processor = load_model_and_processor(model_id, quantization_bit=4)
    model.eval()

    results = []
    for rec in samples:
        # 构造仅含问题的输入（不带 answer），直接喂图片绕开 qwen_vl_utils/torchvision
        from PIL import Image

        image = Image.open(rec["image_path"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": rec["image_path"]},
                {"type": "text", "text": rec["question"]},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = processor(
            text=[text], images=[image],
            padding=True, return_tensors="pt",
        )
        batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=16, do_sample=False)
        pred = processor.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        qtype = rec.get("question_type", "")
        extracted = extract_answer(pred, qtype)
        correct = is_correct(pred, rec["answer"], qtype)
        results.append({
            "id": rec["id"], "pred": pred.strip(), "gold": rec["answer"],
            "extracted": extracted, "question_type": qtype, "correct": correct,
        })
    return results


def run_mock(args, cfg, logger, samples) -> list[dict]:
    """规则版：直接把 gold 当 pred（证明评测计算与写文件链路，非真实模型能力）。"""
    logger.info("[MOCK] 用规则（gold→pred）验证评测链路，非真实模型准确率")
    results = []
    for i, rec in enumerate(samples):
        # 故意让每 5 条错 1 条，得到可复现的非满分数字
        pred = rec["answer"] if i % 5 != 4 else "__WRONG__"
        results.append({
            "id": rec["id"], "pred": pred, "gold": rec["answer"],
            "correct": normalize_answer(pred) == normalize_answer(rec["answer"]),
        })
    return results


def main() -> None:
    args = parse_args()
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.mvp_data import load_samples
    from src.utils import get_logger, load_config

    logger = get_logger("eval_chartqa")
    cfg_path = root / args.config if (root / args.config).exists() else args.config
    cfg = load_config(cfg_path)
    samples = load_samples(root / args.data_dir)[: args.num_samples]
    logger.info("评测样本: %d 条", len(samples))

    results = run_mock(args, cfg, logger, samples) if args.mock else run_real(args, cfg, logger, samples)

    acc = sum(r["correct"] for r in results) / len(results) if results else 0.0
    summary = {
        "mode": "mock" if args.mock else "real",
        "dataset": cfg["data"]["eval_dataset"],
        "num_samples": len(results),
        "accuracy": round(acc, 4),
        "metric": "exact_match",
        "per_sample": results,
    }
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("评测完成: accuracy=%.4f (%d/%d)", acc, sum(r["correct"] for r in results), len(results))
    logger.info("结果已写入: %s", out_path)


if __name__ == "__main__":
    main()
