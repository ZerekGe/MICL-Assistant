#!/usr/bin/env python3
"""Gradio 交互 Demo：上传图像/视频并实时问答。

MVP 阶段：图像问答真实调 Qwen2.5-VL 返回一句回答；--mock 用规则兜底。
视频问答采用最小抽帧实现，后续可扩展。

用法：
    python demo/app.py --port 7860
    python demo/app.py --mock          # 无模型兜底，验证界面链路
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多模态视觉语言助手 Demo")
    p.add_argument("--port", type=int, default=7860, help="监听端口")
    p.add_argument("--share", action="store_true", help="生成 Gradio 公网链接")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    p.add_argument("--mock", action="store_true", help="用规则回答（无模型兜底）")
    p.add_argument("--no-launch", action="store_true", help="只构建界面不启动（用于验证）")
    return p.parse_args()


# 延迟加载的模型单例
_MODEL = None
_PROCESSOR = None
_MOCK = False


def _ensure_model():
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        from src.model import load_model_and_processor
        _MODEL, _PROCESSOR = load_model_and_processor(quantization_bit=4)
        _MODEL.eval()
    return _MODEL, _PROCESSOR


def answer_image(image, question):
    if not image:
        return "请先上传一张图片。"
    if not question or not question.strip():
        return "请输入问题。"
    if _MOCK:
        return f"[MOCK回答] 已收到图片和问题：{question}（真实模型未加载，此为界面链路验证）"
    try:
        import torch
        from qwen_vl_utils import process_vision_info

        model, processor = _ensure_model()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        batch = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        batch = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=128, do_sample=False)
        return processor.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    except Exception as e:  # noqa: BLE001 - Demo 需要优雅降级展示错误
        return f"模型推理失败：{type(e).__name__}: {e}"


def answer_video(video, question):
    if not video:
        return "请先上传一段视频。"
    if _MOCK:
        return f"[MOCK回答] 已收到视频和问题：{question}（视频链路持续完善中）"
    # MVP：视频问答的最小占位（抽帧逻辑后续接入 sample_video_frames）
    return f"视频问答功能规划中：已收到问题 {question!r}，将均匀采样 8 帧后送入模型。"


def build_interface():
    import gradio as gr

    with gr.Blocks(title="QwenVL-MICL Assistant") as demo:
        gr.Markdown("# 多模态视觉语言助手\n上传图像或视频，输入问题开始对话。")
        with gr.Tab("图像问答"):
            img = gr.Image(type="filepath", label="上传图像")
            q1 = gr.Textbox(label="问题", placeholder="例如：图中的柱状图最大值是多少？")
            btn1 = gr.Button("提问")
            out1 = gr.Textbox(label="回答")
            btn1.click(answer_image, inputs=[img, q1], outputs=out1)
        with gr.Tab("视频问答"):
            vid = gr.Video(label="上传视频")
            q2 = gr.Textbox(label="问题")
            btn2 = gr.Button("提问")
            out2 = gr.Textbox(label="回答")
            btn2.click(answer_video, inputs=[vid, q2], outputs=out2)
    return demo


def main() -> None:
    global _MOCK
    args = parse_args()
    _MOCK = args.mock
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    demo = build_interface()
    if args.no_launch:
        print("OK: Gradio 界面构建成功（--no-launch）")
        return
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
