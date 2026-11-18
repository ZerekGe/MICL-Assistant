#!/usr/bin/env python3
"""Gradio 交互 Demo：上传图像/视频并实时问答。

用法：
    python demo/app.py --port 7860
    python demo/app.py --share    # 生成公网临时链接
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多模态视觉语言助手 Demo")
    p.add_argument("--port", type=int, default=7860, help="监听端口")
    p.add_argument("--share", action="store_true", help="生成 Gradio 公网链接")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    return p.parse_args()


def build_interface():
    """构建 Gradio Blocks 界面。"""
    import gradio as gr

    def answer_image(image, question):
        # TODO: 调用 model + retriever 生成答案
        return "（骨架阶段）模型回答占位：" + str(question)

    def answer_video(video, question):
        # TODO: sample_video_frames -> model 生成答案
        return "（骨架阶段）视频问答待实现：" + str(question)

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
    args = parse_args()
    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
