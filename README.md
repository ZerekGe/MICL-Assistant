# QwenVL-MICL-Assistant

一个基于 **Qwen2.5-VL** 的多模态视觉语言助手，支持高清图像、多图输入与视频帧采样下的复杂推理（图表问答、文档理解）。核心特性是**检索式多模态上下文学习（Multimodal In-Context Learning, MICL）**：从标注数据中检索与当前查询最相关的图文示例作为 in-context demo，提升少样本场景下的推理准确率。

## 概览

给定一张图表/文档/视频帧和用户问题，检索最相关的示范问答对拼入提示，交给经 QLoRA 微调的 Qwen2.5-VL 生成答案；用 ChartQA / DocVQA 标准分数 + GPT-4o 裁判双轨评估。

## 核心特性

| 特性 | 说明 |
| --- | --- |
| 检索式 in-context demo 选择 | CLIP 图文相似度 + reward 重排，按查询动态挑选示例 |
| QLoRA 微调 + 投影层解冻 | 4bit 低秩适配，支持视觉投影层的消融对比 |
| 统一多模态预处理 | 动态分辨率、多图拼接、视频均匀帧采样 |
| 双轨评测 | ChartQA 精确率 + GPT-4o-as-judge |

## 目录结构

```
QwenVL-MICL-Assistant/
├── README.md               # 本文件
├── LICENSE                 # MIT 许可证
├── requirements.txt        # 依赖清单
├── configs/
│   └── default.yaml        # 训练 / 评测 / 检索超参
├── src/
│   ├── __init__.py
│   ├── utils.py            # 日志、随机种子、IO 工具
│   ├── retriever.py        # 检索式 demo 选择器
│   ├── data.py             # 数据集加载与预处理
│   ├── model.py            # 模型加载 + QLoRA 注入封装
│   └── mvp_data.py         # 冒烟测试样本生成器（合成图表）
├── scripts/
│   ├── download_model.py   # 模型下载到本地缓存
│   ├── prepare_data.py     # 数据准备 / 清洗 / demo 池构建
│   ├── train_qlora.py      # QLoRA 指令微调主入口
│   └── eval_chartqa.py     # ChartQA 评测 + GPT-4o judge
├── demo/
│   └── app.py              # Gradio 交互 Demo（图像/视频上传问答）
├── assets/                 # 架构图、结果截图、可视化
├── reports/                # 技术报告与实验记录
└── outputs/                # 训练权重 / 日志（已 gitignore）
```

## 环境安装

```bash
# 建议使用 conda 环境，Python 3.10+
conda create -n qwenvl-micl python=3.10 -y
conda activate qwenvl-micl

pip install -r requirements.txt
```

> 显存说明：12GB 显存（如 RTX 4070）用 3B 模型 + QLoRA 即可跑通；7B 训练与基准评测建议使用 24GB 以上显卡或多卡环境。

## 快速开始

```bash
# 1. 准备数据（构建 demo 检索池；冒烟阶段生成 32 条合成图表样本）
python scripts/prepare_data.py --num-samples 32

# 2. QLoRA 微调（1 个 step 冒烟测试；正式训练去掉 --max-steps）
python scripts/train_qlora.py --max-steps 1

# 3. 评测（ChartQA 精确率）
python scripts/eval_chartqa.py --num-samples 32

# 4. 启动交互 Demo
python demo/app.py --port 7860
```

> 模型从 ModelScope 加载，可用 `--model` / `--checkpoint` 指定本地模型路径。7B 基座的正式实验建议在大显存环境运行。

## 消融实验

| 配置 | ChartQA Acc | DocVQA Acc | GPT-4o judge |
| --- | --- | --- | --- |
| 零样本 Qwen2.5-VL-7B | — | — | — |
| + 随机 4-shot 示例 | — | — | — |
| + 检索式 4-shot | — | — | — |
| + LoRA 微调 r=16 | — | — | — |
| + 微调 + 检索 demo | — | — | — |

## 开发日志

- `[2026-11-18]` 项目初始化：搭建目录骨架、配置、依赖与脚本结构。
- `[2026-11-20]` MVP 闭环：数据生成、3B QLoRA 单步训练、评测与 Gradio Demo 全部跑通。

## License

MIT.
