#!/usr/bin/env python3
"""从 ModelScope 下载 Qwen2.5-VL 模型到本地缓存（HF 在本环境不可达）。

用法：
    python scripts/download_model.py
    python scripts/download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct
"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载模型到 D 盘缓存")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--cache-dir", type=str, default="D:/caches/modelscope")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from modelscope import snapshot_download

    print(f"开始下载 {args.model} -> {args.cache_dir}", flush=True)
    t0 = time.time()
    path = snapshot_download(args.model, cache_dir=args.cache_dir)
    print(f"下载完成，耗时 {time.time() - t0:.1f}s", flush=True)
    print(f"LOCAL_PATH={path}", flush=True)

    # 写一份路径标记，供其他脚本/日志引用
    from pathlib import Path

    marker = Path("outputs/model_path.txt")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(path, encoding="utf-8")
    print(f"已记录到 {marker}", flush=True)


if __name__ == "__main__":
    main()
