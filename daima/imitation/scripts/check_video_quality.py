#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""录制数据画面质量自查:亮度 / 清晰度 / 目标(红块)可见性与位置。

用法:
    python scripts/check_video_quality.py <数据集根目录或 mp4 路径>

说明:
- 抽帧(默认每 100 帧取 1)统计三件事:
  1. 亮度:均值过低(<100)说明欠曝/环境光不足,ACT 视觉特征弱;
  2. 清晰度:拉普拉斯均值过低(<20)说明运动模糊,拖动示教太快;
  3. 红块:可见率应 ~100%,且质心应落在画面中央区(不贴边)。
     贴边/出画 = 相机没取全工作台,重摆相机后再录。
- 阈值可按任务调整;本脚本只读视频,不改任何数据。
"""

import argparse
import glob
import sys

import av
import numpy as np
from PIL import Image

BRIGHT_OK = 100.0   # 亮度均值合格线
SHARP_OK = 20.0     # 清晰度(拉普拉斯)合格线
RED_MIN_PIX = 200   # 判定"红块可见"的最少红色像素数


def analyze(video_path: str, step: int = 100) -> dict:
    container = av.open(video_path)
    stream = container.streams.video[0]
    bright, sharp, red_vis, red_edge = [], [], 0, 0
    total, red_frames = 0, 0
    for frame in container.decode(stream):
        if total % step == 0:
            im = np.asarray(frame.to_image().convert("RGB")).astype(float)
            gray = im.mean(2)
            lap = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
            r, g, b = im[..., 0], im[..., 1], im[..., 2]
            red = (r > 100) & (r - g > 50) & (r - b > 50)
            cy, cx = np.where(red)
            bright.append(gray.mean())
            sharp.append(lap)
            if len(cx) > RED_MIN_PIX:
                red_vis += 1
                red_frames += 1
                h, w = im.shape[:2]
                if cy.mean() > h - 80 or cx.mean() > w - 80:
                    red_edge += 1
        total += 1
    n = len(bright)
    return {
        "frames": total,
        "sampled": n,
        "bright_mean": float(np.mean(bright)),
        "sharp_mean": float(np.mean(sharp)),
        "red_visible_ratio": red_vis / n,
        "red_edge_ratio": red_edge / max(red_frames, 1),
    }


def merge_stats(stats: list[dict]) -> dict:
    """合并多个视频分片的统计(按抽样数加权)。"""
    n = sum(s["sampled"] for s in stats)
    return {
        "frames": sum(s["frames"] for s in stats),
        "sampled": n,
        "bright_mean": sum(s["bright_mean"] * s["sampled"] for s in stats) / n,
        "sharp_mean": sum(s["sharp_mean"] * s["sampled"] for s in stats) / n,
        "red_visible_ratio": sum(s["red_visible_ratio"] * s["sampled"] for s in stats) / n,
        "red_edge_ratio": sum(
            s["red_edge_ratio"] * s["red_visible_ratio"] * s["sampled"]
            for s in stats
        ) / max(sum(s["red_visible_ratio"] * s["sampled"] for s in stats), 1e-9),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="录制数据画面质量自查")
    ap.add_argument("path", help="数据集根目录或 mp4 路径")
    ap.add_argument("--step", type=int, default=100, help="抽帧间隔(默认 100)")
    args = ap.parse_args()

    videos = [args.path] if args.path.endswith(".mp4") else sorted(
        glob.glob(f"{args.path}/videos/**/*.mp4", recursive=True)
    )
    if not videos:
        print(f"在 {args.path} 下未找到 mp4,请传入数据集根目录或视频文件路径")
        sys.exit(1)
    for i, v in enumerate(videos):
        print(f"视频分片 [{i}]: {v}")

    s = merge_stats([analyze(v, args.step) for v in videos])
    print(f"总帧 {s['frames']},抽样 {s['sampled']} 帧")
    print(f"亮度均值   {s['bright_mean']:6.1f}/255   (合格线 >{BRIGHT_OK})")
    print(f"清晰度     {s['sharp_mean']:6.1f}       (合格线 >{SHARP_OK})")
    print(f"红块可见率 {s['red_visible_ratio']*100:6.0f}%      (目标 ~100%)")
    print(f"可见时贴边 {s['red_edge_ratio']*100:6.0f}%      (目标 0%,贴边=相机没取全工作台)")

    problems = []
    if s["bright_mean"] < BRIGHT_OK:
        problems.append("画面过暗:补光/打开自动曝光")
    if s["sharp_mean"] < SHARP_OK:
        problems.append("运动模糊严重:示教拖慢,尤其接近目标时")
    if s["red_visible_ratio"] < 0.9:
        problems.append("红块经常不可见:相机视野没覆盖整个工作台,重摆相机")
    if s["red_edge_ratio"] > 0.1:
        problems.append("红块常贴画面边缘:抬高/后移相机,目标应在画面中央区")
    if problems:
        print("\n结论: 不合格 -> " + "; ".join(problems))
        sys.exit(1)
    print("\n结论: 画面质量合格")


if __name__ == "__main__":
    main()
