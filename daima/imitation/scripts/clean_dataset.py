#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集清洗检测:帧率异常 / 运动模糊 / 手部遮掩(肤色),逐 episode 出体检报告。

用法:
    python scripts/clean_dataset.py <数据集路径> [--step 10] [--write-keep]

三个检测器:
1. 帧率异常:逐帧时间戳间隔偏离 1/fps 超过 50% 的帧占比(丢帧/卡顿);
2. 运动模糊:拉普拉斯清晰度低于阈值的帧占比(拖太快/曝光过长);
3. 手部遮掩:肤色像素(YCrCb)占比过高 —— 手入画;启发式,建议对剔除结果
   用 lerobot_dataset_viz 人工复核。

输出:
- 逐 episode 体检表 + 不合格原因;
- keep_list.json(可选,--write-keep):通过名单;
- 物理剔除命令提示(用 lerobot-edit-dataset delete_episodes 另存新数据集)。

注意:检测只是"筛掉坏条",救不了整体过暗/机位错误——若大多数 episode
都不合格,说明要重录而不是清洗(先跑 check_video_quality.py 看整体指标)。
"""

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field

import av
import numpy as np
import pandas as pd

# 默认阈值(可按数据集实测调整)
FPS_TOL_FRAC = 0.02      # 帧率异常帧占比上限(2%)
BLUR_LAP = 8.0           # 清晰度低于此值判为模糊帧
BLUR_TOL_FRAC = 0.5      # 模糊帧占比上限(50%)
SKIN_MEAN_LIMIT = 0.03   # 平均肤色占比上限(3%)
SKIN_MAX_LIMIT = 0.10    # 单帧肤色占比上限(10%,手凑近镜头时突增)

# 肤色阈值(YCrCb,经典区间)
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)


@dataclass
class EpisodeReport:
    idx: int
    frames: int = 0
    bright_mean: float = 0.0
    blur_frac: float = 0.0
    skin_mean: float = 0.0
    skin_max: float = 0.0
    fps_bad_frac: float = 0.0
    fps_max_gap: float = 0.0
    reasons: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def _rgb_to_ycbcr(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB(0-255 float) -> (Cb, Cr),与 OpenCV 公式一致。"""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return cb, cr


def analyze_fps(df: pd.DataFrame, fps: float) -> dict:
    """基于时间戳的逐 episode 帧率异常统计。"""
    out = {}
    expected = 1.0 / fps
    for idx, g in df.groupby("episode_index"):
        ts = np.asarray(g["timestamp"], dtype=float)
        if len(ts) < 2:
            out[int(idx)] = (0.0, 0.0)
            continue
        dt = np.diff(ts)
        bad = np.abs(dt - expected) > 0.5 * expected
        out[int(idx)] = (float(bad.mean()), float(dt.max()))
    return out


def analyze_frames(videos: list[str], episodes: list[int], step: int) -> dict:
    """抽帧统计:亮度 / 模糊 / 肤色,按 episode 聚合。

    视频为所有 episode 按序拼接,全局帧号 -> episode 由 episodes(长度表)换算。
    """
    off = np.concatenate([[0], np.cumsum(episodes)])  # 每 episode 起始全局帧号
    reports: dict[int, EpisodeReport] = {i: EpisodeReport(idx=i) for i in range(len(episodes))}
    global_idx = 0
    for path in videos:
        container = av.open(path)
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if global_idx % step == 0:
                ep = int(np.searchsorted(off, global_idx, side="right") - 1)
                if ep < len(reports):
                    img = np.asarray(frame.to_image().convert("RGB")).astype(float)
                    gray = img.mean(2)
                    lap = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
                    cb, cr = _rgb_to_ycbcr(img)
                    skin = ((cr > SKIN_CR[0]) & (cr < SKIN_CR[1]) & (cb > SKIN_CB[0]) & (cb < SKIN_CB[1])).mean()
                    r = reports[ep]
                    r.frames += 1
                    r.bright_mean += gray.mean()
                    if lap < BLUR_LAP:
                        r.blur_frac += 1
                    r.skin_mean += float(skin)
                    r.skin_max = max(r.skin_max, float(skin))
            global_idx += 1
    for r in reports.values():
        if r.frames:
            r.bright_mean /= r.frames
            r.blur_frac /= r.frames
            r.skin_mean /= r.frames
    return reports


def verdict(reports: dict, fps_stats: dict) -> None:
    """按阈值给每条 episode 下结论,原因写进 reasons。"""
    for idx, r in reports.items():
        bad_frac, max_gap = fps_stats.get(idx, (0.0, 0.0))
        r.fps_bad_frac, r.fps_max_gap = bad_frac, max_gap
        if bad_frac > FPS_TOL_FRAC:
            r.reasons.append(f"帧率异常 {bad_frac*100:.0f}%(最大间隔 {max_gap:.2f}s)")
        if r.blur_frac > BLUR_TOL_FRAC:
            r.reasons.append(f"运动模糊 {r.blur_frac*100:.0f}%")
        if r.skin_mean > SKIN_MEAN_LIMIT:
            r.reasons.append(f"手部遮掩(均值 {r.skin_mean*100:.1f}%)")
        elif r.skin_max > SKIN_MAX_LIMIT:
            r.reasons.append(f"手部遮掩(峰值 {r.skin_max*100:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集清洗检测:帧率/模糊/手部遮掩")
    ap.add_argument("dataset", help="数据集根目录")
    ap.add_argument("--step", type=int, default=10, help="视频抽帧间隔(默认 10)")
    ap.add_argument("--write-keep", action="store_true", help="保存 keep_list.json")
    args = ap.parse_args()

    root = args.dataset.rstrip("/")
    info = json.load(open(f"{root}/meta/info.json"))
    fps = float(info["fps"])
    ep_meta = pd.read_parquet(glob.glob(f"{root}/meta/episodes/**/*.parquet", recursive=True)[0])
    episodes = [int(x) for x in ep_meta.sort_values("episode_index")["length"]]
    print(f"数据集: {root}\nepisode 数: {len(episodes)} | 总帧: {sum(episodes)} | fps: {fps}\n")

    print("[1/2] 时间戳帧率分析...")
    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))]
    fps_stats = analyze_fps(pd.concat(dfs), fps)

    print("[2/2] 视频抽帧分析(亮度/模糊/肤色)...")
    videos = sorted(glob.glob(f"{root}/videos/**/*.mp4", recursive=True))
    reports = analyze_frames(videos, episodes, args.step)
    verdict(reports, fps_stats)

    bad = [r for r in reports.values() if not r.ok]
    keep = [r.idx for r in reports.values() if r.ok]
    print(f"\n{'ep':>3} {'帧':>5} {'亮度':>5} {'模糊%':>5} {'肤色均值%':>8} {'肤色峰值%':>8} {'帧率异常%':>8}  判定")
    for r in reports.values():
        mark = "OK " if r.ok else "FAIL"
        print(
            f"{r.idx:>3} {r.frames*args.step:>5} {r.bright_mean:>5.0f} {r.blur_frac*100:>5.0f} "
            f"{r.skin_mean*100:>8.1f} {r.skin_max*100:>8.1f} {r.fps_bad_frac*100:>8.1f}  {mark}"
        )
    print(f"\n结果: {len(keep)}/{len(episodes)} 条合格")
    for r in bad:
        print(f"  ep {r.idx}: " + "; ".join(r.reasons))

    if bad:
        idxs = "[" + ", ".join(str(r.idx) for r in bad) + "]"
        print("\n物理剔除命令(另存新数据集,不动原数据):")
        print(
            f"python -m lerobot.scripts.lerobot_edit_dataset \\\n"
            f"    --repo_id wujunqi/episode_drag_v1 \\\n"
            f"    --root {root} \\\n"
            f"    --new_repo_id wujunqi/episode_drag_v1_clean \\\n"
            f"    --new_root {root}_clean \\\n"
            f"    --operation.type delete_episodes \\\n"
            f"    --operation.episode_indices '{idxs}'"
        )
        print("\n或训练时只选合格条(不改数据):")
        print(f"    --dataset.episodes='[{', '.join(str(i) for i in keep)}]'")
    else:
        print("\n全部合格,无需清洗")

    if args.write_keep:
        json.dump({"keep": keep, "reject": {r.idx: r.reasons for r in bad}},
                  open("keep_list.json", "w"), ensure_ascii=False, indent=2)
        print("已写 keep_list.json")


if __name__ == "__main__":
    main()
