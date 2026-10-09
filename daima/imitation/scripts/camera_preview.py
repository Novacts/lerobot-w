#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RealSense 实时预览:摆机位 / 调光辅助。

窗口叠加:
- 中央十字 + 建议工作区框(画面中央 70%):红块、盒子、机械臂运动范围
  应全程落在框内(尤其抓取时刻);
- 实时亮度(目标 >100)与清晰度(目标 >20,静止时):调补光灯和曝光用;
- 键盘:q / ESC 退出,s 冻结画面便于仔细看。

用法:
    python scripts/camera_preview.py [--serial 239722073962] [--width 640 --height 480 --fps 30]
"""

import argparse

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser(description="RealSense 实时预览(摆机位/调光)")
    ap.add_argument("--serial", default="239722073962")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipe.start(cfg)
    print(f"预览中: {args.serial} {args.width}x{args.height}@{args.fps} | q/ESC 退出, s 冻结")

    frozen = None
    while True:
        if frozen is None:
            frames = pipe.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
        else:
            img = frozen.copy()
        gray = img.astype(float).mean(2)
        lap = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        bright = gray.mean()

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        # 中央 70% 工作区框
        x0, x1, y0, y1 = int(w * 0.15), int(w * 0.85), int(h * 0.15), int(h * 0.85)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.line(img, (cx - 10, cy), (cx + 10, cy), (0, 255, 0), 1)
        cv2.line(img, (cx, cy - 10), (cx, cy + 10), (0, 255, 0), 1)
        cv2.putText(img, "work area: keep target inside", (x0 + 4, y0 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        color = (0, 255, 0) if bright > 100 else (0, 0, 255)
        cv2.putText(img, f"bright {bright:5.1f} (target >100)  sharp {lap:4.1f} (target >20)",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow("camera preview", img)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            frozen = img.copy() if frozen is None else None
    pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
