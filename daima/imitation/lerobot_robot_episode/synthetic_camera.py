# -*- coding: utf-8 -*-
"""合成相机:无硬件跑通 lerobot 数据管线用。

画面是机械臂的二维示意图,由当前关节角驱动 —— 录制出的视频中
机械臂会跟随示教轨迹运动,便于在数据集可视化里人工检查。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig

from .episode_hw import DEFAULT_POSE, GRIPPER_MAX

# 简化二维运动学(毫米):J2/J3/J4 构成平面肘链,J1 只影响基座指向
_L1 = 200.0  # 上臂 (DH a=200)
_L2 = 56.0 + 192.0  # 前臂 (DH a=56 + d=192)
_SCALE = 0.42  # 像素/毫米


@CameraConfig.register_subclass("synthetic")
@dataclass(kw_only=True)
class SyntheticCameraConfig(CameraConfig):
    # 画面主题:dark / light
    theme: str = "dark"


class SyntheticCamera(Camera):
    """由关节角驱动的合成 RGB 相机。"""

    def __init__(self, config: SyntheticCameraConfig):
        super().__init__(config)
        self.config = config
        self.use_rgb = True
        self.use_depth = False
        self._connected = False
        self._angles = list(DEFAULT_POSE)
        self._gripper = 0.0
        self._frame_id = 0

    # ── 帧内容 ────────────────────────────────────────────

    def set_joint_angles(self, angles: list[float], gripper: float) -> None:
        """由机器人每帧调用,更新渲染状态。"""
        self._angles = list(angles)
        self._gripper = float(gripper)

    def _render(self) -> np.ndarray:
        import cv2

        h, w = self.height, self.width
        self._frame_id += 1

        if self.config.theme == "light":
            img = np.full((h, w, 3), 245, dtype=np.uint8)
            fg, dim = (30, 30, 30), (120, 120, 120)
        else:
            img = np.full((h, w, 3), 25, dtype=np.uint8)
            fg, dim = (240, 240, 240), (90, 90, 90)

        # J1(基座旋转)映射为画面色相,保证视频逐帧有可见变化
        j1 = self._angles[0] - 180.0  # -360..0
        hue_shift = int((j1 % 360) * 0.5)
        tint = np.full((h, w, 3), hue_shift, dtype=np.uint8)

        # 二维肘链:J2 仰角(90=水平),J3 肘(83=伸直),J4 腕
        base = np.array([w // 2, h - 60], dtype=float)
        th1 = np.deg2rad(self._angles[1] - 90.0)
        th2 = np.deg2rad(self._angles[2] - 83.0)
        th3 = np.deg2rad(self._angles[3] - 30.0)
        p1 = base + _L1 * _SCALE * np.array([np.cos(th1), -np.sin(th1)])
        p2 = p1 + _L2 * _SCALE * np.array([np.cos(th1 + th2), -np.sin(th1 + th2)])
        p3 = p2 + 20 * np.array([np.cos(th1 + th2 + th3), -np.sin(th1 + th2 + th3)])

        # 基座圆台 + 三连杆
        cv2.circle(img, tuple(base.astype(int)), 18, dim, -1)
        cv2.ellipse(img, tuple(base.astype(int)), (18, 18), 0, 0, 360, dim, 2)
        for a, b in ((base, p1), (p1, p2), (p2, p3)):
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), fg, 6)
        for p in (base, p1, p2, p3):
            cv2.circle(img, tuple(p.astype(int)), 7, fg, -1)

        # 夹爪:开度映射为两指间距
        open_px = int(4 + self._gripper / GRIPPER_MAX * 14)
        g_dir = np.array([np.cos(th1 + th2 + th3 + np.pi / 2),
                          -np.sin(th1 + th2 + th3 + np.pi / 2)])
        for s in (-1, 1):
            tip = p3 + s * open_px * g_dir
            cv2.line(img, tuple(p3.astype(int)), tuple(tip.astype(int)), fg, 4)

        # 工作台 + 目标块(位置随 J1 微动,颜色随夹爪)
        table_y = h - 30
        cv2.line(img, (0, table_y), (w, table_y), dim, 2)
        block_x = w // 2 + int(j1 * 0.25)
        block_color = (60, 200, 120) if self._gripper > GRIPPER_MAX / 2 else (60, 120, 220)
        cv2.rectangle(img, (block_x - 12, table_y - 22), (block_x + 12, table_y), block_color, -1)

        # 状态栏
        text = " | ".join(
            f"j{i}:{self._angles[i - 1]:7.1f}" for i in (1, 2, 3)
        ) + f" | grip:{self._gripper:5.1f} | f{self._frame_id}"
        cv2.putText(img, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, fg, 1)
        cv2.putText(img, "SYNTH", (w - 60, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim, 1)

        # cv2 是 BGR 绘制,lerobot 数据集约定存 RGB
        return cv2.add(img, tint)[..., ::-1].copy()

    # ── Camera 接口 ────────────────────────────────────────

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return [{"type": "synthetic", "index": 0}]

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, warmup: bool = True) -> None:
        self._connected = True

    def read(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("synthetic camera 未连接")
        return self._render()

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self.read()

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        return self.read()

    def disconnect(self) -> None:
        self._connected = False
