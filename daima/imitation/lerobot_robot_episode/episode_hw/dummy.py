# -*- coding: utf-8 -*-
"""Dummy 仿真后端:无硬件跑通 lerobot 数据管线用。

状态在 read 时按固定步长向目标缓动,叠加小噪声,模拟真实关节运动。
自由模式下可用 drag() 注入关节位移,模拟人手拖动示教。
"""

import time
from typing import List, Optional

import numpy as np

from .base import (
    DEFAULT_POSE,
    GRIPPER_MAX,
    GRIPPER_MIN,
    MOTOR_LIMITS_DEG,
    NUM_JOINTS,
    EpisodeArmInterface,
)

_LO = np.array([lim[0] for lim in MOTOR_LIMITS_DEG], dtype=float)
_HI = np.array([lim[1] for lim in MOTOR_LIMITS_DEG], dtype=float)


class DummyEpisodeArm(EpisodeArmInterface):
    def __init__(
        self,
        max_speed_deg_per_s: float = 45.0,
        noise_deg: float = 0.1,
    ) -> None:
        self.max_speed_deg_per_s = max_speed_deg_per_s
        self.noise_deg = noise_deg
        self._connected = False
        self._pos = np.array(DEFAULT_POSE, dtype=float)
        self._target = self._pos.copy()
        self._gripper = 0
        self._free_mode = False
        self._last_read = time.perf_counter()

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _step(self, dt: float) -> None:
        """按最大速度向目标缓动。"""
        step = self.max_speed_deg_per_s * dt
        delta = self._target - self._pos
        dist = float(np.linalg.norm(delta))
        if dist <= step:
            self._pos = self._target.copy()
        else:
            self._pos = self._pos + delta / dist * step
        np.clip(self._pos, _LO, _HI, out=self._pos)

    def read_angles(self) -> List[Optional[float]]:
        dt = time.perf_counter() - self._last_read
        self._last_read = time.perf_counter()
        self._step(dt)
        angles = self._pos + np.random.normal(0.0, self.noise_deg, NUM_JOINTS)
        return angles.tolist()

    def read_gripper(self) -> Optional[int]:
        return self._gripper

    def send_angles(
        self, angles: List[float], gripper: Optional[int] = None
    ) -> None:
        self._target = np.clip(np.asarray(angles, dtype=float), _LO, _HI)
        if gripper is not None:
            self._gripper = int(np.clip(gripper, GRIPPER_MIN, GRIPPER_MAX))

    def send_gripper(self, gripper: int) -> None:
        self._gripper = int(np.clip(gripper, GRIPPER_MIN, GRIPPER_MAX))

    def set_free_mode(self, enable: bool) -> None:
        self._free_mode = enable

    def emergency_stop(self) -> None:
        self._target = self._pos.copy()

    def home(self) -> None:
        self.send_angles(DEFAULT_POSE, 0)

    # ── 仿真辅助(仅 dummy 有) ────────────────────────────
    @property
    def free_mode(self) -> bool:
        return self._free_mode

    def drag(self, delta_degrees: List[float]) -> None:
        """仿真人手拖动:沿各关节注入位移(仅自由模式下有意义)。"""
        self._pos = np.clip(
            self._pos + np.asarray(delta_degrees, dtype=float), _LO, _HI
        )
        self._target = self._pos.copy()
