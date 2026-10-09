# -*- coding: utf-8 -*-
"""Episode 机械臂硬件后端包。

提供三个可互换后端:
- can_core:    CAN 直驱(episode_controller_core.MotorControl,生产用)
- tcp_server:  厂商上位机 TCP(localhost:12345,开发用)
- dummy:       仿真后端(无硬件跑通数据管线)
"""
from .base import (
    EpisodeArmInterface,
    NUM_JOINTS,
    STATE_DIM,
    ACTION_DIM,
    JOINT_OFFSETS,
    JOINT_LIMITS_DEG,
    MOTOR_LIMITS_DEG,
    SEND_LIMITS_DEG,
    GRIPPER_MIN,
    GRIPPER_MAX,
    DEFAULT_POSE,
)

from .dummy import DummyEpisodeArm

# 硬件后端按需导入:can_core 依赖闭源 wheel,tcp_server 依赖上位机进程
try:
    from .can_core import EpisodeCoreArm
except ImportError:  # 未安装 episode_controller_core wheel 时跳过
    EpisodeCoreArm = None  # type: ignore[assignment]

from .tcp_server import EpisodeTcpArm

__all__ = [
    "EpisodeArmInterface",
    "NUM_JOINTS",
    "STATE_DIM",
    "ACTION_DIM",
    "JOINT_OFFSETS",
    "JOINT_LIMITS_DEG",
    "MOTOR_LIMITS_DEG",
    "SEND_LIMITS_DEG",
    "GRIPPER_MIN",
    "GRIPPER_MAX",
    "DEFAULT_POSE",
    "DummyEpisodeArm",
    "EpisodeCoreArm",
    "EpisodeTcpArm",
]
