# -*- coding: utf-8 -*-
"""Episode1 机械臂的机器人配置。

注册名 "episode_follower",lerobot-record / lerobot-eval / lerobot-rollout 通过
`--robot.type=episode_follower` 选择本机器人。

后端选择:
- backend=dummy : 仿真后端,无硬件跑通数据管线
- backend=tcp   : 厂商上位机(localhost:12345)通信,开发用
- backend=can   : PEAK CAN 直驱(episode_controller_core wheel),生产用
"""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("episode_follower")
@dataclass(kw_only=True)
class EpisodeFollowerConfig(RobotConfig):
    # 硬件后端: dummy / tcp / can
    backend: str = "dummy"

    # can 后端: PCAN-USB 设备索引(与 robot_interface_node.py 的 usb_index 一致)
    usb_index: int = 1

    # tcp 后端: 厂商上位机地址与端口
    tcp_ip: str = "localhost"
    tcp_port: int = 12345

    # 相机(键名即数据集中的 camera 名),例如:
    #   cameras="{top: {type: synthetic, width: 320, height: 240, fps: 15}}"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
