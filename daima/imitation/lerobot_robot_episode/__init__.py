# -*- coding: utf-8 -*-
"""Episode1 机械臂的 lerobot 插件包。

lerobot 通过 dist 名前缀(lerobot_robot_)发现本插件并导入,
导入即完成 draccus 配置注册:
- EpisodeFollowerConfig  ->  --robot.type=episode_follower
- SyntheticCameraConfig  ->  cameras.{name}.type=synthetic

实现文件:
- episode_follower.py : Robot 子类,包装 episode_hw 三个后端
- synthetic_camera.py : 合成相机(无硬件跑通数据管线用)
- episode_hw/        : 硬件后端抽象(dummy / tcp / can)
"""

from .config_episode import EpisodeFollowerConfig
from .episode_follower import EpisodeFollower
from .synthetic_camera import SyntheticCamera, SyntheticCameraConfig

__all__ = [
    "EpisodeFollower",
    "EpisodeFollowerConfig",
    "SyntheticCamera",
    "SyntheticCameraConfig",
]
