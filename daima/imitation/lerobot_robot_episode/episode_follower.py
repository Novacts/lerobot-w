# -*- coding: utf-8 -*-
"""Episode1 机械臂的 Robot 实现(lerobot 插件)。

状态/动作空间:7 维 = 6 关节电机角(度)+ 1 夹爪角度(录制用 0-75 度,硬件上限 110)。
命名 j1.pos..j6.pos / gripper.pos,数据集特征为 observation.state / action。

硬件后端在 episode_hw 中抽象(dummy / tcp / can),本类只依赖抽象接口。
"""

import logging
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_episode import EpisodeFollowerConfig
from .episode_hw import (
    DEFAULT_POSE,
    GRIPPER_MAX,
    GRIPPER_MIN,
    NUM_JOINTS,
    SEND_LIMITS_DEG,
    DummyEpisodeArm,
    EpisodeArmInterface,
    EpisodeCoreArm,
    EpisodeTcpArm,
)
from .synthetic_camera import SyntheticCamera

logger = logging.getLogger(__name__)

JOINT_NAMES = [f"j{i}.pos" for i in range(1, NUM_JOINTS + 1)]
GRIPPER_NAME = "gripper.pos"
MOTOR_NAMES = JOINT_NAMES + [GRIPPER_NAME]

# 下发钳位用名义限位(controller_core 只认名义限位,见 base.SEND_LIMITS_DEG)
_LO = [lim[0] for lim in SEND_LIMITS_DEG]
_HI = [lim[1] for lim in SEND_LIMITS_DEG]


def make_episode_backend(config: EpisodeFollowerConfig) -> EpisodeArmInterface:
    """按配置构造硬件后端。"""
    backend = config.backend.lower()
    if backend == "dummy":
        return DummyEpisodeArm()
    if backend == "tcp":
        return EpisodeTcpArm(ip=config.tcp_ip, port=config.tcp_port)
    if backend == "can":
        return EpisodeCoreArm(usb_index=config.usb_index)
    raise ValueError(
        f"未知后端 '{config.backend}',可选: dummy / tcp / can"
    )


class EpisodeFollower(Robot):
    """Episode1 机械臂(lerobot Robot 兼容)。

    lerobot 式标定(calibrate)对 Episode1 不适用(回零是硬件动作),恒视为已标定。
    connect() 只建立通信,不动电机;回零需人工确认后调用 `robot.arm.home()`。
    """

    config_class = EpisodeFollowerConfig
    name = "episode_follower"

    def __init__(self, config: EpisodeFollowerConfig):
        super().__init__(config)
        self.config = config
        self.arm: EpisodeArmInterface = make_episode_backend(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        # 读数兜底:硬件阻塞/缺失时沿用上一帧
        self._last_angles = list(DEFAULT_POSE)
        self._last_gripper = 0

    @cached_property
    def observation_features(self) -> dict:
        features = {name: float for name in MOTOR_NAMES}
        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                features[cam_key] = (cam.height, cam.width, 3)
            if getattr(cam, "use_depth", False):
                features[f"{cam_key}_depth"] = (cam.height, cam.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict:
        return {name: float for name in MOTOR_NAMES}

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        # Episode1 无 lerobot 式标定流程(回零是独立硬件操作)
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.arm.connect()
        for cam in self.cameras.values():
            cam.connect()
        logger.info("%s connected (backend=%s)", self, self.config.backend)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        angles = self.arm.read_angles()
        # CAN 阻塞时个别元素为 None -> 沿用上一帧
        fixed = []
        for i, a in enumerate(angles):
            if a is None:
                fixed.append(self._last_angles[i])
            else:
                fixed.append(float(a))
        self._last_angles = fixed

        gripper = self.arm.read_gripper()
        if gripper is None:
            gripper = self._last_gripper
        else:
            gripper = int(gripper)
            self._last_gripper = gripper

        obs: RobotObservation = {}
        for name, value in zip(JOINT_NAMES, fixed):
            obs[name] = value
        obs[GRIPPER_NAME] = float(gripper)

        for cam_key, cam in self.cameras.items():
            # 合成相机依赖当前关节角渲染画面
            if isinstance(cam, SyntheticCamera):
                cam.set_joint_angles(fixed, gripper)
            if getattr(cam, "use_rgb", True):
                obs[cam_key] = cam.read_latest()
            if getattr(cam, "use_depth", False):
                obs[f"{cam_key}_depth"] = cam.read_latest_depth()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        angles = []
        for name in JOINT_NAMES:
            if name not in action:
                raise KeyError(f"动作缺少关节 {name}")
            value = float(action[name])
            value = max(_LO[len(angles)], min(_HI[len(angles)], value))
            angles.append(value)

        gripper = None
        if GRIPPER_NAME in action and action[GRIPPER_NAME] is not None:
            gripper = int(max(GRIPPER_MIN, min(GRIPPER_MAX, action[GRIPPER_NAME])))
            self._last_gripper = gripper

        self.arm.send_angles(angles, gripper=gripper)

        sent: RobotAction = {
            name: value for name, value in zip(JOINT_NAMES, angles)
        }
        sent[GRIPPER_NAME] = float(
            self._last_gripper if gripper is None else gripper
        )
        return sent

    @check_if_not_connected
    def send_gripper(self, value: float) -> None:
        """直控夹爪并同步回读兜底值(录制键控用,保持 obs/action 一致)。"""
        gripper = int(max(GRIPPER_MIN, min(GRIPPER_MAX, value)))
        self.arm.send_gripper(gripper)
        self._last_gripper = gripper

    @check_if_not_connected
    def disconnect(self) -> None:
        self.arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected", self)
