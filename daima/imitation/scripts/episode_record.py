#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Episode1 拖动示教录制脚本(lerobot 数据集格式)。

与 lerobot-record 的区别:
- 拖动示教(kinesthetic teaching):机械臂进入自由模式,人手直接拖动机器臂,
  每个采样帧的「动作」= 拖到位的当前关节角 + 键盘控制的夹爪值。
- 不调用 robot.send_action(自由模式下关节不应被下发力矩)。
- 夹爪用键盘控制:c 闭合 / o 张开 / g 切换。
- 键位:n 结束本条 episode 并存盘 / r 重录本条 / q 退出 / ESC 退出。

用法(无硬件冒烟):
    python scripts/episode_record.py \
        --robot.type=episode_follower \
        --robot.backend=dummy \
        --robot.cameras="{top: {type: synthetic, width: 320, height: 240, fps: 15}}" \
        --dataset.repo_id=wujunqi/episode_smoke \
        --dataset.root=/media/.../datasets \
        --dataset.fps=15 \
        --dataset.episode_time_s=10 \
        --dataset.reset_time_s=3 \
        --dataset.num_episodes=2 \
        --dataset.single_task="拖动示教演示" \
        --dataset.push_to_hub=false \
        --trajectory=circle

真机(拖动示教):
    --robot.backend=can --robot.usb_index=1 --drag_mode=true

数据集录制完毕后可视化:
    python -m lerobot.scripts.lerobot_dataset_viz --repo-id <repo_id>
"""

import logging
import os
import queue
import select
import sys
import termios
import threading
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  注册 --robot.cameras.{name}.type=opencv
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401  注册 type=intelrealsense
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from lerobot_robot_episode import EpisodeFollower, EpisodeFollowerConfig  # noqa: F401
from lerobot_robot_episode.episode_follower import GRIPPER_NAME, JOINT_NAMES, MOTOR_NAMES
from lerobot_robot_episode.episode_hw import DEFAULT_POSE

OBS_STR = "observation"
ACTION = "action"

# 键盘语义(拖动示教时人手在机械臂上,键位要少而明确)
KEY_QUIT = {"q", "\x1b"}          # 退出(ESC)
KEY_NEXT = {"n"}                  # 结束当前 episode 并存盘
KEY_RERECORD = {"r"}              # 丢弃当前 episode 重录
KEY_GRIP_CLOSE = {"c"}            # 夹爪闭合
KEY_GRIP_OPEN = {"o"}             # 夹爪张开
KEY_GRIP_TOGGLE = {"g"}           # 夹爪开/合切换

# 夹爪目标角(度):闭合 0,张开 75(硬件行程上限 110,张开调小以适配抓取)
GRIPPER_CLOSED = 0.0
GRIPPER_OPEN = 75.0


class TtyKeyListener(threading.Thread):
    """非阻塞终端按键读取(termios 原始模式)。

    拖动示教时 pynput 全局监听在 Wayland/SSH 下不可用,而终端读取
    在任何终端环境(含 SSH)都可用,故自实现。
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.tty_ok = sys.stdin.isatty()

    def run(self) -> None:
        if not self.tty_ok:
            return
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~(termios.ICANON | termios.ECHO)
        new[6][termios.VMIN] = 0
        new[6][termios.VTIME] = 0
        try:
            termios.tcsetattr(fd, termios.TCSANOW, new)
            while not self._stop.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    try:
                        ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    except OSError:
                        break
                    if ch:
                        self._queue.put(ch)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old)

    def drain(self) -> set[str]:
        """返回自上次 drain 以来按下的键。"""
        pressed = set()
        while not self._queue.empty():
            try:
                pressed.add(self._queue.get_nowait())
            except queue.Empty:
                break
        return pressed

    def stop(self) -> None:
        self._stop.set()


@dataclass
class EpisodeRecordConfig:
    # 声明为基类,由 --robot.type=episode_follower 在解析时分发
    robot: RobotConfig
    dataset: DatasetRecordConfig

    # 进入自由模式拖动示教(真机必须人工托举机械臂后再开启!)
    drag_mode: bool = True

    # 仅 dummy 后端:模拟人手拖动的脚本轨迹 none / still / circle
    trajectory: str = "none"

    play_sounds: bool = False

    def __post_init__(self):
        if self.robot.type != "episode_follower":
            raise ValueError(
                f"本脚本只支持 episode_follower,收到 {self.robot.type!r}"
            )
        if self.trajectory != "none" and self.robot.backend != "dummy":
            raise ValueError("--trajectory 仅在 dummy 后端可用(真机请人工拖动)")


def _process_keys(keys: set[str], robot: EpisodeFollower, gripper_cmd: float) -> tuple[str | None, float]:
    """处理按键,返回 (episode 事件, 更新后的夹爪指令)。

    episode 事件: None / "next" / "rerecord" / "quit"
    """
    event = None
    for key in keys:
        if key in KEY_QUIT:
            event = "quit"
        elif key in KEY_NEXT:
            event = "next"
        elif key in KEY_RERECORD:
            event = "rerecord"
        elif key in KEY_GRIP_CLOSE:
            gripper_cmd = GRIPPER_CLOSED
            robot.send_gripper(GRIPPER_CLOSED)
            logging.info("夹爪: 闭合")
        elif key in KEY_GRIP_OPEN:
            gripper_cmd = GRIPPER_OPEN
            robot.send_gripper(GRIPPER_OPEN)
            logging.info("夹爪: 张开")
        elif key in KEY_GRIP_TOGGLE:
            gripper_cmd = (
                GRIPPER_CLOSED
                if gripper_cmd > (GRIPPER_CLOSED + GRIPPER_OPEN) / 2
                else GRIPPER_OPEN
            )
            robot.send_gripper(gripper_cmd)
            logging.info(
                "夹爪: 切换 -> %s",
                "闭合" if gripper_cmd == GRIPPER_CLOSED else "张开",
            )
    return event, gripper_cmd


def _scripted_drag(robot: EpisodeFollower, phase: float, trajectory: str, gripper_cmd: float) -> float:
    """dummy 后端专用:注入正弦拖动,模拟人手示教轨迹。"""
    arm = robot.arm
    if not hasattr(arm, "drag"):
        return gripper_cmd
    t = phase
    if trajectory == "circle":
        # 围绕默认位姿画圈(幅度内联于限位内)
        arm.drag([
            25.0 * np_sin(t),
            15.0 * np_sin(t * 1.3 + 1.0),
            10.0 * np_sin(t * 0.7 + 2.0),
            0.0,
            0.0,
            0.0,
        ])
        gripper_cmd = (GRIPPER_OPEN / 2) * (1 + np_sin(t * 0.5))
        robot.send_gripper(gripper_cmd)
    return gripper_cmd


def np_sin(x: float) -> float:
    import numpy as np

    return float(np.sin(x))


def record_loop(
    robot: EpisodeFollower,
    listener: TtyKeyListener,
    dataset: LeRobotDataset,
    features: dict,
    fps: int,
    episode_time_s: float,
    task: str,
    drag_mode: bool,
    trajectory: str,
) -> tuple[str, float]:
    """单条 episode 的采样循环。返回 (退出事件, 当前夹爪指令)。"""
    control_interval = 1.0 / fps
    start = time.perf_counter()
    gripper_cmd = 0.0
    phase = 0.0

    while True:
        loop_t = time.perf_counter()
        timestamp = loop_t - start
        if episode_time_s > 0 and timestamp >= episode_time_s:
            return "next", gripper_cmd

        phase += control_interval
        gripper_cmd = _scripted_drag(robot, phase, trajectory, gripper_cmd)

        obs = robot.get_observation()
        obs_frame = build_dataset_frame(features, obs, prefix=OBS_STR)

        # 拖动示教: 动作 = 拖到位的当前关节角 + 键盘夹爪指令
        act = {name: obs[name] for name in MOTOR_NAMES}
        act[GRIPPER_NAME] = float(gripper_cmd)
        action_frame = build_dataset_frame(features, act, prefix=ACTION)

        dataset.add_frame({**obs_frame, **action_frame, "task": task})

        event, gripper_cmd = _process_keys(listener.drain(), robot, gripper_cmd)
        if event is not None:
            return event, gripper_cmd

        dt = time.perf_counter() - loop_t
        if dt > control_interval:
            logging.warning(
                "录制循环慢于目标 FPS: %.1f Hz < %d Hz",
                1.0 / dt,
                fps,
            )
        precise_sleep(max(control_interval - dt, 0.0))


def _wait_reset(
    robot: EpisodeFollower,
    listener: TtyKeyListener,
    reset_time_s: float,
    gripper_cmd: float,
) -> tuple[str, float]:
    """episode 之间的人工复位等待:关节保持自由,人手拖回起点,夹爪键仍有效。"""
    if reset_time_s <= 0:
        return None, gripper_cmd
    log_say("把机械臂拖回起点,或按 n 立即开始下一条", False)
    deadline = time.perf_counter() + reset_time_s
    while time.perf_counter() < deadline:
        event, gripper_cmd = _process_keys(listener.drain(), robot, gripper_cmd)
        if event == "quit":
            return "quit", gripper_cmd
        if event == "next":
            return None, gripper_cmd
        time.sleep(0.05)
    return None, gripper_cmd


@parser.wrap()
def record(cfg: EpisodeRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)

    features = combine_feature_dicts(
        hw_to_dataset_features(robot.action_features, ACTION, use_video=cfg.dataset.video),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=cfg.dataset.video),
    )

    num_cameras = len(robot.cameras)
    dataset = None
    listener = TtyKeyListener()
    try:
        robot.connect()
        listener.start()

        if not listener.tty_ok:
            logging.warning(
                "stdin 不是终端,按键控制不可用,将仅按 episode 时长录制。"
                "请改用 conda activate 激活环境后直接运行 python 本脚本,"
                "不要用 conda run(它会断开 stdin)。"
            )

        if cfg.drag_mode:
            # 前置检查:回零后机械臂必在默认位姿。
            # 断电后编码器坐标复位,读数会在限位内但相对默认位姿漂移
            # (实测 J3 漂到 165 仍能通过限位检查),故必须按默认位姿校验。
            pre = robot.arm.read_angles()
            diffs = [
                abs(a - d) if a is not None else float("inf")
                for a, d in zip(pre, DEFAULT_POSE)
            ]
            if max(diffs) > 3.0:
                raise RuntimeError(
                    f"码盘读数偏离默认位姿: {[round(a, 1) if a is not None else None for a in pre]}\n"
                    f"机械臂未回零或断电后坐标已丢失,请先执行: "
                    f"python scripts/bench_check.py home --yes"
                )
            logging.warning(
                "进入拖动示教(自由模式)前,请确认已用双手托住机械臂,防止坠落!"
            )
            robot.arm.set_free_mode(True)
        else:
            logging.info("非拖动模式:关节保持力矩,仅夹爪受键盘控制")

        if num_cameras == 0:
            logging.warning("未配置相机,数据集将只有状态/动作(无视频)")

        cfg.dataset.stamp_repo_id()
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            rgb_encoder=cfg.dataset.rgb_encoder,
            depth_encoder=cfg.dataset.depth_encoder,
            encoder_threads=cfg.dataset.encoder_threads,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
        )
        logging.info("数据集特征: %s", pformat(dataset.features))

        recorded = 0
        gripper_cmd = 0.0
        quit_requested = False
        with VideoEncodingManager(dataset):
            while recorded < cfg.dataset.num_episodes and not quit_requested:
                log_say(f"开始录制 episode {dataset.num_episodes}", cfg.play_sounds)
                event, gripper_cmd = record_loop(
                    robot=robot,
                    listener=listener,
                    dataset=dataset,
                    features=features,
                    fps=cfg.dataset.fps,
                    episode_time_s=cfg.dataset.episode_time_s,
                    task=cfg.dataset.single_task,
                    drag_mode=cfg.drag_mode,
                    trajectory=cfg.trajectory,
                )

                if event == "quit":
                    quit_requested = True
                    break
                if event == "rerecord":
                    log_say("重录本条 episode", cfg.play_sounds)
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded += 1
                logging.info("episode %d/%d 已保存", recorded, cfg.dataset.num_episodes)

                if recorded < cfg.dataset.num_episodes:
                    event, gripper_cmd = _wait_reset(
                        robot, listener, cfg.dataset.reset_time_s, gripper_cmd
                    )
                    if event == "quit":
                        quit_requested = True
    finally:
        log_say("停止录制", cfg.play_sounds, blocking=True)
        if dataset is not None:
            dataset.finalize()
        if robot.is_connected:
            if cfg.drag_mode:
                robot.arm.set_free_mode(False)
            robot.disconnect()
        listener.stop()
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
