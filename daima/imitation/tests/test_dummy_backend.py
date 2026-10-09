# -*- coding: utf-8 -*-
"""Dummy 后端冒烟测试:验证 episode_hw 包导入与仿真语义。

不依赖 lerobot / 硬件。运行:
    python tests/test_dummy_backend.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_episode.episode_hw import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_POSE,
    GRIPPER_MAX,
    NUM_JOINTS,
    STATE_DIM,
    DummyEpisodeArm,
)

TOL = 1e-6


def test_constants():
    assert NUM_JOINTS == 6
    assert STATE_DIM == 7 and ACTION_DIM == 7
    assert len(DEFAULT_POSE) == 6
    assert GRIPPER_MAX == 110


def test_connect_and_read():
    arm = DummyEpisodeArm(noise_deg=0.0)
    assert not arm.is_connected
    arm.connect()
    assert arm.is_connected
    angles = arm.read_angles()
    assert len(angles) == 6
    # 初始位于默认位姿,无噪声时应完全一致
    assert all(abs(a - b) < TOL for a, b in zip(angles, DEFAULT_POSE))
    arm.disconnect()
    assert not arm.is_connected


def test_move_to_target():
    arm = DummyEpisodeArm(max_speed_deg_per_s=100.0, noise_deg=0.0)
    arm.connect()
    target = [150.0, 100.0, 50.0, 60.0, 80.0, 90.0]
    arm.send_angles(target, gripper=60)
    # 100 度/秒,总位移 ~108 度,读 2 秒应到位(含噪声时为 0)
    deadline = time.perf_counter() + 3.0
    while time.perf_counter() < deadline:
        angles = arm.read_angles()
        if all(abs(a - t) < 0.5 for a, t in zip(angles, target)):
            break
        time.sleep(0.02)
    else:
        raise AssertionError(f"未在时限内到达目标: {angles} vs {target}")
    assert arm.read_gripper() == 60


def test_drag_in_free_mode():
    """自由模式下 drag() 模拟人手拖动,位置立即改变且不回落。"""
    arm = DummyEpisodeArm(noise_deg=0.0)
    arm.connect()
    arm.set_free_mode(True)
    assert arm.free_mode
    before = arm.read_angles()
    arm.drag([-10.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    after = arm.read_angles()
    assert abs(after[0] - before[0] + 10.0) < 1e-3
    assert abs(after[1] - before[1] - 5.0) < 1e-3
    # 拖动后 read 不应把位置拉回旧目标
    again = arm.read_angles()
    assert abs(again[0] - after[0]) < 1e-3


def test_clamping():
    """发送超限目标时钳位到 MOTOR 限位(名义限位+2°余量)。"""
    arm = DummyEpisodeArm(noise_deg=0.0)
    arm.connect()
    arm.send_angles([999.0, -999.0, 0.0, 0.0, 0.0, 0.0], gripper=999)
    angles = arm.read_angles()
    assert angles[0] <= 342.0 + 1e-6  # J1 名义上限 340 + 2°余量
    assert angles[1] >= 0.0 - 1e-6   # J2 下限
    assert arm.read_gripper() == GRIPPER_MAX


def test_emergency_stop():
    arm = DummyEpisodeArm(max_speed_deg_per_s=30.0, noise_deg=0.0)
    arm.connect()
    arm.send_angles([160.0, 0.0, -80.0, -30.0, -110.0, -30.0])
    time.sleep(0.2)
    arm.read_angles()          # 先走一段
    arm.emergency_stop()       # 急停:目标=当前位置
    first = arm.read_angles()
    time.sleep(0.3)
    second = arm.read_angles()
    assert all(abs(a - b) < 1e-3 for a, b in zip(first, second))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{tests.__len__()} tests passed")
