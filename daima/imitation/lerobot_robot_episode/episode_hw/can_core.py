# -*- coding: utf-8 -*-
"""CAN 直驱后端:包装 episode_controller_core.MotorControl(闭源 Cython wheel)。

前置条件:
- PEAK CAN 驱动已安装(pcaninfo 可见设备)
- wheelhouse 下的 episode_controller_core wheel 已 pip install
- 机械臂上电;connect() 会自动 enable_motors(True)

台架实测结论(2026-09-10):
- 必须 enable_motors(True),否则电机不响应运动命令、码盘恒 0
- execute_motors_degrees_normal 非阻塞:1ms 返回,返回值=预计运动时间(s)
  -> 15Hz 控制环可直接逐帧下发
- trigger_home 后 home 标志先 False(回零中)再 True(完成);
  触发前标志可能残留 True,轮询前须先等它变 False
"""

import time
from typing import List, Optional

from .base import (
    DEFAULT_POSE,
    GRIPPER_MAX,
    GRIPPER_MIN,
    NUM_JOINTS,
    SEND_LIMITS_DEG,
    EpisodeArmInterface,
)

try:
    from episode_controller_core.controller_core import MotorControl
except ImportError:  # 无 wheel 环境(如 dummy 测试)时延迟报错
    MotorControl = None

# 命令钳位用名义限位:controller_core 对下发指令只认名义限位,
# ±2° 余量(MOTOR_LIMITS_DEG)会放行越界指令导致刷屏报错。
_LO = [lim[0] for lim in SEND_LIMITS_DEG]
_HI = [lim[1] for lim in SEND_LIMITS_DEG]


def _clamp_angle(value: float, joint: int) -> float:
    return max(_LO[joint], min(_HI[joint], value))


class EpisodeCoreArm(EpisodeArmInterface):
    def __init__(self, usb_index: int = 1) -> None:
        self.usb_index = usb_index
        self._mc = None
        self._last_angles: List[float] = list(DEFAULT_POSE)
        self._last_gripper: int = 0

    def connect(self) -> None:
        if MotorControl is None:
            raise RuntimeError(
                "episode_controller_core 未安装,请先 pip install wheelhouse/*.whl"
            )
        self._mc = MotorControl(self.usb_index)
        if not getattr(self._mc, "initialized", False):
            raise RuntimeError(
                f"CAN 设备初始化失败(usb_index={self.usb_index}),"
                "请检查 PEAK 驱动与硬件连接"
            )
        # 实测:不使能则电机不响应运动、码盘恒 0(台架结论 2026-09-10)
        self._mc.enable_motors(True)

    def disconnect(self) -> None:
        if self._mc is not None:
            try:
                self._mc.closeCAN()
            except Exception:
                pass
            self._mc = None

    @property
    def is_connected(self) -> bool:
        return self._mc is not None

    def read_angles(self) -> List[Optional[float]]:
        if self._mc is None:
            raise RuntimeError("CAN 后端未连接")
        angles = self._mc.read_motor_angles(timeout=0.05)
        if not angles:
            return list(self._last_angles)
        # CAN 阻塞时个别元素为 None -> 沿用上一帧
        fixed = []
        for i, a in enumerate(angles):
            if a is None:
                fixed.append(self._last_angles[i])
            else:
                fixed.append(float(a))
        self._last_angles = fixed
        return fixed

    def read_gripper(self) -> Optional[int]:
        # controller_core 未暴露舵机回读接口
        return None

    def send_angles(
        self, angles: List[float], gripper: Optional[int] = None
    ) -> None:
        if self._mc is None:
            raise RuntimeError("CAN 后端未连接")
        clamped = [
            _clamp_angle(float(a), i) for i, a in enumerate(angles[:NUM_JOINTS])
        ]
        # 实测非阻塞:约 1ms 返回,返回值=预计运动时间(s)。
        # 15Hz 控制环逐帧下发,电机侧自行跟踪目标(台架结论 2026-09-10)
        self._mc.execute_motors_degrees_normal(clamped, speed_ratio=1)
        if gripper is not None:
            g = int(max(GRIPPER_MIN, min(GRIPPER_MAX, gripper)))
            self._mc.servo_gripper(g)
            self._last_gripper = g

    def send_gripper(self, gripper: int) -> None:
        if self._mc is None:
            raise RuntimeError("CAN 后端未连接")
        g = int(max(GRIPPER_MIN, min(GRIPPER_MAX, gripper)))
        self._mc.servo_gripper(g)
        self._last_gripper = g

    def set_free_mode(self, enable: bool) -> None:
        if self._mc is None:
            raise RuntimeError("CAN 后端未连接")
        self._mc.set_free_mode(1 if enable else 0)

    def emergency_stop(self) -> None:
        if self._mc is not None:
            self._mc.emergency_stop()

    def home(self) -> None:
        """逐个关节回零(带超时),再移动到默认位姿。

        与 robot_interface_node.py 的 _perform_initialization(init_mode=0)一致,
        并修正三点(台架实测 2026-09-10):
        - 必须先 enable_motors(True)(connect() 已做),否则回零/运动全部无效
        - 触发回零前 home 标志可能残留 True,须先等它变 False 再等回 True
        - J4-J6 实测存在标志恒 True 的情况(已在零位的关节不再回零):
          等 2s 仍未开始则视为已回零,告警跳过
        - execute_motors_degrees_normal 非阻塞,需按预计时间等待到位
        """
        if self._mc is None:
            raise RuntimeError("CAN 后端未连接")
        for addr in self._mc.motor_address:
            self._mc.trigger_home(addr, 0x02)
            # 等回零真正开始(标志 False);2s 内未开始则跳过(视为已回零)
            t0 = time.perf_counter()
            started = False
            while self._mc.check_home_status(addr):
                if time.perf_counter() - t0 > 2.0:
                    break
                time.sleep(0.1)
            else:
                started = True
            if not started:
                print(
                    f"警告:关节 address={addr} 回零未启动(标志恒 True),"
                    "视为已在零位,跳过。若机械臂姿态异常请检查该关节。"
                )
                continue
            # 等回零完成(标志回 True)
            t0 = time.perf_counter()
            while not self._mc.check_home_status(addr):
                if time.perf_counter() - t0 > 120.0:
                    raise TimeoutError(f"关节回零超时 (address={addr})")
                time.sleep(0.5)
        time.sleep(1)
        # 非阻塞下发,按预计时间等待;到位后按读数兜底校验
        estimated = self._mc.execute_motors_degrees_normal(DEFAULT_POSE)
        wait = float(estimated) if isinstance(estimated, (int, float)) else 8.0
        deadline = time.perf_counter() + wait + 30.0
        while time.perf_counter() < deadline:
            angles = self.read_angles()
            if all(abs(a - d) < 1.0 for a, d in zip(angles, DEFAULT_POSE)):
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"回零后未到达默认位姿: {[round(a, 1) for a in self.read_angles()]}"
        )
