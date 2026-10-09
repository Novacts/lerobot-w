# -*- coding: utf-8 -*-
"""Episode 机械臂硬件接口抽象层。

三个后端(can_core / tcp_server / dummy)都实现本接口,
episode_robot.py 只依赖本抽象,后端通过配置切换。

常量来源:
- 零点偏移:   student_ros/student_ros1/src/episode_controller/episode_controller/robot_interface_node.py
- 关节限位:   6d/daima/6d/episodeApp.py 的 degree_range_list(电机角度制)
- 夹爪量程:   robot_interface_node.py 的 servo_gripper 钳位(0-110 度)
"""

from abc import ABC, abstractmethod
from typing import List, Optional

# ── 机械臂常量(电机角度制) ────────────────────────────────
NUM_JOINTS = 6
# 电机角度 -> MoveIt/DH 关节角 的零点偏移
JOINT_OFFSETS = [180.0, 90.0, 83.0, 30.0, 110.0, 30.0]
# 各关节电机角度限位(度)。
# 注意:episodeApp.py 的 degree_range_list 是 DH 系限位,换算到电机系后
# 与 controller_core 的 degree_limit_list(台架实测读出)完全一致:
#   [[0,340],[0,180],[0,163],[0,335],[0,220],[0,335]]
JOINT_LIMITS_DEG = [
    [0, 340],    # J1 (DH [-180,160] 平移 180)
    [0, 180],    # J2 (DH [0,180] 镜像,限位不变)
    [0, 163],    # J3 (DH [-80,83] 镜像平移 83)
    [0, 335],    # J4 (DH [-30,305] 平移 30)
    [0, 220],    # J5 (DH [-110,110] 镜像平移 110)
    [0, 335],    # J6 (DH [-30,305] 平移 30)
]
# 舵机夹爪角度范围(度)
GRIPPER_MIN, GRIPPER_MAX = 0, 110
# 回零后的默认位姿(与 robot_interface_node.py:199 一致,在限位内)
DEFAULT_POSE = [180.0, 90.0, 83.0, 30.0, 110.0, 30.0]

# 读值/拖动容差限位(名义限位 ±2°)。台架实测人手推压时电机允许轻微
# 超出名义限位,读取与数据校验用此容差;仅 dummy 模拟沿用。
# 注意:命令下发钳位必须用 SEND_LIMITS_DEG——controller_core 对下发
# 指令只认名义限位,±2° 放行会刷"电机N角度超出范围"/负数警告(2026-09-23 实测)。
MOTOR_LIMITS_DEG = [
    [lo - 2.0, hi + 2.0] for lo, hi in JOINT_LIMITS_DEG
]

# 命令下发钳位 = 名义限位(controller_core 的 degree_limit_list 与
# JOINT_LIMITS_DEG 完全一致,台架实测 2026-09-10)
SEND_LIMITS_DEG = JOINT_LIMITS_DEG

# 状态/动作空间维度:6 关节角 + 1 夹爪
STATE_DIM = NUM_JOINTS + 1
ACTION_DIM = STATE_DIM


class EpisodeArmInterface(ABC):
    """Episode 机械臂底层接口抽象。

    约定:
    - 所有角度单位均为度(电机角度制),不做零点偏移换算
    - read_angles 返回 6 元素列表,CAN 阻塞时个别元素为 None(调用方兜底)
    - send_angles 尽量非阻塞;阻塞实现的延迟需在台架测试实测
    """

    @abstractmethod
    def connect(self) -> None:
        """建立连接(CAN 初始化 / TCP 握手),失败抛异常。"""

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接并释放资源。"""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """连接状态。"""

    @abstractmethod
    def read_angles(self) -> List[Optional[float]]:
        """读取 6 个电机角度(度)。失败元素允许为 None。"""

    @abstractmethod
    def read_gripper(self) -> Optional[int]:
        """读取舵机夹爪角度(0-110 度)。硬件不支持回读时返回 None。"""

    @abstractmethod
    def send_angles(
        self, angles: List[float], gripper: Optional[int] = None
    ) -> None:
        """下发 6 关节目标角(度)+ 可选夹爪角度。应尽量非阻塞。"""

    @abstractmethod
    def send_gripper(self, gripper: int) -> None:
        """仅下发夹爪角度,不动关节。

        拖动示教(自由模式)下调用:此时关节由人手引导,
        不能下发关节目标,否则会重新上力矩。
        """

    @abstractmethod
    def set_free_mode(self, enable: bool) -> None:
        """拖动示教自由模式开关。进入前需确认有人托举机械臂。"""

    @abstractmethod
    def emergency_stop(self) -> None:
        """急停。"""

    @abstractmethod
    def home(self) -> None:
        """回零并移动到默认位姿。"""
