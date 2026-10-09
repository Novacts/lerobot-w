# -*- coding: utf-8 -*-
"""TCP 上位机后端:与厂商上位机(localhost:12345)通信,开发期使用。

协议(与 6d/daima/6d/episodeApp.py 一致):
- 每命令新建一条 socket,一问一答
- 帧格式:8 字节大端长度头 + UTF-8 JSON
- 已知 action: angle_mode / move_xyz_rotation / move_linear_xyz_rotation /
  gripper_on / gripper_off / servo_gripper / set_free_mode /
  get_motor_angles / get_T / get_pose / emergency_stop
"""

import json
import socket
from typing import List, Optional

from .base import (
    DEFAULT_POSE,
    GRIPPER_MAX,
    GRIPPER_MIN,
    NUM_JOINTS,
    SEND_LIMITS_DEG,
    EpisodeArmInterface,
)

# 命令钳位用名义限位(与 can_core 一致,上位机侧同样只认名义限位)
_LO = [lim[0] for lim in SEND_LIMITS_DEG]
_HI = [lim[1] for lim in SEND_LIMITS_DEG]


class EpisodeTcpArm(EpisodeArmInterface):
    def __init__(self, ip: str = "localhost", port: int = 12345) -> None:
        self.server_address = (ip, port)
        self._connected = False
        self._last_angles: List[float] = list(DEFAULT_POSE)
        self._last_gripper: int = 0

    def _send_command(self, command: dict, timeout: float = 5.0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(self.server_address)
            data = json.dumps(command).encode("utf-8")
            s.sendall(len(data).to_bytes(8, "big") + data)
            head = s.recv(8)
            if not head:
                return None
            length = int.from_bytes(head, "big")
            body = b""
            while len(body) < length:
                packet = s.recv(length - len(body))
                if not packet:
                    break
                body += packet
        return json.loads(body.decode("utf-8"))

    def connect(self) -> None:
        result = self._send_command({"action": "get_motor_angles"})
        if result is None:
            raise RuntimeError(
                f"无法连接上位机 {self.server_address},请确认 3.gui_server_uni 已启动"
            )
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def read_angles(self) -> List[Optional[float]]:
        result = self._send_command({"action": "get_motor_angles"})
        if not result:
            return list(self._last_angles)
        fixed = []
        for i, a in enumerate(result):
            if a is None:
                fixed.append(self._last_angles[i])
            else:
                fixed.append(float(a))
        self._last_angles = fixed
        return fixed

    def read_gripper(self) -> Optional[int]:
        # 上位机协议未暴露舵机回读
        return None

    def send_angles(
        self, angles: List[float], gripper: Optional[int] = None
    ) -> None:
        clamped = [
            max(_LO[i], min(_HI[i], float(a)))
            for i, a in enumerate(angles[:NUM_JOINTS])
        ]
        # angle_mode 返回预计运动时间;上位机侧排队执行,这里不等待
        self._send_command(
            {"action": "angle_mode", "params": [clamped, 1]}
        )
        if gripper is not None:
            g = int(max(GRIPPER_MIN, min(GRIPPER_MAX, gripper)))
            self._send_command({"action": "servo_gripper", "params": g})
            self._last_gripper = g

    def send_gripper(self, gripper: int) -> None:
        g = int(max(GRIPPER_MIN, min(GRIPPER_MAX, gripper)))
        self._send_command({"action": "servo_gripper", "params": g})
        self._last_gripper = g

    def set_free_mode(self, enable: bool) -> None:
        self._send_command({"action": "set_free_mode", "params": int(enable)})

    def emergency_stop(self) -> None:
        self._send_command({"action": "emergency_stop", "params": 1})

    def home(self) -> None:
        self.send_angles(DEFAULT_POSE, 0)
