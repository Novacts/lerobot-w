# -*- coding: utf-8 -*-
"""CAN 总线嗅探:抓取 SDK 下发的原始帧,对比不同关节/指令的协议差异。

以第二客户端身份打开 PCAN 通道,后台线程持续读帧,
主线程调用 MotorControl 执行指定操作,打印期间捕获的帧。
"""

import threading
import time

from episode_controller_core import PCANBasic
from episode_controller_core.controller_core import MotorControl

CHANNEL = PCANBasic.PCAN_USBBUS1  # usb_index=1 对应


class Sniffer:
    def __init__(self):
        self._pcan = PCANBasic.PCANBasic()
        self._frames = []
        self._stop = False

    def start(self):
        # 第二客户端:保留已有客户端(不 INITIALIZE),只读
        res = self._pcan.Initialize(CHANNEL, PCANBasic.PCAN_BAUD_1M)
        print(f"sniffer Initialize -> {res} (0x{res:X})")

        def loop():
            while not self._stop:
                res, msg, ts = self._pcan.Read(CHANNEL)
                if res == PCANBasic.PCAN_ERROR_OK and msg is not None:
                    self._frames.append((ts, msg.ID, bytes(msg.DATA[: msg.LEN])))
                elif res == PCANBasic.PCAN_ERROR_QRCVEMPTY:
                    time.sleep(0.001)
                else:
                    time.sleep(0.002)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()

    def reset(self):
        self._frames.clear()

    def dump(self, label):
        print(f"--- {label}: {len(self._frames)} 帧 ---")
        for ts, fid, data in self._frames:
            print(f"  t={ts:8.1f}ms ID=0x{fid:03X} [{data.hex(' ')}]")
        self.reset()

    def stop(self):
        self._stop = True
        self._pcan.Uninitialize(CHANNEL)


def main():
    sn = Sniffer()
    sn.start()
    time.sleep(0.3)
    sn.reset()

    mc = MotorControl(1)
    mc.enable_motors(True)
    time.sleep(0.2)

    print("### read_motor_angles ###")
    sn.reset()
    mc.read_motor_angles(timeout=0.05)
    time.sleep(0.2)
    sn.dump("read_motor_angles")

    print("### trigger_home J3 (768) ###")
    sn.reset()
    mc.trigger_home(768, 0x02)
    time.sleep(0.5)
    sn.dump("trigger_home J3")

    print("### trigger_home J4 (1024) ###")
    sn.reset()
    mc.trigger_home(1024, 0x02)
    time.sleep(0.5)
    sn.dump("trigger_home J4")

    print("### trigger_home J5 (1280) ###")
    sn.reset()
    mc.trigger_home(1280, 0x02)
    time.sleep(0.5)
    sn.dump("trigger_home J5")

    print("### check_home_status J4 ###")
    sn.reset()
    print("  status:", mc.check_home_status(1024))
    time.sleep(0.3)
    sn.dump("check_home_status J4")

    mc.closeCAN()
    sn.stop()


if __name__ == "__main__":
    main()
