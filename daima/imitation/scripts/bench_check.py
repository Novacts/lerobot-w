#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Episode1 台架实测脚本。

每个子命令对应一项实测;运动类命令需显式 --yes(务必先确认机械臂
周围无障碍)。任何异常/Ctrl+C 都会触发急停并退出自由模式。

用法:
  python scripts/bench_check.py check             # 环境自检(只读)
  python scripts/bench_check.py read              # 连接 + 读码盘(只读)
  python scripts/bench_check.py blocking --yes    # 阻塞语义:J2 ±3° 小往返
  python scripts/bench_check.py j1-probe --yes    # J1 160-180 干涉检查(慢速)
  python scripts/bench_check.py free --yes        # 自由模式(需人手托臂!)
  python scripts/bench_check.py home --yes        # 回零
  python scripts/bench_check.py gripper --yes     # 夹爪 0/55/110
  python scripts/bench_check.py bandwidth --yes   # 15Hz 控制环测速
"""

import argparse
import sys
import time
import traceback

from lerobot_robot_episode.episode_hw import (
    DEFAULT_POSE,
    GRIPPER_MAX,
    MOTOR_LIMITS_DEG,
    EpisodeCoreArm,
)

USAGE_ORDER = ["check", "read", "home", "blocking", "j1-probe", "free", "gripper", "bandwidth"]


def _require_yes(args):
    if not getattr(args, "yes", False):
        print("!! 本命令会使机械臂运动。请确认:机械臂周围无障碍、无人在行程内。")
        print("!! 确认无误后加 --yes 重跑。")
        sys.exit(1)


def _get_arm(args):
    arm = EpisodeCoreArm(usb_index=args.usb_index)
    arm.connect()
    return arm


def _ensure_homed(arm, tol_deg=15.0):
    """运动类测试前置检查:当前位置需接近默认位姿(即已回零)。

    若未回零,编码器读数可能为任意值,运动会不可控。
    """
    angles = arm.read_angles()
    if any(a is None for a in angles):
        print(f"!! 码盘读取失败: {angles},请检查机械臂上电状态")
        sys.exit(1)
    diffs = [abs(a - d) for a, d in zip(angles, DEFAULT_POSE)]
    if max(diffs) > tol_deg:
        print(f"!! 当前位置 {[round(a,1) for a in angles]} 距默认位姿过远(> {tol_deg}°)")
        print("!! 请先执行: python scripts/bench_check.py home --yes")
        sys.exit(1)
    return angles


# ── 只读检查 ──────────────────────────────────────────────

def cmd_check(args):
    """环境自检:wheel、PCAN 驱动与设备、控制器初始化。"""
    print("=" * 60)
    print("[1/3] episode_controller_core wheel")
    try:
        from episode_controller_core.controller_core import MotorControl

        print("  OK: MotorControl 可导入")
    except Exception as e:
        print(f"  FAIL: {e}")
        sys.exit(1)

    print("[2/3] PEAK CAN 设备")
    try:
        with open("/proc/pcan") as f:
            content = f.read()
        print("  OK: /proc/pcan 存在")
        for line in content.splitlines():
            if line.strip().startswith(("n-", "32")):
                print(f"    {line.strip()}")
    except OSError as e:
        print(f"  FAIL: /proc/pcan 不可读 ({e}),检查驱动 modprobe pcan")
        sys.exit(1)

    print(f"[3/3] 控制器初始化 (usb_index={args.usb_index})")
    arm = EpisodeCoreArm(usb_index=args.usb_index)
    try:
        arm.connect()
        print("  OK: MotorControl 初始化成功")
        print(f"  initialized 属性: {getattr(arm._mc, 'initialized', '<无>')}")
        print(f"  motor_address: {getattr(arm._mc, 'motor_address', '<无>')}")
        print(f"  实例属性: {sorted(a for a in vars(arm._mc) if not a.startswith('__'))}")
        arm.disconnect()
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)


def cmd_read(args):
    """连接后连续读 10 次码盘(无任何运动)。"""
    arm = _get_arm(args)
    try:
        print("连续读取码盘 10 次(每次 timeout=0.05s):")
        for i in range(10):
            t0 = time.perf_counter()
            angles = arm.read_angles()
            dt = (time.perf_counter() - t0) * 1e3
            pretty = [f"{a:7.1f}" if a is not None else "   None" for a in angles]
            print(f"  #{i:2d}  {dt:5.1f}ms  [{', '.join(pretty)}]  gripper={arm.read_gripper()}")
            time.sleep(0.05)
        print(f"\n对比: 默认位姿 {DEFAULT_POSE}")
        print("结论: 若读数接近默认位姿且抖动小,码盘读取正常;")
        print("      若全为 None,检查 CAN 终端电阻/波特率/电机上电。")
    finally:
        arm.disconnect()


# ── 运动类检查 ────────────────────────────────────────────

def cmd_home(args):
    """回零并移动到默认位姿。"""
    _require_yes(args)
    print("!! 回零会逐关节找限位,机械臂会连续运动!")
    print("!! 确认机械臂周围无障碍后,回零将在 2 秒后开始 (Ctrl+C 取消)")
    time.sleep(2)
    arm = _get_arm(args)
    try:
        t0 = time.perf_counter()
        arm.home()
        dt = time.perf_counter() - t0
        angles = arm.read_angles()
        print(f"回零完成,耗时 {dt:.1f}s")
        print(f"当前位置: {[round(a,1) for a in angles]}")
        print(f"默认位姿: {DEFAULT_POSE}")
        print("结论: 逐关节回零是否全部成功?最终位姿是否与默认位姿一致?")
    finally:
        arm.disconnect()


def cmd_blocking(args):
    """阻塞语义:J2 往返 ±3°,测量 send 返回耗时 vs 实际到位耗时。"""
    _require_yes(args)
    arm = _get_arm(args)
    try:
        start = _ensure_homed(arm)
        print("!! J2 将做 ±3° 小幅度往返,请保持观察。")
        for target_delta, label in [(3.0, "+3°"), (-3.0, "-3°")]:
            target = list(start)
            target[1] += target_delta
            print(f"\n--- J2 {label} -> {target[1]:.1f}° ---")
            t0 = time.perf_counter()
            ret = arm._mc.execute_motors_degrees_normal(target, speed_ratio=1)
            t_return = time.perf_counter() - t0
            print(f"  send 返回耗时: {t_return:.3f}s, 返回值: {ret}")

            # 每 50ms 读一次,测到位时间
            settle = None
            for _ in range(int(t_return * 20) + 80):
                a = arm.read_angles()
                if a and abs(a[1] - target[1]) < 1.0:
                    settle = time.perf_counter() - t0
                    break
                time.sleep(0.05)
            print(f"  到位耗时: {settle:.3f}s" if settle else "  未在 4s 内到位")
            print(f"  当前位置: {[round(x,1) for x in arm.read_angles()]}")
            if ret is not None and isinstance(ret, (int, float)):
                print(f"  结论: 返回值 {ret}s 为预计运动时间 -> 阻塞语义(等值回);"
                      f"若返回即达 {t_return:.3f}s -> 非阻塞。")
        target = list(start)
        arm._mc.execute_motors_degrees_normal(target, speed_ratio=1)
        print("\n已回到起始位姿。")
    finally:
        arm.disconnect()


def cmd_j1_probe(args):
    """J1 160-180° 干涉检查:慢速往返,人眼观察挡板间隙。"""
    _require_yes(args)
    arm = _get_arm(args)
    try:
        start = _ensure_homed(arm)
        print("!! J1 将慢速 180° -> 160° -> 180°,每个驻点停 2 秒。")
        print("!! 请重点观察 J1 关节与挡板/线缆的间隙,随时 Ctrl+C。")
        for j1 in [170.0, 165.0, 160.0, 170.0, 180.0]:
            target = list(start)
            target[0] = j1
            print(f"\n--- J1 -> {j1}° ---")
            arm._mc.execute_motors_degrees_normal(target, speed_ratio=1)
            time.sleep(2.5)
            print(f"  当前位置: {[round(x,1) for x in arm.read_angles()]}")
        print("\n结论: 若全程无干涉 -> J1 可安全使用到 180(与厂商默认位姿一致),")
        print("      MOTOR_LIMITS_DEG 无需修改;若有干涉,记录角度并收紧钳位限位。")
    finally:
        arm.disconnect()


def cmd_free(args):
    """自由模式:力矩释放、码盘可读性、拖动后读数变化。"""
    _require_yes(args)
    print("!! 自由模式会释放关节力矩,机械臂将自然下垂!")
    print("!! 进入前必须用双手托住机械臂!")
    print("!! 3 秒后进入自由模式 (Ctrl+C 取消)")
    time.sleep(3)
    arm = _get_arm(args)
    try:
        start = _ensure_homed(arm)
        print(f"进入自由模式前读数: {[round(a,1) for a in start]}")
        arm.set_free_mode(True)
        print("已进入自由模式。请托住机械臂缓慢拖动 5 秒,期间每秒读码盘:")
        for i in range(5):
            angles = arm.read_angles()
            pretty = [f"{a:7.1f}" if a is not None else "   None" for a in angles]
            print(f"  t={i}s [{', '.join(pretty)}]")
            time.sleep(1)
        print("\n结论: 若拖动期间读数连续变化 -> 自由模式下码盘可读,拖动示教可行;")
        print("      若为 None 或冻结 -> 需换读取时机/协议。")
        try:
            input("请继续托住机械臂,按回车恢复力矩...")
        except EOFError:  # stdin 被重定向/conda run 场景
            print("(stdin 不可读,直接恢复力矩)")
        arm.set_free_mode(False)
        print("已恢复力矩。请确认机械臂保持当前位置不下垂。")
    finally:
        try:
            arm.set_free_mode(False)
        except Exception:
            pass
        arm.disconnect()


def cmd_gripper(args):
    """夹爪 0 -> 55 -> 110 -> 0,验证动作与回读。"""
    _require_yes(args)
    arm = _get_arm(args)
    try:
        for g in [0, 55, GRIPPER_MAX, 0]:
            print(f"--- 夹爪 -> {g}° ---")
            arm.send_gripper(g)
            time.sleep(1.0)
            print(f"  回读 read_gripper(): {arm.read_gripper()}")
        print("\n结论: 确认夹爪实际动作正常;read_gripper() 若为 None 则确认无回读,")
        print("      录制管线将使用最后下发值。")
    finally:
        arm.disconnect()


def cmd_bandwidth(args):
    """15Hz 控制环带宽:读码盘 100 次计时 + send 耗时。"""
    _require_yes(args)
    arm = _get_arm(args)
    try:
        _ensure_homed(arm)
        n = 100
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            arm.read_angles()
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        mean = sum(times) / n
        print(f"read_motor_angles x{n}: 均值 {mean:.1f}ms, P50 {times[n//2]:.1f}ms, "
              f"P99 {times[min(n-1, int(n*0.99))]:.1f}ms")
        print(f"-> 理论读频率: {1000/mean:.0f} Hz {'(满足 15Hz)' if 1000/mean > 15 else '(不满足 15Hz!)'}")

        # send 耗时(原地 ±0.5° 微动,测量下发本身)
        current = arm.read_angles()
        target = [a + 0.5 for a in current]
        t0 = time.perf_counter()
        ret = arm._mc.execute_motors_degrees_normal(target, speed_ratio=1)
        t_send = time.perf_counter() - t0
        print(f"send(±0.5°) 耗时: {t_send*1e3:.1f}ms, 返回: {ret}")
        print(f"-> 读+写一轮: {mean + t_send*1e3:.1f}ms (15Hz 需 < 66.7ms)")
    finally:
        arm.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Episode1 台架实测")
    parser.add_argument("step", choices=USAGE_ORDER, help="实测步骤")
    parser.add_argument("--yes", action="store_true", help="确认允许机械臂运动")
    parser.add_argument("--usb-index", type=int, default=1, help="PCAN USB 索引")
    args = parser.parse_args()

    cmds = {
        "check": cmd_check,
        "read": cmd_read,
        "home": cmd_home,
        "blocking": cmd_blocking,
        "j1-probe": cmd_j1_probe,
        "free": cmd_free,
        "gripper": cmd_gripper,
        "bandwidth": cmd_bandwidth,
    }
    print(f"===== bench_check: {args.step} =====")
    try:
        cmds[args.step](args)
    except KeyboardInterrupt:
        print("\n!! Ctrl+C,触发急停!")
        print("  请立即按住机械臂,防止自由模式下垂。")
        # 尽力急停
        try:
            arm = EpisodeCoreArm(usb_index=args.usb_index)
            arm.connect()
            arm.emergency_stop()
            arm.set_free_mode(False)
            arm.disconnect()
            print("  急停已执行。")
        except Exception as e:
            print(f"  急停失败,请立即切断机械臂电源! ({e})")
        sys.exit(130)


if __name__ == "__main__":
    main()
