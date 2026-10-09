# Episode1 台架实测 + 真机录制操作手册

所有命令在 `/home/wujunqi/6d/daima/imitation` 目录下执行。
外置卷(环境/数据集所在)路径:`/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25`。

```bash
cd /home/wujunqi/6d/daima/imitation
LEROBOT_PY="conda run -p /media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/envs/lerobot python"
```

---

## A. 开机/重启后的启动检查

```bash
# 1. 外置卷(重启后可能没自动挂载)
ls /media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/envs/lerobot
# 若不存在(必须用 UUID 挂载!NVMe 盘序会变,按设备名挂可能挂到 Windows 盘):
udisksctl mount -b /dev/disk/by-uuid/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25

# 2. PCAN 驱动与设备
cat /proc/pcan            # 应看到 "32 usb ..." 一行
lsusb | grep -i peak      # 应看到 PEAK System PCAN-USB

# 3. 机械臂上电 + CAN 通信(只读,无运动)
$LEROBOT_PY scripts/bench_check.py read
# 读数不为 None 且每次 ~5ms -> 通信正常
# 读数全为 0.0 -> 机械臂已上电但未回零,正常,继续做回零
```

---

## B. 台架实测(按顺序执行,运动步骤必须有人在场)

```bash
# B1. 环境自检(只读)
$LEROBOT_PY scripts/bench_check.py check

# B2. 码盘读取(只读) — A.3 同款
$LEROBOT_PY scripts/bench_check.py read

# B3. 回零(运动!确认机械臂周围无障碍后执行)
$LEROBOT_PY scripts/bench_check.py home --yes
# 预期:逐关节回零后停在默认位姿 [180, 90, 83, 30, 110, 30]

# B4. 阻塞语义:J2 ±3° 小往返(运动)
$LEROBOT_PY scripts/bench_check.py blocking --yes
# 结论判读:send 返回耗时 ≈ 预计运动时间 -> 阻塞;≈0 -> 非阻塞

# B5. J1 160-180° 干涉检查(运动,慢速,重点看挡板间隙)
$LEROBOT_PY scripts/bench_check.py j1-probe --yes

# B6. 自由模式(运动!先双手托住机械臂再执行)
$LEROBOT_PY scripts/bench_check.py free --yes
# 拖动期间每秒打印码盘;读数连续变化 -> 拖动示教可行

# B7. 夹爪 0/55/110(运动)
$LEROBOT_PY scripts/bench_check.py gripper --yes

# B8. 15Hz 控制环测速(微动 ±0.5°)
$LEROBOT_PY scripts/bench_check.py bandwidth --yes
```

**急停**:任何命令 Ctrl+C 都会触发急停流程;若失灵,立即切断机械臂电源。

---

## C. 拖动示教录制(真机)

```bash
# 前置:已完成回零(B3),机械臂停在默认位姿
$LEROBOT_PY scripts/episode_record.py \
    --robot.type=episode_follower \
    --robot.backend=can --robot.usb_index=1 \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=<你的用户名>/episode_drag_v1 \
    --dataset.root=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/datasets/episode_drag_v1 \
    --dataset.fps=15 --dataset.episode_time_s=60 --dataset.reset_time_s=10 \
    --dataset.num_episodes=50 \
    --dataset.single_task="抓取红色方块放入盒子" \
    --dataset.push_to_hub=false \
    --drag_mode=true
```

流程:提示进入自由模式 → **双手托住机械臂** → 拖动示教 → 按 n 结束本条存盘 →
拖回起点 → 下一条 → 全部录完按 q 退出(自动退出自由模式并断开)。

键位:`n` 结束存盘 / `r` 重录 / `q` 退出 / `c` 夹爪闭合 / `o` 夹爪张开 / `g` 切换。

---

## D. 训练 ACT

```bash
$LEROBOT_PY -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=<你的用户名>/episode_drag_v1 \
    --dataset.root=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/datasets/episode_drag_v1 \
    --policy.type=act --policy.push_to_hub=false \
    --output_dir=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/outputs/act_v1 \
    --steps=20000 --batch_size=32 \
    --save_freq=2000 --log_freq=50 \
    --wandb.enable=false
```

数据集可视化(训练前检查数据质量):

```bash
$LEROBOT_PY -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id <你的用户名>/episode_drag_v1 \
    --root /media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/datasets/episode_drag_v1
```

---

## E. 部署 rollout(真机执行策略)

```bash
# 前置:机械臂已回零、周围无障碍
$LEROBOT_PY -m lerobot.scripts.lerobot_rollout \
    --strategy.type=base \
    --policy.path=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/outputs/act_v1/checkpoints/last/pretrained_model \
    --robot.type=episode_follower --robot.backend=can --robot.usb_index=1 \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="抓取红色方块放入盒子" --duration=60
```

---

## F. 故障排查速查

| 现象 | 处理 |
|---|---|
| `/media/wujunqi/` 空 | A.1 重挂载 |
| `check` 报 CAN 初始化失败 | 查机械臂电源、USB 线、`cat /proc/pcan` |
| `read` 全 None | 电机没上电或波特率/终端电阻问题 |
| `read` 全 0.0 | 未回零或刚断电(计数复位),执行 B3 回零 |
| 运动命令无效、码盘恒 0 | 没调 enable_motors(本项目 connect() 已自动使能;厂商 ROS 节点没有此步,回零是假象,勿用 ROS 节点回零) |
| 回零某关节卡住"标志恒 True" | 该关节已在零位,脚本会告警跳过,正常 |
| 训练 OOM | `--batch_size=16 --policy.dim_model=256` 降规模 |
| 录制时机械臂下滑 | 立即按 q 退出,自由模式恢复力矩前托住 |
| rollout 刷"电机N角度超出范围"/"degree_list中有负数" | 策略输出越界(ACT 外插+数据里 J3 到 165°)撞控制器名义限位;命令钳位已收紧至 SEND_LIMITS_DEG(2026-09-23),若仍出现检查回零与数据质量 |

## 台架实测结论(2026-09-10,全部通过)

1. **必须 enable_motors(True)**:MotorControl 初始化后电机默认不响应运动命令(码盘恒 0)。
   厂商 ROS 节点(robot_interface_node)没有此步骤,其"回零完成"是标志位残留的假象,不可信。
2. **execute_motors_degrees_normal 非阻塞**:1-3ms 返回,返回值=预计运动时间(有时明显偏大,勿用于精确计时,按读数轮询到位)。
3. **回零流程**:trigger_home → 标志 False(回零中)→ True(完成);已在零位的关节标志恒 True,跳过即可。
4. **读数带宽**:read 均值 4.5ms(P99 4.9ms),读+写一轮 6.3ms,15Hz 预算 66.7ms 余量 10 倍。
5. **自由模式**:拖动期间码盘连续可读 → 拖动示教可行。
6. **夹爪**:无回读,取最后下发值。
7. **J1 160-180°**:实测无干涉,保持 MOTOR_LIMITS_DEG 不变。
8. **断电后坐标丢失**:重新上电后编码器计数复位(出现越限读数如 J3=160),必须重新回零。
