# Episode1 机械臂 × lerobot 模仿学习

基于 HuggingFace lerobot 0.6.1 的拖动示教(kinesthetic teaching)模仿学习管线。
已验证:采集 → 数据集 → ACT 训练 → rollout 部署,全程无硬件(dummy 后端 + 合成相机)。

## 环境

- conda 环境:`/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/envs/lerobot`
  (根分区空间不足,环境与数据集都在外置卷上;pip 缓存也在该卷)
- lerobot 0.6.1 + torch 2.11.0(CUDA 13,RTX 5060 可用)
- 本机无法访问 GitHub,lerobot 从 PyPI 安装;插件包为本地 editable 安装

```bash
LEROBOT_PY="conda run -p /media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/envs/lerobot python"
cd /home/wujunqi/6d/daima/imitation
$LEROBOT_PY -m pip install -e .   # 重装插件(改代码后需重装)
```

## 目录结构

```
imitation/
├── pyproject.toml                  # dist 名 lerobot_robot_episode(插件发现机制依赖前缀)
├── lerobot_robot_episode/          # lerobot 第三方插件包
│   ├── config_episode.py           # EpisodeFollowerConfig(--robot.type=episode_follower)
│   ├── episode_follower.py         # Robot 实现:7 维状态/动作(6 关节角+夹爪,度)
│   ├── synthetic_camera.py         # 合成相机(--robot.cameras="{top:{type:synthetic,...}}")
│   └── episode_hw/                 # 硬件后端抽象
│       ├── base.py                 # 接口 + 常量(限位/默认位姿,来源见注释)
│       ├── dummy.py                # 仿真后端(含 drag() 模拟人手拖动)
│       ├── tcp_server.py           # 厂商上位机 JSON 协议(localhost:12345)
│       └── can_core.py             # PEAK CAN 直驱(episode_controller_core wheel)
├── scripts/episode_record.py       # 拖动示教录制脚本
└── tests/test_dummy_backend.py     # 后端冒烟测试(6 项)
```

## 无硬件冒烟(已跑通)

```bash
# 1. 录制(2 条 × 6 秒,dummy 机械臂按 circle 轨迹模拟人手拖动)
$LEROBOT_PY scripts/episode_record.py \
    --robot.type=episode_follower --robot.backend=dummy \
    --robot.cameras="{top: {type: synthetic, width: 320, height: 240, fps: 10}}" \
    --dataset.repo_id=wujunqi/episode_smoke \
    --dataset.root=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/datasets/episode_smoke_v1 \
    --dataset.fps=10 --dataset.episode_time_s=6 --dataset.reset_time_s=1 \
    --dataset.num_episodes=2 --dataset.single_task="拖动示教画圈演示" \
    --dataset.push_to_hub=false --trajectory=circle

# 2. 训练(微型 ACT,30 步冒烟)
$LEROBOT_PY -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=wujunqi/episode_smoke \
    --dataset.root=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/datasets/episode_smoke_v1 \
    --dataset.image_transforms.enable=false \
    --policy.type=act --policy.push_to_hub=false \
    --policy.dim_model=64 --policy.n_heads=4 --policy.dim_feedforward=256 \
    --policy.n_encoder_layers=2 --policy.n_vae_encoder_layers=2 --policy.latent_dim=16 \
    --policy.pretrained_backbone_weights=null \
    --output_dir=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/outputs/act_smoke \
    --steps=30 --batch_size=8 --eval_steps=0 --log_freq=5 --save_freq=1000 \
    --wandb.enable=false --num_workers=2

# 3. 部署(官方 rollout,策略驱动 dummy 机械臂)
$LEROBOT_PY -m lerobot.scripts.lerobot_rollout \
    --strategy.type=base \
    --policy.path=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/outputs/act_smoke/checkpoints/last/pretrained_model \
    --robot.type=episode_follower --robot.backend=dummy \
    --robot.cameras="{top: {type: synthetic, width: 320, height: 240, fps: 10}}" \
    --task="拖动示教画圈演示" --duration=15

# 4. 数据集可视化
$LEROBOT_PY -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id wujunqi/episode_smoke \
    --root /media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/datasets/episode_smoke_v1
```

## 真机录制(拖动示教)

```bash
# 先回零(人工确认机械臂无干涉)
$LEROBOT_PY -c "from lerobot_robot_episode import EpisodeFollowerConfig; \
from lerobot_robot_episode.episode_follower import EpisodeFollower; \
r = EpisodeFollower(EpisodeFollowerConfig(backend='can', usb_index=1)); \
r.connect(); r.arm.home(); r.disconnect()"

# 录制: 进入自由模式前务必双手托住机械臂!
$LEROBOT_PY scripts/episode_record.py \
    --robot.type=episode_follower --robot.backend=can --robot.usb_index=1 \
    --robot.cameras="{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=<user>/episode_drag_v1 \
    --dataset.root=/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25/datasets/episode_drag_v1 \
    --dataset.fps=15 --dataset.episode_time_s=60 --dataset.reset_time_s=10 \
    --dataset.num_episodes=50 --dataset.single_task="抓取红色方块放入盒子" \
    --dataset.push_to_hub=false --drag_mode=true
```

键位:n=结束本条存盘 / r=重录 / q=退出 / c=夹爪闭合 / o=夹爪张开 / g=切换。

## 已知问题 / 台架 TODO

1. **J1 限位**:episodeApp.py 的 qlim 上限 160(IK 用),但回零默认位姿下发 180。
   钳位限位已扩展到含默认位姿(MOTOR_LIMITS_DEG),160-180 区间需台架确认无干涉。
2. **can_core.send_angles 阻塞语义**:execute_motors_degrees_normal 返回预计运动时间,
   当前按阻塞处理。若台架实测有非阻塞下发(dynamic_move),替换以支持 15Hz 控制环。
3. **自由模式读码盘**:拖动示教依赖 set_free_mode 后 read_motor_angles 仍可读,
   需台架确认。
4. **rollout 帧率**:rollout 默认 30fps,实际受相机帧率限制;真机建议与数据集 fps 一致。
5. **TCP 后端 read_gripper**:上位机协议未暴露舵机回读,夹爪状态取最后下发值。
6. 录制脚本根目录(root)不能预先存在(lerobot 要求自建)。
