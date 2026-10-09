#!/usr/bin/env bash
# 清除旧实验数据(dummy 冒烟数据集、坏录制、冒烟训练产物)。
# 环境/代码/pip 缓存不动。
set -e

VOL="/media/wujunqi/7bd21c4e-c188-4fcd-b8ce-2c4bc7189f25"

echo "===== 待清理目标 ====="
for d in \
    "$VOL/datasets/episode_smoke_v1" \
    "$VOL/datasets/episode_drag_v1" \
    "$VOL/outputs/act_smoke"; do
    [ -e "$d" ] && echo "  删除: $d" || echo "  不存在(跳过): $d"
done

read -r -p "确认删除以上目录? 输入 yes 继续: " ans
[ "$ans" = "yes" ] || { echo "已取消"; exit 1; }

for d in \
    "$VOL/datasets/episode_smoke_v1" \
    "$VOL/datasets/episode_drag_v1" \
    "$VOL/outputs/act_smoke"; do
    rm -rf "$d"
done
echo "===== 清理完成 ====="
df -h "$VOL" | tail -1
