#!/bin/bash

# 源目录和目标目录设置
SOURCE_DIR="/tmp/"
DEST_DIR="/root/autodl-tmp/pykt-toolkit/examples/run_logs"
# 同步间隔（秒），可以根据需要调整，5-10秒通常足够实时了
INTERVAL=5

# 确保目标文件夹存在
mkdir -p "$DEST_DIR"

echo "开始同步日志: ${SOURCE_DIR}ts-out.* -> ${DEST_DIR}"
echo "同步间隔: ${INTERVAL} 秒"

# 无限循环
while true; do
    # 使用 rsync 进行增量同步
    # -a: 归档模式，保持文件属性不变
    # --include="ts-out.*" --exclude="*": 只匹配 ts-out. 开头的文件，忽略其他 /tmp 下的杂乱文件
    # -q: 安静模式，不在终端输出一大堆同步信息
    rsync -aq --include="ts-out.*" --exclude="*" "$SOURCE_DIR" "$DEST_DIR/"
    
    # 休眠指定时间后再次执行
    sleep $INTERVAL
done