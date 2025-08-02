#!/bin/bash

# 参数设置
TEST_DIR="$1"          # 通过命令行参数指定测试目录
TEST_FILE="speedtest"  # 测试文件名
TEST_SIZE="1G"         # 测试文件大小（1GB）
BLOCK_SIZE="1M"        # 块大小（1MB）

# 检查目录是否存在
if [ ! -d "$TEST_DIR" ]; then
    echo "错误：目录 '$TEST_DIR' 不存在！"
    exit 1
fi

# 进入测试目录
cd "$TEST_DIR" || exit 1

echo "-------------------------------------"
echo "测试目录: $(pwd)"
echo "测试文件: $TEST_FILE (大小: $TEST_SIZE)"
echo "块大小: $BLOCK_SIZE"
echo "-------------------------------------"

# 写入速度测试
echo "正在测试写入速度..."
WRITE_SPEED=$(dd if=/dev/zero of="$TEST_FILE" bs="$BLOCK_SIZE" count=$((${TEST_SIZE%G} * 1024)) oflag=direct status=progress 2>&1 | tail -n 1 | awk '{print $(NF-1), $NF}')
echo "写入速度: $WRITE_SPEED"

# 读取速度测试
echo "正在测试读取速度..."
READ_SPEED=$(dd if="$TEST_FILE" of=/dev/null bs="$BLOCK_SIZE" iflag=direct status=progress 2>&1 | tail -n 1 | awk '{print $(NF-1), $NF}')
echo "读取速度: $READ_SPEED"

# 清理测试文件
echo "清理测试文件..."
rm -f "$TEST_FILE"

echo "-------------------------------------"
echo "测试完成！"
echo "写入速度: $WRITE_SPEED"
echo "读取速度: $READ_SPEED"
echo "-------------------------------------"