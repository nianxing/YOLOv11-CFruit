#!/bin/bash

# Docker网络问题解决脚本
echo "=== Docker网络问题解决脚本 ==="
echo ""

# 检查Docker是否运行
if ! docker info >/dev/null 2>&1; then
    echo "✗ Docker未运行，请先启动Docker"
    exit 1
fi

echo "✓ Docker正在运行"
echo ""

# 方法1: 测试网络连接
echo "方法1: 测试网络连接..."
if ping -c 1 docker.io >/dev/null 2>&1; then
    echo "✓ 网络连接正常"
else
    echo "⚠ 网络连接可能有问题"
fi

# 方法2: 配置Docker镜像源
echo ""
echo "方法2: 配置Docker镜像源..."
echo "建议使用国内镜像源加速下载"

# 检查是否已配置镜像源
if [ -f /etc/docker/daemon.json ]; then
    echo "当前Docker配置:"
    cat /etc/docker/daemon.json
else
    echo "未找到Docker配置文件"
fi

echo ""
echo "推荐配置国内镜像源:"
echo "1. 创建或编辑 /etc/docker/daemon.json"
echo "2. 添加以下内容:"
echo '{'
echo '  "registry-mirrors": ['
echo '    "https://docker.mirrors.ustc.edu.cn",'
echo '    "https://hub-mirror.c.163.com",'
echo '    "https://mirror.baidubce.com"'
echo '  ]'
echo '}'
echo "3. 重启Docker: sudo systemctl restart docker"

# 方法3: 使用不同的基础镜像
echo ""
echo "方法3: 使用不同的基础镜像..."
echo "如果python:3.12-slim无法下载，可以尝试:"
echo "- python:3.11-slim (已更新到Dockerfile)"
echo "- python:3.10-slim"
echo "- python:3.9-slim"

# 方法4: 手动拉取镜像
echo ""
echo "方法4: 手动拉取镜像..."
echo "尝试手动拉取镜像:"
echo "docker pull python:3.11-slim"

# 方法5: 清理Docker缓存
echo ""
echo "方法5: 清理Docker缓存..."
echo "如果问题持续，可以清理Docker缓存:"
echo "docker system prune -a"

echo ""
echo "=== 解决方案总结 ==="
echo "1. 检查网络连接"
echo "2. 配置国内镜像源"
echo "3. 使用python:3.11-slim (已更新)"
echo "4. 手动拉取镜像"
echo "5. 清理Docker缓存"
echo ""
echo "建议按顺序尝试上述方法" 