# Windows Docker环境设置指南

## 概述
本指南将帮助您在Windows系统上设置Docker环境来运行YOLOv11-CFruit项目。

## 系统要求
- Windows 10/11 64位专业版、企业版或教育版
- 至少4GB RAM
- 支持虚拟化的CPU
- 管理员权限

## 步骤1：启用Windows功能

### 方法1：使用PowerShell脚本（推荐）
1. 以管理员身份打开PowerShell
2. 运行项目根目录下的设置脚本：
```powershell
.\setup_docker_windows.ps1
```

### 方法2：手动启用
1. 打开"控制面板" → "程序" → "启用或关闭Windows功能"
2. 启用以下功能：
   - Hyper-V
   - 容器
   - Windows子系统for Linux
   - 虚拟机平台

## 步骤2：安装Docker Desktop

1. **下载Docker Desktop**
   - 访问 [Docker官网](https://www.docker.com/products/docker-desktop/)
   - 下载Windows版本的Docker Desktop

2. **安装Docker Desktop**
   - 运行下载的安装程序
   - 按照安装向导完成安装
   - 重启计算机

3. **启动Docker Desktop**
   - 从开始菜单启动Docker Desktop
   - 等待Docker服务完全启动（托盘图标变为绿色）

## 步骤3：验证安装

运行验证脚本检查Docker是否正确安装：
```powershell
.\verify_docker.ps1
```

应该看到类似以下输出：
```
Docker version 24.0.0, build 2ff0b2c
Docker Compose version v2.20.0
Hello from Docker!
```

## 步骤4：运行YOLOv11-CFruit项目

### 快速启动
运行项目根目录下的启动脚本：
```powershell
.\run_docker.ps1
```

### 手动启动

#### CPU版本（推荐用于测试）
```powershell
# 构建并启动容器
docker-compose up -d yolov11-cfruit

# 进入容器
docker exec -it yolov11-cfruit bash
```

#### GPU版本（需要NVIDIA GPU）
```powershell
# 构建并启动GPU容器
docker-compose up -d yolov11-cfruit-gpu

# 进入容器
docker exec -it yolov11-cfruit-gpu bash
```

## 步骤5：在容器内使用项目

进入容器后，您可以运行以下命令：

### 测试项目
```bash
python test_project.py
```

### 准备数据
```bash
python scripts/prepare_data.py
```

### 训练模型
```bash
python scripts/train.py
```

### 运行推理
```bash
python examples/basic_detection.py
```

## 常用Docker命令

### 查看容器状态
```powershell
docker ps
```

### 停止容器
```powershell
docker-compose down
```

### 查看容器日志
```powershell
docker logs yolov11-cfruit
```

### 重新构建镜像
```powershell
docker-compose build --no-cache
```

## 故障排除

### 问题1：Docker Desktop无法启动
- 确保已启用Hyper-V和容器功能
- 检查BIOS中的虚拟化设置
- 重启计算机

### 问题2：容器启动失败
- 检查端口8888是否被占用
- 确保有足够的磁盘空间
- 查看Docker日志：`docker logs <container_name>`

### 问题3：GPU版本无法使用GPU
- 确保安装了NVIDIA驱动
- 安装nvidia-docker2
- 检查CUDA版本兼容性

### 问题4：权限问题
- 确保以管理员身份运行PowerShell
- 检查Docker Desktop设置中的文件共享权限

## 数据持久化

项目使用以下卷挂载来保持数据：
- `./data` → `/workspace/data` (数据集)
- `./checkpoints` → `/workspace/checkpoints` (模型权重)
- `./output` → `/workspace/output` (输出结果)

## 性能优化

### CPU版本优化
- 在Docker Desktop设置中增加内存分配
- 增加CPU核心数分配

### GPU版本优化
- 确保使用最新的NVIDIA驱动
- 在Docker Desktop中启用GPU支持

## 安全注意事项

1. 不要在容器中运行敏感操作
2. 定期更新Docker Desktop
3. 使用官方镜像作为基础镜像
4. 不要在容器中存储敏感数据

## 获取帮助

如果遇到问题，请：
1. 查看Docker Desktop日志
2. 检查项目文档
3. 查看GitHub Issues
4. 联系项目维护者 