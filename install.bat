@echo off
echo 正在安装 YOLOv8-CFruit 依赖...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo Python版本:
python --version
echo.

REM 升级pip
echo 升级pip...
python -m pip install --upgrade pip

REM 安装基础依赖
echo 安装基础依赖...
pip install PyYAML torch torchvision

REM 安装项目依赖
echo 安装项目依赖...
pip install -r requirements.txt

REM 安装项目
echo 安装项目...
pip install -e .

echo.
echo 安装完成！
echo.
echo 运行测试:
echo python test_project.py
echo.
pause 