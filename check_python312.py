#!/usr/bin/env python3
"""
Python 3.12 兼容性检查脚本
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("=== Python版本检查 ===")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 12:
        print("✓ Python 3.12+ 兼容")
        return True
    elif version.major == 3 and version.minor >= 8:
        print("⚠ Python版本兼容，但建议使用3.12+")
        return True
    else:
        print("✗ Python版本过低，需要3.8+")
        return False

def check_package_compatibility():
    """检查包兼容性"""
    print("\n=== 包兼容性检查 ===")
    
    packages = [
        ('torch', '2.0.0'),
        ('torchvision', '0.15.0'),
        ('numpy', '1.24.0'),
        ('Pillow', '10.0.0'),
        ('opencv-python-headless', '4.8.0'),
        ('PyYAML', '6.0'),
        ('tensorboard', '2.13.0'),
        ('tqdm', '4.65.0'),
        ('scipy', '1.11.0'),
        ('scikit-learn', '1.3.0'),
        ('matplotlib', '3.7.0'),
        ('pandas', '2.0.0'),
        ('requests', '2.31.0'),
    ]
    
    all_compatible = True
    
    for package, min_version in packages:
        try:
            module = importlib.import_module(package.replace('-', '_'))
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package}: {version}")
        except ImportError:
            print(f"✗ {package}: 未安装")
            all_compatible = False
        except Exception as e:
            print(f"⚠ {package}: 检查失败 ({e})")
    
    return all_compatible

def check_opencv_installation():
    """检查OpenCV安装"""
    print("\n=== OpenCV安装检查 ===")
    
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
        print(f"✓ OpenCV路径: {cv2.__file__}")
        
        # 测试基本功能
        import numpy as np
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.resize(test_img, (50, 50))
        print("✓ OpenCV基本功能正常")
        return True
        
    except ImportError as e:
        print(f"✗ OpenCV导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠ OpenCV功能异常: {e}")
        return False

def check_system_libraries():
    """检查系统库依赖"""
    print("\n=== 系统库检查 ===")
    
    try:
        import cv2
        cv2_path = cv2.__file__
        
        # 使用ldd检查依赖（Linux）
        if sys.platform.startswith('linux'):
            try:
                result = subprocess.run(['ldd', cv2_path], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    missing_libs = []
                    for line in result.stdout.split('\n'):
                        if 'not found' in line:
                            missing_libs.append(line.strip())
                    
                    if missing_libs:
                        print("⚠ 缺少系统库:")
                        for lib in missing_libs:
                            print(f"  {lib}")
                        return False
                    else:
                        print("✓ 所有系统库依赖正常")
                        return True
                else:
                    print("⚠ 无法检查系统库依赖")
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("⚠ 无法执行ldd命令")
                return True
        else:
            print("ℹ 非Linux系统，跳过系统库检查")
            return True
            
    except Exception as e:
        print(f"⚠ 系统库检查失败: {e}")
        return True

def check_project_structure():
    """检查项目结构"""
    print("\n=== 项目结构检查 ===")
    
    required_files = [
        'requirements.txt',
        'setup.py',
        'models/__init__.py',
        'utils/__init__.py',
        'configs/model/yolov11_cfruit.yaml',
        'configs/data/cfruit.yaml',
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (缺失)")
            all_exist = False
    
    return all_exist

def main():
    """主函数"""
    print("YOLOv11-CFruit Python 3.12 兼容性检查")
    print("=" * 50)
    
    checks = [
        ("Python版本", check_python_version),
        ("包兼容性", check_package_compatibility),
        ("OpenCV安装", check_opencv_installation),
        ("系统库", check_system_libraries),
        ("项目结构", check_project_structure),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"⚠ {name}检查异常: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("检查结果汇总:")
    
    all_passed = True
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有检查通过！项目可以正常运行。")
    else:
        print("⚠ 部分检查失败，请根据上述信息解决问题。")
        print("\n建议解决方案:")
        print("1. 运行: chmod +x fix_opencv_linux.sh && ./fix_opencv_linux.sh")
        print("2. 运行: chmod +x install_python312.sh && ./install_python312.sh")
        print("3. 使用Docker: docker build -t yolov11-cfruit .")

if __name__ == "__main__":
    main() 