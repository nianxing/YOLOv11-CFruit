#!/usr/bin/env python3
"""
快速标签重命名脚本
将JSON文件中的"youcha"标签替换为"cfruit"
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """快速执行标签重命名"""
    
    # 获取当前脚本所在目录的父目录（项目根目录）
    project_root = Path(__file__).parent.parent
    
    print("=== 油茶果标签重命名工具 ===")
    print(f"项目根目录: {project_root}")
    
    # 询问用户要处理的目录
    print("\n请选择要处理的目录:")
    print("1. 整个项目目录 (推荐)")
    print("2. data 目录")
    print("3. 自定义目录")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        target_dir = str(project_root)
    elif choice == "2":
        target_dir = str(project_root / "data")
    elif choice == "3":
        custom_dir = input("请输入目录路径: ").strip()
        if os.path.exists(custom_dir):
            target_dir = custom_dir
        else:
            print(f"错误: 目录不存在 {custom_dir}")
            return
    else:
        print("无效选择")
        return
    
    print(f"\n目标目录: {target_dir}")
    
    # 询问是否要试运行
    print("\n是否要试运行（不实际修改文件）?")
    dry_run_choice = input("输入 'y' 进行试运行，直接回车执行实际修改: ").strip().lower()
    
    dry_run = dry_run_choice == 'y'
    
    # 构建命令
    script_path = project_root / "scripts" / "rename_labels.py"
    cmd = [
        sys.executable, str(script_path),
        "--directory", target_dir,
        "--old-label", "youcha",
        "--new-label", "cfruit",
        "--verbose"
    ]
    
    if dry_run:
        cmd.append("--dry-run")
        print("\n=== 试运行模式 ===")
    else:
        print("\n=== 执行实际修改 ===")
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 确认执行
    if not dry_run:
        confirm = input("\n确认要执行修改吗? (y/N): ").strip().lower()
        if confirm != 'y':
            print("已取消操作")
            return
    
    # 执行脚本
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.stdout:
            print("\n=== 输出 ===")
            print(result.stdout)
        
        if result.stderr:
            print("\n=== 错误 ===")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n=== 执行成功 ===")
            if dry_run:
                print("试运行完成，请检查输出结果")
                print("如需实际执行，请重新运行并选择不试运行")
            else:
                print("标签重命名完成！")
                print("原文件已备份为 .backup 文件")
        else:
            print(f"\n=== 执行失败 (返回码: {result.returncode}) ===")
            
    except Exception as e:
        print(f"执行脚本时出错: {e}")

if __name__ == '__main__':
    main() 