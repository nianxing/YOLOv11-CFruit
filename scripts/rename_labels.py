#!/usr/bin/env python3
"""
标签重命名脚本
将JSON文件中的"youcha"标签替换为"cfruit"
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def find_json_files(directory: str) -> List[str]:
    """递归查找目录下的所有JSON文件"""
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files


def replace_label_in_json(json_data: Any, old_label: str, new_label: str) -> tuple[Any, bool]:
    """
    递归替换JSON数据中的标签
    返回: (修改后的数据, 是否发生修改)
    """
    modified = False
    
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if isinstance(value, str) and value == old_label:
                json_data[key] = new_label
                modified = True
            elif isinstance(value, (dict, list)):
                new_value, was_modified = replace_label_in_json(value, old_label, new_label)
                if was_modified:
                    json_data[key] = new_value
                    modified = True
    elif isinstance(json_data, list):
        for i, item in enumerate(json_data):
            if isinstance(item, str) and item == old_label:
                json_data[i] = new_label
                modified = True
            elif isinstance(item, (dict, list)):
                new_item, was_modified = replace_label_in_json(item, old_label, new_label)
                if was_modified:
                    json_data[i] = new_item
                    modified = True
    
    return json_data, modified


def process_json_file(file_path: str, old_label: str, new_label: str, dry_run: bool = False) -> bool:
    """
    处理单个JSON文件
    返回: 是否发生修改
    """
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 替换标签
        modified_data, was_modified = replace_label_in_json(data, old_label, new_label)
        
        if was_modified:
            if dry_run:
                logging.info(f"[DRY RUN] 将修改文件: {file_path}")
                logging.info(f"[DRY RUN] 替换 '{old_label}' -> '{new_label}'")
            else:
                # 备份原文件
                backup_path = file_path + '.backup'
                if not os.path.exists(backup_path):
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    logging.info(f"已创建备份: {backup_path}")
                
                # 保存修改后的文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(modified_data, f, ensure_ascii=False, indent=2)
                logging.info(f"已修改文件: {file_path}")
                logging.info(f"替换 '{old_label}' -> '{new_label}'")
            
            return True
        else:
            logging.debug(f"文件无需修改: {file_path}")
            return False
            
    except json.JSONDecodeError as e:
        logging.error(f"JSON解析错误 {file_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"处理文件错误 {file_path}: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将JSON文件中的标签从"youcha"替换为"cfruit"')
    parser.add_argument('--directory', '-d', type=str, default='.',
                       help='要处理的目录路径 (默认: 当前目录)')
    parser.add_argument('--old-label', type=str, default='youcha',
                       help='要替换的旧标签 (默认: youcha)')
    parser.add_argument('--new-label', type=str, default='cfruit',
                       help='新的标签名称 (默认: cfruit)')
    parser.add_argument('--dry-run', action='store_true',
                       help='试运行模式，不实际修改文件')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    # 检查目录是否存在
    if not os.path.exists(args.directory):
        logging.error(f"目录不存在: {args.directory}")
        return
    
    # 查找JSON文件
    logging.info(f"在目录 {args.directory} 中查找JSON文件...")
    json_files = find_json_files(args.directory)
    
    if not json_files:
        logging.warning(f"在目录 {args.directory} 中未找到JSON文件")
        return
    
    logging.info(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理文件
    modified_count = 0
    total_count = len(json_files)
    
    for file_path in json_files:
        if process_json_file(file_path, args.old_label, args.new_label, args.dry_run):
            modified_count += 1
    
    # 输出统计信息
    logging.info(f"\n=== 处理完成 ===")
    logging.info(f"总文件数: {total_count}")
    logging.info(f"修改文件数: {modified_count}")
    logging.info(f"替换标签: '{args.old_label}' -> '{args.new_label}'")
    
    if args.dry_run:
        logging.info("这是试运行模式，未实际修改文件")
        logging.info("使用 --dry-run 参数来实际执行修改")
    else:
        logging.info("所有修改已完成")
        logging.info("原文件已备份为 .backup 文件")


if __name__ == '__main__':
    main() 