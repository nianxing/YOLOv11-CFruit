# 标签重命名工具使用说明

这个工具用于将JSON文件中的"youcha"标签替换为"cfruit"标签。

## 文件说明

- `rename_labels.py` - 主要的标签重命名脚本
- `quick_rename_labels.py` - 快速执行脚本（推荐使用）

## 使用方法

### 方法1: 使用快速脚本（推荐）

```bash
# 在项目根目录下执行
python scripts/quick_rename_labels.py
```

这个脚本会引导你：
1. 选择要处理的目录
2. 选择是否试运行
3. 确认执行修改

### 方法2: 直接使用主脚本

```bash
# 试运行（不实际修改文件）
python scripts/rename_labels.py --directory data --dry-run --verbose

# 实际执行修改
python scripts/rename_labels.py --directory data --verbose

# 处理整个项目目录
python scripts/rename_labels.py --directory . --verbose

# 自定义标签替换
python scripts/rename_labels.py --old-label youcha --new-label cfruit --directory data
```

## 参数说明

- `--directory, -d`: 要处理的目录路径（默认：当前目录）
- `--old-label`: 要替换的旧标签（默认：youcha）
- `--new-label`: 新的标签名称（默认：cfruit）
- `--dry-run`: 试运行模式，不实际修改文件
- `--verbose, -v`: 显示详细信息

## 功能特点

1. **递归搜索**: 自动搜索指定目录下的所有JSON文件
2. **智能替换**: 递归遍历JSON结构，替换所有匹配的标签
3. **安全备份**: 修改前自动创建 `.backup` 备份文件
4. **试运行**: 支持试运行模式，预览修改效果
5. **详细日志**: 显示处理过程和结果统计

## 示例输出

```
=== 油茶果标签重命名工具 ===
项目根目录: /path/to/YOLOv11-CFruit

请选择要处理的目录:
1. 整个项目目录 (推荐)
2. data 目录
3. 自定义目录

请输入选择 (1-3): 1

目标目录: /path/to/YOLOv11-CFruit

是否要试运行（不实际修改文件）?
输入 'y' 进行试运行，直接回车执行实际修改: y

=== 试运行模式 ===
执行命令: python scripts/rename_labels.py --directory /path/to/YOLOv11-CFruit --old-label youcha --new-label cfruit --verbose --dry-run

=== 输出 ===
2024-01-01 12:00:00 - INFO - 在目录 /path/to/YOLOv11-CFruit 中查找JSON文件...
2024-01-01 12:00:00 - INFO - 找到 5 个JSON文件
2024-01-01 12:00:00 - INFO - [DRY RUN] 将修改文件: data/annotations.json
2024-01-01 12:00:00 - INFO - [DRY RUN] 替换 'youcha' -> 'cfruit'
...

=== 处理完成 ===
总文件数: 5
修改文件数: 3
替换标签: 'youcha' -> 'cfruit'
这是试运行模式，未实际修改文件
```

## 注意事项

1. **备份文件**: 修改前会自动创建 `.backup` 备份文件
2. **试运行**: 建议先使用试运行模式检查效果
3. **编码**: 脚本使用UTF-8编码处理文件
4. **错误处理**: 遇到JSON解析错误时会跳过该文件并记录错误

## 恢复备份

如果需要恢复原文件：

```bash
# 恢复单个文件
cp data/annotations.json.backup data/annotations.json

# 批量恢复所有备份文件
find . -name "*.json.backup" -exec bash -c 'cp "$1" "${1%.backup}"' _ {} \;
```

## 支持的JSON结构

脚本支持各种JSON结构中的标签替换：

```json
{
  "annotations": [
    {
      "label": "youcha",  // 会被替换为 "cfruit"
      "bbox": [100, 100, 200, 200]
    }
  ],
  "metadata": {
    "classes": ["youcha"],  // 会被替换为 ["cfruit"]
    "version": "1.0"
  }
}
``` 