#!/usr/bin/env python
# 文件名: add_indent.py

import sys

def add_indent_to_block(filename, start_line, end_line, spaces=4):
    """
    给指定行范围添加缩进
    
    Args:
        filename: 文件名
        start_line: 起始行号（从1开始）
        end_line: 结束行号
        spaces: 缩进空格数
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # 添加缩进
    for i in range(start_line-1, min(end_line, len(lines))):
        lines[i] = ' ' * spaces + lines[i]
    
    # 写回文件
    with open(filename, 'w') as f:
        f.writelines(lines)
    
    print(f"已为 {filename} 的第 {start_line}-{end_line} 行添加 {spaces} 空格缩进")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python add_indent.py <文件名> <起始行> <结束行>")
        sys.exit(1)
    
    filename = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    add_indent_to_block(filename, start, end)