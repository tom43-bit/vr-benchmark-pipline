#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import glob

def move_csv_files():
    # 获取当前目录
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    # 目标文件夹路径
    target_dir = os.path.join(parent_dir, 'csvs')
    
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建文件夹: {target_dir}")
    
    # 使用glob查找所有.csv文件
    csv_files = glob.glob(os.path.join(parent_dir, '*.csv'))
    
    # 过滤掉已经在目标文件夹中的文件（防止移动已经移动过的文件）
    csv_files = [f for f in csv_files if not f.startswith(target_dir)]
    
    if not csv_files:
        print("当前目录下没有找到.csv文件")
        return
    
    # 移动每个文件
    moved_count = 0
    for file_path in csv_files:
        try:
            file_name = os.path.basename(file_path)
            destination = os.path.join(target_dir, file_name)
            
            # 如果目标文件已存在，添加数字后缀
            if os.path.exists(destination):
                base_name, ext = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(os.path.join(target_dir, f"{base_name}_{counter}{ext}")):
                    counter += 1
                destination = os.path.join(target_dir, f"{base_name}_{counter}{ext}")
            
            shutil.move(file_path, destination)
            print(f"已移动: {file_name} -> csvs/")
            moved_count += 1
            
        except Exception as e:
            print(f"移动文件 {file_name} 时出错: {e}")
    
    print(f"\n完成！共移动了 {moved_count} 个文件到 ./csvs 文件夹")

if __name__ == "__main__":
    move_csv_files()