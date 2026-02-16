#!/usr/bin/env python3
"""Trim datasets.py to remove duplicate preprocessing code"""

file_path = r'e:\Chadi\Projects\websites\AutoMl\backend\app\routes\datasets.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Original file: {len(lines)} lines")

# Keep only first 997 lines (0-996 index)
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines[:997])

print(f"Trimmed to: 997 lines")
print("✅ Removed duplicate preprocessing code!")
