#!/usr/bin/env python
"""
测试脚本：验证深度图生成脚本的路径配置是否正确
"""
import os
import sys

def test_paths():
    """测试路径配置"""
    print("="*60)
    print("Testing Path Configuration")
    print("="*60)
    
    # 从notes.txt读取路径
    notes_path = "doc/notes.txt"
    if not os.path.exists(notes_path):
        print(f"Error: {notes_path} not found")
        return False
    
    print(f"\n1. Reading paths from {notes_path}...")
    with open(notes_path, 'r') as f:
        lines = f.readlines()
    
    student_path = None
    da3_path = None
    data_path = None
    
    for i, line in enumerate(lines):
        if "学生模型权重保存路径" in line:
            student_path = line.split(':')[-1].strip()
        elif "DA3模型路径" in line and i + 1 < len(lines):
            da3_path = lines[i + 1].strip()
        elif "数据集路径" in line and i + 1 < len(lines):
            data_path = lines[i + 1].strip()
    
    print(f"   Student model path: {student_path}")
    print(f"   DA3 model path: {da3_path}")
    print(f"   Data path: {data_path}")
    
    # 测试路径是否存在
    print("\n2. Testing path existence...")
    all_ok = True
    
    if student_path:
        student_path = os.path.expanduser(student_path)
        if os.path.exists(student_path):
            print(f"   ✓ Student model path exists: {student_path}")
            # 检查权重文件
            encoder_path = os.path.join(student_path, "encoder.pth")
            if os.path.exists(encoder_path):
                print(f"     ✓ encoder.pth found")
            else:
                print(f"     ✗ encoder.pth NOT found")
                all_ok = False
        else:
            print(f"   ✗ Student model path does NOT exist: {student_path}")
            all_ok = False
    else:
        print("   ✗ Student model path not found in notes.txt")
        all_ok = False
    
    if da3_path:
        da3_path = os.path.expanduser(da3_path)
        if os.path.exists(da3_path):
            print(f"   ✓ DA3 model path exists: {da3_path}")
        else:
            print(f"   ✗ DA3 model path does NOT exist: {da3_path}")
            print("     (Will try to load from HuggingFace)")
    else:
        print("   ⚠ DA3 model path not found in notes.txt (will use HuggingFace)")
    
    if data_path:
        data_path = os.path.expanduser(data_path)
        if os.path.exists(data_path):
            print(f"   ✓ Data path exists: {data_path}")
            # 检查测试集文件夹
            test_folder = os.path.join(data_path, "nyu2_test")
            if os.path.exists(test_folder):
                print(f"     ✓ nyu2_test folder found")
                # 检查测试图片
                test_image = os.path.join(test_folder, "00000_colors.png")
                if os.path.exists(test_image):
                    print(f"     ✓ Test image found: {test_image}")
                else:
                    print(f"     ✗ Test image NOT found: {test_image}")
                    # 列出文件夹中的一些文件
                    files = os.listdir(test_folder)[:5]
                    print(f"     Files in folder: {files}...")
            else:
                print(f"     ✗ nyu2_test folder NOT found")
                all_ok = False
        else:
            print(f"   ✗ Data path does NOT exist: {data_path}")
            all_ok = False
    else:
        print("   ✗ Data path not found in notes.txt")
        all_ok = False
    
    # 测试splits文件
    print("\n3. Testing splits file...")
    splits_file = "splits/nyu/test_files.txt"
    if os.path.exists(splits_file):
        print(f"   ✓ Splits file exists: {splits_file}")
        with open(splits_file, 'r') as f:
            lines = f.readlines()
        print(f"     Total test files: {len(lines)}")
        if len(lines) > 0:
            first_line = lines[0].strip()
            print(f"     First entry: {first_line}")
            # 检查对应的图片是否存在
            parts = first_line.split()
            if len(parts) >= 2:
                folder = parts[0]
                frame_index = parts[1]
                test_image_path = os.path.join(data_path, folder, f"{frame_index}_colors.png")
                if os.path.exists(test_image_path):
                    print(f"     ✓ First test image exists: {test_image_path}")
                else:
                    print(f"     ✗ First test image NOT found: {test_image_path}")
                    all_ok = False
    else:
        print(f"   ✗ Splits file NOT found: {splits_file}")
        all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ All paths are valid! You can run generate_depth_maps.py")
        return True
    else:
        print("✗ Some paths are invalid. Please check the configuration.")
        return False

if __name__ == '__main__':
    success = test_paths()
    sys.exit(0 if success else 1)

