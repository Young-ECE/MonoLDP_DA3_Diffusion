#!/bin/bash
# 升级 Python 版本以支持 DA3

echo "=========================================="
echo "升级 Python 版本以支持 Depth Anything V3"
echo "=========================================="
echo ""

# 检查当前版本
echo "当前 Python 版本:"
python --version
echo ""

# 确认升级
read -p "是否升级到 Python 3.10? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "取消升级"
    exit 0
fi

echo ""
echo "开始升级 Python..."

# 升级 Python
conda install python=3.10 -y

# 验证新版本
echo ""
echo "升级后的 Python 版本:"
python --version

# 重新安装 pip 包
echo ""
echo "重新安装依赖包..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "升级完成！"
echo "=========================================="
echo ""
echo "现在可以安装 DA3:"
echo "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

