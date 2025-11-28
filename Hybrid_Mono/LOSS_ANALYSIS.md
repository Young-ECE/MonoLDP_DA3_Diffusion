# Loss差异分析：旧版本 vs 当前版本

## 问题描述

- **旧版本（1d9cdca）**：loss初始值约3.多，快速下降
- **当前版本**：loss初始值约0.9，震荡不下降

## 关键发现

### 1. `photometric_weight` 默认值差异 ⚠️ **关键差异**

**旧版本（1d9cdca）：**
```python
# options.py
self.parser.add_argument("--photometric_weight",
                        type=float,
                        default=0.5,  # ✅ 默认值0.5
                        ...)
```

**当前版本：**
```python
# trainer.py (第718行)
photometric_weight = getattr(self.opt, 'photometric_weight', 0.2)  # ❌ fallback值0.2
```

**影响分析：**

假设初始状态：
- `photometric_loss` ≈ 2.5（包含reprojection + smoothness + plane + line）
- `alignment_loss` ≈ 0.5（teacher-student L1 + MSE + SSIM + DDIM）

**旧版本（photometric_weight=0.5）：**
```
total_loss = 0.5 * 2.5 + 0.5 = 1.25 + 0.5 = 1.75
```
但实际可能是多个scale的平均，或者还有其他损失项，所以初始值约3.多。

**当前版本（photometric_weight=0.2）：**
```
total_loss = 0.2 * 2.5 + 0.5 = 0.5 + 0.5 = 1.0
```
但实际可能更小，约0.9。

### 2. 损失计算逻辑对比

两个版本的损失计算逻辑**完全相同**：
```python
scale_complete_loss = (
    photometric_weight * scale_photometric_loss +
    scale_teacher_student_losses[scale]
)
losses["loss"] = total_loss_all_scales / len(self.opt.scales)
```

### 3. 为什么旧版本能快速下降？

1. **更大的初始loss值**：
   - photometric_weight=0.5 使得初始loss更大
   - 更大的loss意味着更大的梯度信号
   - 模型更容易找到优化方向

2. **更好的损失平衡**：
   - photometric_weight=0.5 在几何损失和对齐损失之间取得更好的平衡
   - photometric_weight=0.2 可能过于偏向对齐损失，导致几何一致性不足

3. **训练稳定性**：
   - 更大的loss值可能提供更稳定的梯度信号
   - 较小的loss值可能导致梯度信号过弱，容易震荡

## 解决方案

### 方案1：恢复旧版本的photometric_weight默认值（推荐）

修改 `options.py`，将 `photometric_weight` 的默认值改为 `0.5`：

```python
self.parser.add_argument("--photometric_weight",
                        type=float,
                        default=0.5,  # 恢复为0.5
                        ...)
```

同时修改 `trainer.py` 中的fallback值：

```python
photometric_weight = getattr(self.opt, 'photometric_weight', 0.5)  # 改为0.5
```

### 方案2：完全恢复旧版本的训练逻辑

如果需要完全恢复旧版本的训练逻辑，可以：
1. 恢复旧版本的 `compute_losses` 方法（使用Python int初始化）
2. 恢复旧版本的 `set_train` 方法（所有模型都train模式）
3. 恢复旧版本的 `photometric_weight` 默认值（0.5）

## 建议

**优先尝试方案1**：只修改 `photometric_weight` 的默认值，这应该就能解决loss初始值过小和震荡的问题。

如果方案1不行，再考虑方案2（完全恢复旧版本逻辑）。

