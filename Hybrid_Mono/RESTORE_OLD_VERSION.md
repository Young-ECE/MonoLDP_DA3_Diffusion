# 恢复旧版本训练逻辑指南

## 问题分析

### 主要差异

1. **`photometric_weight` fallback值**：
   - 旧版本：0.5（在options.py中）
   - 当前版本：0.2（在trainer.py的fallback中）✅ **已修复为0.5**

2. **`compute_losses` 中的类型初始化**：
   - 旧版本：`total_loss = 0`（Python int）
   - 当前版本：`total_loss = torch.tensor(0.0).to(self.device)`（Tensor）

3. **`set_train()` 方法**：
   - 旧版本：所有模型都设置为train模式
   - 当前版本：教师模型保持eval模式

### 为什么旧版本loss初始值更大？

1. **photometric_weight=0.5 vs 0.2**：
   - 如果photometric_loss≈2.5，alignment_loss≈0.5
   - 旧版本：0.5 * 2.5 + 0.5 = 1.75（多scale平均后可能≈3.0）
   - 当前版本（修复前）：0.2 * 2.5 + 0.5 = 1.0（多scale平均后可能≈0.9）

2. **更大的loss值提供更强的梯度信号**，有助于快速下降

## 恢复方案

### 方案1：只修复photometric_weight（推荐，已修复）

已修复 `trainer.py` 中的 `photometric_weight` fallback值从0.2改为0.5。

### 方案2：完全恢复旧版本逻辑

如果需要完全恢复旧版本的训练逻辑，需要修改以下部分：

#### 2.1 恢复 `compute_losses` 中的类型初始化

```python
# 当前版本（修复后）
total_loss = torch.tensor(0.0).to(self.device)
loss = torch.tensor(0.0).to(self.device)

# 恢复为旧版本
total_loss = 0
loss = 0
```

**注意**：这会导致类型不一致问题，如果所有损失被禁用会出错。

#### 2.2 恢复 `set_train()` 方法

```python
# 当前版本（修复后）
def set_train(self):
    for name, m in self.models.items():
        if name == "depth_anything_v3_teacher":
            m.eval()
        else:
            m.train()

# 恢复为旧版本
def set_train(self):
    for m in self.models.values():
        m.train()
```

**注意**：这会导致教师模型被错误设置为train模式。

## 建议

**优先使用方案1**（已修复）：
- 只修改了 `photometric_weight` 的fallback值
- 保持了类型安全性和教师模型的正确模式
- 应该能恢复旧版本的loss初始值和下降趋势

如果方案1不行，再考虑方案2，但需要注意类型安全问题。

