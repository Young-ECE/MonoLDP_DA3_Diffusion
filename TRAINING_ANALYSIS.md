# MonoLDP 训练问题分析与解决方案

## 问题1: 280000步时所有loss显著提升，输出结果急剧恶化

### 可能原因分析

#### 1.1 学习率调度器问题（最可能）
**当前配置**：
- 使用 `StepLR`，每20个epoch降低10倍（`gamma=0.1`）
- 280000步 ≈ 280000 / (数据集大小 / batch_size) 个epoch

**问题**：
- StepLR在特定epoch突然大幅降低学习率（降低10倍）
- 如果此时模型处于不稳定状态，突然的学习率下降可能导致：
  - 梯度更新方向错误
  - 模型陷入局部最优
  - 训练崩溃

**证据**：
- 所有loss同时上升，说明是全局性问题
- 发生在特定步数，符合学习率调度器的周期性行为

#### 1.2 梯度爆炸/数值不稳定
- 长时间训练可能导致梯度累积
- 扩散模型的噪声预测可能在某些情况下不稳定

#### 1.3 优化器状态问题
- Adam优化器的动量状态可能累积了错误信息
- 长时间训练后，优化器状态可能不再适合当前参数

### 解决方案

#### 方案1: 改用更平滑的学习率调度器（推荐）
参考MonoDiffusion的实现，使用ChainedScheduler（warmup + cosine annealing）：

```python
# 优点：
# 1. 平滑的学习率衰减，避免突然下降
# 2. Warm restart机制，帮助跳出局部最优
# 3. 更稳定的训练过程
```

#### 方案2: 调整StepLR参数
- 减小gamma（如0.5而不是0.1）
- 增加step_size（如30-40个epoch）
- 添加学习率下限（min_lr）

#### 方案3: 添加梯度裁剪
```python
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
```

#### 方案4: 检查点恢复策略
- 在280000步之前保存检查点
- 如果发生崩溃，从检查点恢复并调整学习率

---

## 问题2: 学生模型输出没有很好跟随教师模型

### 可能原因分析

#### 2.1 Loss权重不平衡
**当前配置**：
```python
losses["loss"] = (1.0 * losses['photometric'] + 
                 self.opt.diffusion_l1_weight * losses['l1'] + 
                 self.opt.diffusion_ddim_weight * losses['ddim'])
```

**问题**：
- `photometric` loss可能占主导地位
- L1 loss权重（默认1.0）可能不够
- 需要平衡不同loss的尺度

#### 2.2 教师模型输出质量
- 检查教师模型的输出是否稳定
- 教师模型的输出范围是否与学生模型匹配

#### 2.3 训练策略问题
- 可能需要分阶段训练：
  1. 先让模型学习跟随教师（高L1权重）
  2. 再学习几何一致性（高photometric权重）

### 解决方案

#### 方案1: 调整Loss权重（推荐）
```python
# 建议配置：
--diffusion_l1_weight 2.0  # 增加L1权重，强制跟随教师
--diffusion_ddim_weight 1.0  # 保持DDIM权重
# photometric权重保持1.0
```

#### 方案2: 使用加权L1 Loss
对不同区域使用不同权重：
- 边缘区域：高权重
- 平滑区域：低权重

#### 方案3: 分阶段训练
```python
# 阶段1（前50%训练）：高L1权重，低photometric权重
--diffusion_l1_weight 5.0
--smoothness_weight 0.1

# 阶段2（后50%训练）：降低L1权重，增加photometric权重
--diffusion_l1_weight 1.0
--smoothness_weight 0.2
```

#### 方案4: 添加一致性损失
```python
# 计算学生和教师输出的结构相似性
ssim_loss = SSIM(student_disp, teacher_disp)
loss += consistency_weight * ssim_loss
```

---

## 问题3: DepthEstimation vs MonoDiffusion 在Diffusion使用上的关键差异

### 关键差异对比

| 特性 | DepthEstimation | MonoDiffusion | 影响 |
|------|----------------|---------------|------|
| **优化器** | Adam | AdamW (weight_decay=1e-2) | ⚠️ 重要：AdamW有正则化效果 |
| **学习率调度** | StepLR (gamma=0.1, step=20) | ChainedScheduler (warmup+cosine) | ⚠️ 重要：平滑衰减vs突然下降 |
| **Loss权重** | L1=1.0, DDIM=1.0 | 需要查看 | - |
| **训练步数** | 单次前向 | 单次前向 | 相同 |
| **Mask训练** | 未实现 | 有mask训练 | ⚠️ 可能影响泛化 |
| **编码器** | ResNet | LiteMono | 架构差异 |
| **扩散步数** | [5,4,3] | [5,4,3] | 相同 |
| **时间步数** | [250,200,150] | [250,200,150] | 相同 |

### 关键疏漏

#### 1. 缺少Mask训练（重要）
MonoDiffusion中有mask训练机制：
```python
outputs_mask, losses_mask = self.process_batch_mask(inputs, outputs)
```
这有助于模型学习处理遮挡和边界情况。

#### 2. 优化器选择
- **AdamW vs Adam**: AdamW有weight decay，有助于正则化
- **Weight decay**: MonoDiffusion使用1e-2，有助于防止过拟合

#### 3. 学习率调度策略
- **StepLR**: 突然的学习率下降可能导致训练不稳定
- **ChainedScheduler**: 平滑衰减 + warm restart，更稳定

#### 4. 可能缺少的损失项
需要检查MonoDiffusion是否有额外的损失项或训练技巧。

---

## 综合修复建议

### 优先级1: 修复学习率调度器（紧急）
```python
# 方案A: 改用ChainedScheduler（推荐）
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler

self.model_lr_scheduler = ChainedScheduler(
    self.model_optimizer,
    T_0=15,  # 初始周期
    T_mul=1,
    eta_min=1e-6,  # 最小学习率
    last_epoch=-1,
    max_lr=self.opt.learning_rate,
    warmup_steps=1000,  # warmup步数
    gamma=0.9
)

# 方案B: 改进StepLR（临时方案）
self.model_lr_scheduler = optim.lr_scheduler.StepLR(
    self.model_optimizer, 
    step_size=30,  # 增加步长
    gamma=0.5  # 减小衰减幅度
)
```

### 优先级2: 改用AdamW优化器
```python
self.model_optimizer = optim.AdamW(
    self.parameters_to_train, 
    self.opt.learning_rate,
    weight_decay=1e-2  # 添加weight decay
)
```

### 优先级3: 调整Loss权重
```python
# 在options.py中修改默认值
--diffusion_l1_weight 2.0  # 从1.0增加到2.0
--diffusion_ddim_weight 1.0  # 保持
```

### 优先级4: 添加梯度裁剪
```python
# 在trainer.py的backward之后添加
torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=1.0)
```

### 优先级5: 添加训练监控
```python
# 监控梯度范数
total_norm = 0
for p in self.parameters_to_train:
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
self.log("train", inputs, outputs, {"grad_norm": total_norm})
```

---

## 调参策略建议

### 阶段1: 稳定训练（前30%训练）
- 高L1权重（3.0-5.0），强制跟随教师
- 较低的学习率（5e-5）
- 平滑的学习率调度

### 阶段2: 平衡训练（30%-70%）
- 中等L1权重（1.5-2.0）
- 正常学习率（1e-4）
- 增加photometric权重

### 阶段3: 精炼训练（70%-100%）
- 低L1权重（1.0）
- 较低学习率（5e-5）
- 高几何正则化权重

---

## 实验建议

1. **先修复学习率调度器**，观察280000步是否还会崩溃
2. **增加L1权重到2.0**，观察学生模型是否更好跟随教师
3. **改用AdamW**，观察训练稳定性
4. **添加梯度监控**，在TensorBoard中观察梯度范数
5. **如果问题持续**，考虑从检查点恢复并调整策略

---

## 代码修改清单

- [ ] 修改优化器为AdamW
- [ ] 修改学习率调度器为ChainedScheduler或改进的StepLR
- [ ] 调整默认loss权重
- [ ] 添加梯度裁剪
- [ ] 添加梯度监控
- [ ] 添加学习率监控到TensorBoard
- [ ] 考虑添加mask训练（参考MonoDiffusion）

