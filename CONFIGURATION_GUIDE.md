# 配置指南：损失权重与训练选项

## 1. 扩散步数说明

### 1.1 Diffusion Inference Steps (扩散推理步数)

**含义**：DDIM采样时的去噪步数，控制从噪声到清晰深度图的迭代次数。

**工作原理**：
- 扩散模型从纯噪声开始，逐步去噪生成深度图
- 更多步数 = 更精细的去噪过程 = 更好的细节恢复
- 更少步数 = 更快的推理速度，但可能丢失细节

**配置格式**：
```bash
--diffusion_steps [scale_2, scale_1, scale_0]
```

**示例**：
- `[5, 4, 3]`：粗尺度5步，中尺度4步，细尺度3步（基础配置）
- `[8, 6, 5]`：增强细节配置（当前默认）
- `[10, 8, 6]`：最大化细节（更慢）
- `[3, 2, 2]`：快速推理（可能丢失细节）

**建议**：
- **凸显细节**：使用 `[8, 6, 5]` 或 `[10, 8, 6]`
- **平衡**：使用 `[5, 4, 3]`
- **快速推理**：使用 `[3, 2, 2]`

### 1.2 Diffusion Training Timesteps (扩散训练时间步数)

**含义**：训练时扩散调度器的总时间步数，定义噪声调度。

**作用**：控制训练时添加噪声的粒度，更多步数提供更细粒度的噪声级别。

**配置格式**：
```bash
--diffusion_timesteps [scale_2, scale_1, scale_0]
```

**默认值**：`[250, 200, 150]`

**建议**：通常保持默认值，除非需要特殊调整。

---

## 2. 损失权重配置

### 2.1 当前默认配置（有利于细节）

```bash
# 平滑损失（最不利于细节）
--smoothness_weight 0.05          # 从0.2降低

# 几何正则化
--plane_weight 0.5                 # 从2.0降低
--line_weight 0.1                   # 从0.5降低

# 扩散损失
--diffusion_l1_weight 2.0          # 学生-教师一致性
--diffusion_ddim_weight 1.0         # 扩散去噪损失

# 光度损失
--photometric_weight 1.0           # 基础损失
```

### 2.2 配置选项说明

#### Smoothness Weight (平滑损失权重)
- **作用**：惩罚相邻像素的深度不连续性
- **影响**：⚠️ **最不利于细节保留**
- **建议值**：
  - 凸显细节：`0.05-0.1`
  - 平衡：`0.1-0.2`
  - 平滑优先：`0.2-0.5`

#### Plane Weight (平面正则化权重)
- **作用**：强制4个点共面
- **影响**：⚠️ 不利于细节保留
- **建议值**：
  - 凸显细节：`0.5-1.0` 或禁用
  - 平衡：`1.0-2.0`
  - 几何优先：`2.0-5.0`

#### Line Weight (线段正则化权重)
- **作用**：强制3个点共线
- **影响**：⚠️ 中等程度不利于细节
- **建议值**：
  - 凸显细节：`0.1-0.2` 或禁用
  - 平衡：`0.2-0.5`
  - 几何优先：`0.5-1.0`

#### Diffusion L1 Weight (扩散L1损失权重)
- **作用**：学生模型与教师模型的一致性
- **影响**：✅ 有助于学习教师知识
- **建议值**：`1.0-3.0`

#### Diffusion DDIM Weight (扩散DDIM损失权重)
- **作用**：扩散模型的去噪损失
- **影响**：✅ **有利于细节生成**
- **建议值**：`0.5-2.0`

#### Photometric Weight (光度损失权重)
- **作用**：重投影误差
- **影响**：✅ 有利于细节对齐
- **建议值**：通常保持 `1.0`

---

## 3. Mask训练配置

### 3.1 启用Mask训练

```bash
--use_mask_training
```

### 3.2 Mask训练参数

```bash
# Mask概率（被遮挡的概率）
--mask_probability 0.2    # 0.2 = 20%遮挡，80%保留

# Mask损失权重
--mask_loss_weight 0.1   # Masked预测与完整预测的一致性损失权重
```

### 3.3 Mask训练原理

1. **随机遮挡**：对编码器特征应用随机mask（20%区域被遮挡）
2. **Masked预测**：使用masked特征进行深度预测
3. **一致性约束**：Masked预测应与完整预测一致
4. **优势**：增强泛化能力，处理遮挡情况，有助于细节保留

---

## 4. 推荐配置方案

### 方案1：最大化细节（当前默认）

```bash
python train.py \
  --smoothness_weight 0.05 \
  --plane_weight 0.5 \
  --line_weight 0.1 \
  --diffusion_steps 8 6 5 \
  --diffusion_l1_weight 2.0 \
  --diffusion_ddim_weight 1.0 \
  --use_mask_training \
  --mask_probability 0.2 \
  --mask_loss_weight 0.1
```

### 方案2：平衡配置

```bash
python train.py \
  --smoothness_weight 0.1 \
  --plane_weight 1.0 \
  --line_weight 0.2 \
  --diffusion_steps 5 4 3 \
  --diffusion_l1_weight 2.0 \
  --diffusion_ddim_weight 1.0
```

### 方案3：禁用几何正则化（极端细节）

```bash
python train.py \
  --disable_plane_regularization \
  --disable_line_regularization \
  --smoothness_weight 0.05 \
  --diffusion_steps 10 8 6 \
  --diffusion_l1_weight 2.0 \
  --diffusion_ddim_weight 1.5 \
  --use_mask_training
```

### 方案4：快速训练（可能丢失细节）

```bash
python train.py \
  --smoothness_weight 0.2 \
  --plane_weight 2.0 \
  --line_weight 0.5 \
  --diffusion_steps 3 2 2 \
  --diffusion_l1_weight 1.0 \
  --diffusion_ddim_weight 0.5
```

---

## 5. 消融实验建议

### 5.1 损失权重消融

```bash
# 实验1：平滑损失的影响
--smoothness_weight 0.0   # 禁用
--smoothness_weight 0.05  # 低
--smoothness_weight 0.1   # 中
--smoothness_weight 0.2   # 高

# 实验2：几何正则化的影响
--disable_plane_regularization  # 禁用平面
--disable_line_regularization    # 禁用线段
--plane_weight 0.5 1.0 2.0      # 不同权重
--line_weight 0.1 0.2 0.5      # 不同权重

# 实验3：扩散步数的影响
--diffusion_steps 3 2 2   # 快速
--diffusion_steps 5 4 3   # 基础
--diffusion_steps 8 6 5   # 增强细节
--diffusion_steps 10 8 6  # 最大化细节
```

### 5.2 Mask训练消融

```bash
# 实验4：Mask训练的影响
# 不使用mask训练
python train.py ...

# 使用mask训练
python train.py --use_mask_training

# 不同mask概率
--mask_probability 0.1  # 10%遮挡
--mask_probability 0.2  # 20%遮挡（默认）
--mask_probability 0.3  # 30%遮挡

# 不同mask损失权重
--mask_loss_weight 0.05  # 低权重
--mask_loss_weight 0.1    # 默认
--mask_loss_weight 0.2    # 高权重
```

---

## 6. 配置检查清单

训练前检查以下配置：

- [ ] 扩散推理步数是否合适（`--diffusion_steps`）
- [ ] 平滑损失权重是否调整（`--smoothness_weight`）
- [ ] 几何正则化权重是否调整（`--plane_weight`, `--line_weight`）
- [ ] 是否启用mask训练（`--use_mask_training`）
- [ ] 所有损失权重是否合理
- [ ] 是否禁用了不需要的损失（`--disable_*`）

---

## 7. 监控建议

在TensorBoard中监控：

1. **学习率曲线**：检查学习率调度是否正常
2. **梯度范数**：检查是否有梯度爆炸（>10可能有问题）
3. **各损失项**：观察各损失的相对大小
4. **Mask训练可视化**：如果启用，检查mask和masked预测
5. **学生vs教师**：比较学生和教师的预测差异

---

## 8. 常见问题

### Q: 如何知道当前配置是否有利于细节？

A: 检查TensorBoard中的预测结果，特别是：
- 小物体的边缘是否清晰
- 纹理细节是否保留
- 与教师模型的对比

### Q: 扩散步数增加会显著增加训练时间吗？

A: 主要影响推理时间，训练时间影响较小。建议从`[8, 6, 5]`开始。

### Q: Mask训练会增加多少训练时间？

A: 大约增加20-30%（因为需要额外的前向传播）。

### Q: 如何快速找到最佳配置？

A: 建议进行网格搜索或贝叶斯优化，重点关注：
- `smoothness_weight`: [0.05, 0.1, 0.2]
- `diffusion_steps`: [[5,4,3], [8,6,5], [10,8,6]]
- `use_mask_training`: [True, False]

