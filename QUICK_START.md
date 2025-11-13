# 快速开始指南 - MonoDiffusion 集成

## 🚀 5分钟快速上手

### 第一步：验证文件已复制
```bash
cd /home/jingyang/MonoLDP/DepthEstimation

# 检查关键文件
ls diffusers/schedulers/scheduling_ddim.py  # 应该存在
ls networks/depth_decoder_diffusion.py      # 应该存在

# 测试导入
python3 << EOF
from networks import DepthDecoderDiffusion
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
print("✅ 所有模块导入成功！")
EOF
```

### 第二步：修改 trainer.py 的三个位置

打开 `DepthEstimation/trainer.py`，按照以下修改：

#### 📍 位置1：导入模块（文件顶部）
在文件开头的import部分添加：
```python
import torch.nn.functional as F
import os
```

#### 📍 位置2：初始化模型（约第74行）

**查找**:
```python
self.models["depth"] = networks.DepthDecoder(
```

**在这之前添加**:
```python
# 扩散模块初始化
if self.opt.use_diffusion:
    print("🔄 使用扩散深度解码器")
    
    # 教师模型
    self.models["pre_depth_encoder"] = networks.ResnetEncoder(
        self.opt.num_layers, self.opt.weights_init == "pretrained")
    self.models["pre_depth_decoder"] = networks.DepthDecoder(
        self.models["pre_depth_encoder"].num_ch_enc, self.opt.scales,
        PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation)
    
    # 加载教师权重
    if self.opt.teacher_weights_folder is not None:
        enc_path = os.path.join(self.opt.teacher_weights_folder, "encoder.pth")
        dep_path = os.path.join(self.opt.teacher_weights_folder, "depth.pth")
        if os.path.exists(enc_path) and os.path.exists(dep_path):
            self.models["pre_depth_encoder"].load_state_dict(torch.load(enc_path))
            self.models["pre_depth_decoder"].load_state_dict(torch.load(dep_path))
            print("✅ 教师模型加载成功")
    
    # 冻结教师
    for p in self.models["pre_depth_encoder"].parameters(): p.requires_grad = False
    for p in self.models["pre_depth_decoder"].parameters(): p.requires_grad = False
    self.models["pre_depth_encoder"].to(self.device).eval()
    self.models["pre_depth_decoder"].to(self.device).eval()
    
    # 学生模型
    self.models["depth"] = networks.DepthDecoderDiffusion(
        self.models["encoder"].num_ch_enc, self.opt.scales,
        num_output_channels=3,
        PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation)
else:
```

**然后保持原来的代码**:
```python
    self.models["depth"] = networks.DepthDecoder(...)
```

#### 📍 位置3：修改 process_batch 方法

在 `process_batch` 方法中，找到深度预测的部分，添加扩散逻辑：

**在调用 self.models["depth"] 之前添加**:
```python
# 生成归一化像素坐标
norm_pix_coords = {}
for scale in self.opt.scales:
    h = self.opt.height // (2 ** scale)
    w = self.opt.width // (2 ** scale)
    y = torch.linspace(0, 1, h).view(h, 1).repeat(1, w)
    x = torch.linspace(0, 1, w).view(1, w).repeat(h, 1)
    coords = torch.stack([x, y], 0).unsqueeze(0)
    coords = coords.repeat(inputs[("color", 0, 0)].shape[0], 1, 1, 1)
    norm_pix_coords[scale] = coords.to(self.device)

# 扩散处理
if self.opt.use_diffusion:
    with torch.no_grad():
        pre_feat = self.models["pre_depth_encoder"](inputs["color_aug", 0, 0])
        pre_out = self.models["pre_depth_decoder"](pre_feat, norm_pix_coords)
    
    features = self.models["encoder"](inputs["color_aug", 0, 0])
    gt_diff = {("disp_diffusion", s): pre_out[("disp", s)].detach() 
               for s in self.opt.scales}
    outputs = self.models["depth"](features, norm_pix_coords, gt_diff)
    
    for s in self.opt.scales:
        outputs[("predisp", s)] = pre_out[("disp", s)]
else:
    features = self.models["encoder"](inputs["color_aug", 0, 0])
    outputs = self.models["depth"](features, norm_pix_coords)
```

**在 compute_losses 之后添加**:
```python
losses = self.compute_losses(inputs, outputs)

# 扩散损失
if self.opt.use_diffusion:
    # L1损失
    l1 = sum(F.l1_loss(outputs[("predisp", s)], outputs[("disp", s)]) 
             for s in self.opt.scales) / len(self.opt.scales)
    
    # DDIM损失
    ddim = sum(outputs.get(("ddim_loss", s), 0) 
               for s in self.opt.scales) / len(self.opt.scales)
    
    losses['l1'] = l1
    losses['ddim'] = ddim if isinstance(ddim, torch.Tensor) else torch.tensor(ddim).to(self.device)
    losses['photometric'] = losses["loss"]
    losses["loss"] = losses['photometric'] + self.opt.diffusion_l1_weight * l1 + \
                     self.opt.diffusion_ddim_weight * losses['ddim']
```

### 第三步：训练

#### 阶段1：训练基础模型
```bash
python train.py \
    --model_name base_model \
    --data_path /path/to/your/nyu_data \
    --num_epochs 15 \
    --batch_size 12
```

#### 阶段2：使用扩散训练
```bash
python train.py \
    --model_name diffusion_model \
    --use_diffusion \
    --teacher_weights_folder ./logs/base_model/models/weights_14 \
    --data_path /path/to/your/nyu_data \
    --num_epochs 10 \
    --batch_size 6 \
    --learning_rate 5e-5
```

## ✅ 检查清单

- [ ] 已复制 `diffusers/` 文件夹
- [ ] 已创建 `depth_decoder_diffusion.py`
- [ ] 已更新 `networks/__init__.py`
- [ ] 已添加参数到 `options.py`
- [ ] 已修改 `trainer.py` 的3个位置
- [ ] 可以成功导入 `DepthDecoderDiffusion`
- [ ] 已训练基础模型或有预训练权重

## 🔧 常见错误排查

### 错误1: ImportError: No module named 'timm'
```bash
pip install timm
```

### 错误2: No module named 'diffusers'
检查路径：`DepthEstimation/diffusers/` 必须存在

### 错误3: CUDA out of memory
减小 batch size:
```bash
--batch_size 4  # 或更小
```

### 错误4: 找不到教师权重
确保路径正确：
```bash
ls ./logs/base_model/models/weights_14/encoder.pth  # 必须存在
ls ./logs/base_model/models/weights_14/depth.pth    # 必须存在
```

## 📊 监控训练

```bash
tensorboard --logdir=./logs
```

浏览器打开：http://localhost:6006

查看：
- **Scalars**: loss, l1, ddim, photometric
- **Images**: disp_0 (深度), predisp_0 (教师), residual_0 (残差)

## 📝 完整的文档

- `集成总结.md`: 详细的中文指南
- `DIFFUSION_INTEGRATION_GUIDE.md`: 技术细节
- `TRAINER_MODIFICATIONS.md`: 代码修改说明

## 🎯 核心要点

1. **教师模型**: 冻结的预训练模型，生成伪GT
2. **学生模型**: 带扩散的模型，学习精炼深度
3. **三个损失**: 光度 + L1 + DDIM
4. **级联精炼**: Scale 2 → 1 → 0，从粗到细

## 💡 调试技巧

**测试扩散模块是否工作**:
在 `process_batch` 中添加打印：
```python
if self.opt.use_diffusion:
    print(f"Teacher disp range: [{pre_out[('disp', 0)].min():.3f}, {pre_out[('disp', 0)].max():.3f}]")
    print(f"Student disp range: [{outputs[('disp', 0)].min():.3f}, {outputs[('disp', 0)].max():.3f}]")
    if ("ddim_loss", 0) in outputs:
        print(f"DDIM loss: {outputs[('ddim_loss', 0)]:.4f}")
```

应该看到合理的范围值和DDIM loss。

---

完成以上步骤后，你就成功集成了扩散模块！🎉

