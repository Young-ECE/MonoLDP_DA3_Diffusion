# Trainer.py 修改指南

本文档详细说明如何修改 `DepthEstimation/trainer.py` 以支持扩散模块。

## 修改1：在 __init__ 方法中添加预训练教师模型

**位置**：在深度解码器初始化之前（约第74行）

**原代码**：
```python
self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales,
                                             PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation)
```

**修改为**：
```python
# 选项1：使用原始深度解码器（不使用扩散）
if not self.opt.use_diffusion:
    self.models["depth"] = networks.DepthDecoder(
        self.models["encoder"].num_ch_enc, self.opt.scales,
        PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation
    )
else:
    # 选项2：使用带扩散的深度解码器
    # 首先创建教师模型（用于生成伪GT）
    self.models["pre_depth_encoder"] = networks.ResnetEncoder(
        self.opt.num_layers, self.opt.weights_init == "pretrained"
    )
    self.models["pre_depth_decoder"] = networks.DepthDecoder(
        self.models["pre_depth_encoder"].num_ch_enc, 
        self.opt.scales,
        PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation
    )
    
    # 加载预训练教师模型权重（如果提供）
    if self.opt.teacher_weights_folder is not None:
        teacher_path = self.opt.teacher_weights_folder
        encoder_path = os.path.join(teacher_path, "encoder.pth")
        decoder_path = os.path.join(teacher_path, "depth.pth")
        
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            print(f"Loading teacher model from {teacher_path}")
            encoder_dict = torch.load(encoder_path)
            decoder_dict = torch.load(decoder_path)
            
            model_dict = self.models["pre_depth_encoder"].state_dict()
            depth_model_dict = self.models["pre_depth_decoder"].state_dict()
            
            self.models["pre_depth_encoder"].load_state_dict(
                {k: v for k, v in encoder_dict.items() if k in model_dict}
            )
            self.models["pre_depth_decoder"].load_state_dict(
                {k: v for k, v in decoder_dict.items() if k in depth_model_dict}
            )
            print("Teacher model loaded successfully")
        else:
            print(f"Warning: Teacher weights not found at {teacher_path}")
            print("Training teacher model from scratch...")
    
    # 冻结教师模型
    for param in self.models["pre_depth_encoder"].parameters():
        param.requires_grad = False
    for param in self.models["pre_depth_decoder"].parameters():
        param.requires_grad = False
    
    self.models["pre_depth_encoder"].to(self.device)
    self.models["pre_depth_decoder"].to(self.device)
    self.models["pre_depth_encoder"].eval()
    self.models["pre_depth_decoder"].eval()
    
    # 创建学生模型（带扩散）
    self.models["depth"] = networks.DepthDecoderDiffusion(
        self.models["encoder"].num_ch_enc, 
        self.opt.scales,
        PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation
    )
```

## 修改2：在 process_batch 方法中添加扩散训练逻辑

**位置**：process_batch 方法中的前向传播部分

### 2.1 生成伪GT（使用教师模型）

在深度预测之前添加：

```python
def process_batch(self, inputs):
    """Pass a minibatch through the network and generate images and losses"""
    for key, ipt in inputs.items():
        inputs[key] = ipt.to(self.device)
    
    # 如果使用扩散，先用教师模型生成伪GT
    if self.opt.use_diffusion:
        with torch.no_grad():
            pre_features = self.models["pre_depth_encoder"](inputs[("color_aug", 0, 0)])
            pre_norm_pix_coords = self.generate_norm_pix_coords(inputs)
            pre_outputs = self.models["pre_depth_decoder"](pre_features, pre_norm_pix_coords)
    
    # 学生模型前向传播
    # ... [现有的编码器代码]
    features = self.models["encoder"](inputs[("color_aug", 0, 0)])
    
    # 生成归一化像素坐标
    norm_pix_coords = self.generate_norm_pix_coords(inputs)
    
    # 准备扩散的GT
    if self.opt.use_diffusion:
        gt_for_diffusion = {}
        for scale in self.opt.scales:
            # 使用教师模型的预测作为扩散的伪GT
            gt_for_diffusion[("disp_diffusion", scale)] = pre_outputs[("disp", scale)].detach()
        
        # 使用扩散解码器
        outputs = self.models["depth"](features, norm_pix_coords, gt_for_diffusion)
        
        # 保存教师预测用于可视化和损失计算
        for scale in self.opt.scales:
            outputs[("predisp", scale)] = pre_outputs[("disp", scale)]
    else:
        # 不使用扩散，正常预测
        outputs = self.models["depth"](features, norm_pix_coords)
    
    # ... [其余代码保持不变]
```

### 2.2 修改损失计算

在 `process_batch` 方法的最后，损失计算部分：

```python
    # 生成重投影图像
    self.generate_images_pred(inputs, outputs)
    
    # 计算光度损失
    losses = self.compute_losses(inputs, outputs)
    
    # 如果使用扩散，添加额外的损失
    if self.opt.use_diffusion:
        # L1损失（学生与教师的一致性）
        l1_loss = 0
        for scale in self.opt.scales:
            l1_loss += F.l1_loss(outputs[("predisp", scale)], outputs[("disp", scale)])
        losses['l1'] = l1_loss / self.num_scales
        
        # DDIM损失
        if ("ddim_loss", 0) in outputs:
            losses['ddim'] = (outputs["ddim_loss", 0] + 
                             outputs["ddim_loss", 1] + 
                             outputs["ddim_loss", 2]) / self.num_scales
        else:
            losses['ddim'] = torch.tensor(0.0).to(self.device)
        
        # 保存原始光度损失
        losses['photometric'] = losses["loss"]
        
        # 总损失
        losses["loss"] = (1.0 * losses['photometric'] + 
                         1.0 * losses['l1'] + 
                         1.0 * losses['ddim'])
    
    return outputs, losses
```

## 修改3：添加 generate_norm_pix_coords 辅助方法

在 Trainer 类中添加这个方法（如果还没有）：

```python
def generate_norm_pix_coords(self, inputs):
    """Generate normalized pixel coordinates for each scale"""
    norm_pix_coords = {}
    
    for scale in self.opt.scales:
        h = self.opt.height // (2 ** scale)
        w = self.opt.width // (2 ** scale)
        
        # 生成网格坐标
        y_coords = torch.linspace(0, 1, h).view(h, 1).repeat(1, w)
        x_coords = torch.linspace(0, 1, w).view(1, w).repeat(h, 1)
        
        # 堆叠为 [2, H, W]
        coords = torch.stack([x_coords, y_coords], 0)
        
        # 扩展batch维度 [B, 2, H, W]
        coords = coords.unsqueeze(0).repeat(inputs[("color", 0, 0)].shape[0], 1, 1, 1)
        coords = coords.to(self.device)
        
        norm_pix_coords[scale] = coords
    
    return norm_pix_coords
```

## 修改4：更新日志记录

在 `log` 方法中添加扩散相关的可视化：

```python
def log(self, mode, inputs, outputs, losses):
    """Write an event to the tensorboard events file"""
    writer = self.writers[mode]
    
    # 记录标量损失
    for l, v in losses.items():
        writer.add_scalar("{}".format(l), v, self.step)
    
    # 可视化图像
    for j in range(min(4, self.opt.batch_size)):
        for s in self.opt.scales:
            # ... [原有的可视化代码]
            
            # 添加扩散相关的可视化
            if self.opt.use_diffusion:
                # 教师模型的预测
                if ("predisp", s) in outputs:
                    writer.add_image(
                        "predisp_{}/{}".format(s, j),
                        normalize_image(outputs[("predisp", s)][j].data), 
                        self.step
                    )
                
                # 残差（教师 - 学生）
                if ("predisp", s) in outputs and ("disp", s) in outputs:
                    residual = outputs[("predisp", s)][j] - outputs[("disp", s)][j]
                    writer.add_image(
                        "residual_{}/{}".format(s, j),
                        normalize_image(residual.data, center_zero=True), 
                        self.step
                    )
                
                # 扩散过程的中间结果
                if s == 0 and "re-diffusion" in outputs:
                    for i in range(min(3, len(outputs["re-diffusion"]))):
                        writer.add_image(
                            "re-diffusion_{}/{}".format(i, j),
                            normalize_image(outputs["re-diffusion"][i][j].data), 
                            self.step
                        )
                    
                    # 不确定性估计
                    if len(outputs["re-diffusion"]) >= 3:
                        depth_stack = torch.cat([
                            outputs["re-diffusion"][-1], 
                            outputs["re-diffusion"][-2], 
                            outputs["re-diffusion"][-3]
                        ], 1)
                        depth_uncertainty = torch.std(depth_stack, dim=1, keepdim=True)
                        writer.add_image(
                            "uncertainty_{}/{}".format(s, j),
                            normalize_image(depth_uncertainty[j].data), 
                            self.step
                        )

def normalize_image(img, center_zero=False):
    """Normalize image for visualization"""
    if center_zero:
        # For residuals, center around zero
        max_val = torch.abs(img).max()
        return (img + max_val) / (2 * max_val + 1e-7)
    else:
        # Standard min-max normalization
        min_val = img.min()
        max_val = img.max()
        return (img - min_val) / (max_val - min_val + 1e-7)
```

## 修改5：在 options.py 中添加命令行参数

在 `DepthEstimation/options.py` 中添加：

```python
# 在 MonodepthOptions 类的 __init__ 方法中添加
self.parser.add_argument("--use_diffusion",
                        action="store_true",
                        help="if set, use diffusion-based depth decoder")

self.parser.add_argument("--teacher_weights_folder",
                        type=str,
                        help="path to pretrained teacher model weights",
                        default=None)

self.parser.add_argument("--diffusion_steps",
                        nargs="+",
                        type=int,
                        default=[5, 4, 3],
                        help="number of diffusion inference steps for each scale")

self.parser.add_argument("--diffusion_timesteps",
                        nargs="+",
                        type=int,
                        default=[250, 200, 150],
                        help="number of training timesteps for each scale")
```

## 完整的训练命令示例

### 阶段1：训练基础模型（不使用扩散）
```bash
python train.py \
    --model_name monoldp_base \
    --num_epochs 20 \
    --batch_size 12 \
    --learning_rate 1e-4
```

### 阶段2：使用扩散微调
```bash
python train.py \
    --model_name monoldp_diffusion \
    --use_diffusion \
    --teacher_weights_folder ./tmp/monoldp_base/models/weights_19 \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 5e-5
```

## 关键点总结

1. **教师模型**：用于生成伪GT，应该是预训练好的模型
2. **学生模型**：带扩散的新模型，学习精炼教师的预测
3. **三个损失**：
   - 光度损失（Photometric loss）：重投影一致性
   - L1损失：学生与教师的一致性
   - DDIM损失：扩散模型的去噪损失
4. **多尺度扩散**：从粗到细级联精炼（scale 2 -> 1 -> 0）
5. **冻结教师**：教师模型始终是frozen的，不参与训练

## 常见问题

### Q1: 显存不足
**解决方案**：
- 减小batch size
- 减少扩散推理步数
- 使用gradient checkpointing

### Q2: 训练不稳定
**解决方案**：
- 降低DDIM损失权重（从1.0降到0.5）
- 使用更小的学习率
- 增加warmup步数

### Q3: 没有预训练教师模型
**解决方案**：
- 先训练一个基础模型作为教师
- 或者使用自身作为教师（self-distillation）
- 初期教师和学生使用相同的初始化

## 验证安装

运行以下命令检查是否正确安装：

```python
python -c "from networks import DepthDecoderDiffusion; print('Success!')"
python -c "from diffusers.schedulers.scheduling_ddim import DDIMScheduler; print('Success!')"
```

