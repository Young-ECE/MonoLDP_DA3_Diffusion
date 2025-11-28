# DA3模型本地路径配置说明

## 已配置的本地路径

DA3模型本地路径已设置为：
```
/home/jingyang/.cache/huggingface/hub/models--depth-anything--DA3MONO-LARGE/snapshots/f465978e618db8cc79c83b8bbf24964857db1875
```

## 配置说明

### 1. 训练时（trainer.py）

**参数**：`--depth_anything_v3_weights`

**默认值**：已设置为上述本地路径

**使用方式**：
```bash
# 使用默认路径（自动使用本地模型）
python train.py

# 或者显式指定（如果需要使用其他路径）
python train.py --depth_anything_v3_weights /path/to/other/model

# 如果想从HuggingFace下载（不使用本地路径）
python train.py --depth_anything_v3_weights None
```

### 2. 评估时（evaluate_da3mono_nyu_depth.py）

**参数**：`--da3_model_path`

**默认值**：已设置为上述本地路径

**使用方式**：
```bash
# 使用默认路径（自动使用本地模型）
python evaluate_da3mono_nyu_depth.py

# 或者显式指定（如果需要使用其他路径）
python evaluate_da3mono_nyu_depth.py --da3_model_path /path/to/other/model

# 如果想从HuggingFace下载（不使用本地路径）
python evaluate_da3mono_nyu_depth.py --da3_model_path None
```

## 验证配置

运行以下命令验证路径是否正确：

```bash
# 检查路径是否存在
ls -la /home/jingyang/.cache/huggingface/hub/models--depth-anything--DA3MONO-LARGE/snapshots/f465978e618db8cc79c83b8bbf24964857db1875/

# 应该能看到 model.safetensors 和 config.json
```

## 注意事项

1. **路径格式**：这是一个HuggingFace snapshot目录，包含 `model.safetensors` 和 `config.json`
2. **符号链接**：如果看到符号链接（`->`），这是正常的，HuggingFace使用符号链接管理模型文件
3. **自动检测**：代码会自动检测这个路径并加载模型，无需额外配置

## 修改配置

如果需要修改路径，编辑 `options.py`：

1. **训练配置**（第366-369行）：
   ```python
   self.parser.add_argument("--depth_anything_v3_weights",
                           default="/your/new/path")
   ```

2. **评估配置**（第539-543行）：
   ```python
   self.parser.add_argument("--da3_model_path",
                           default="/your/new/path")
   ```

