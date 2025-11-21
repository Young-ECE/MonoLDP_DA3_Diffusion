# Networks文件夹重构计划

## 📋 当前状态分析

### 模块使用情况
- ✅ **ResnetEncoder**: 训练中使用（encoder和pose_encoder）
- ✅ **DepthDecoderDiffusion**: 训练中使用（主要深度解码器）
- ✅ **DepthDecoder**: 仅在评估/推理中使用，不在训练中使用
- ✅ **PoseDecoder**: 训练中使用
- ✅ **PoseDecoderRec**: 训练中使用
- ✅ **PoseDecoderThird**: 训练中使用
- ✅ **ScaleNetwork**: 训练中使用
- ✅ **ProbabilisticScaleRegressionHead**: 训练中使用
- ✅ **DepthAnythingV3Wrapper**: 训练中使用（教师模型）
- ✅ **lib.py (NONLocalBlock2D)**: 被ScaleNetwork使用

**结论**: 所有模块都被使用，但可以优化结构。

## 🔍 发现的问题

1. **代码重复**: PoseDecoder, PoseDecoderRec, PoseDecoderThird 几乎完全相同
2. **命名不一致**: 有些用下划线，有些用驼峰
3. **组织混乱**: 所有文件都在根目录，没有分类
4. **缺少文档**: 很多模块缺少清晰的文档说明

## 🎯 重构目标

1. **统一代码风格**: 统一命名规范和代码格式
2. **减少重复**: 合并相似的pose decoder
3. **清晰组织**: 按功能分类组织文件
4. **保持兼容**: 确保重构后功能完全一致
5. **改进文档**: 添加清晰的文档字符串

## 📁 重构后的文件结构

```
networks/
├── __init__.py                    # 统一导出接口
├── common.py                      # 共享工具和基类
├── encoders/
│   └── resnet_encoder.py          # ResNet编码器
├── decoders/
│   ├── depth_decoder.py           # 标准深度解码器（评估用）
│   └── depth_decoder_diffusion.py # 扩散深度解码器（训练用）
├── pose/
│   └── pose_decoder.py            # 统一的姿态解码器（支持多种变体）
├── scale/
│   ├── scale_network.py            # 尺度网络
│   └── scale_head.py               # 尺度回归头
├── teacher/
│   └── depth_anything_v3_wrapper.py # Depth Anything V3包装器
└── utils/
    └── non_local_block.py          # 非局部块（从lib.py重命名）
```

## 🔧 重构步骤

1. 创建common.py - 共享工具函数和基类
2. 创建子目录结构
3. 重构pose decoder - 合并为单一模块
4. 重构scale模块 - 分离网络和头
5. 重命名lib.py为non_local_block.py
6. 更新所有导入
7. 验证功能一致性

## ⚠️ 注意事项

- 保持所有网络的前向传播逻辑完全一致
- 保持所有参数名称和默认值不变
- 保持所有输出格式不变
- 确保向后兼容性

