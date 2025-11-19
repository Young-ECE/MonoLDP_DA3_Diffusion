# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Unsupervised Indoor Depth-Pose Learning options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="/oldisk/home/jingyang/monoldp/datasets/nyu_data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="/oldisk/home/jingyang/monoldp/temp")

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="monoldp_diffusion_model")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="nyu")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="nyu")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=256)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=320)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=10.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -2, 2])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=8)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=15)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=30)  # 从20增加到30，避免学习率下降过快
        self.parser.add_argument("--scheduler_gamma",
                                 type=float,
                                 help="gamma (decay factor) of the scheduler",
                                 default=0.5)  # 从0.1改为0.5，更平滑的衰减
        self.parser.add_argument("--use_cosine_scheduler",
                                 help="use cosine annealing scheduler instead of step scheduler",
                                 action="store_true")
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay for AdamW optimizer",
                                 default=1e-2)
        self.parser.add_argument("--max_grad_norm",
                                 type=float,
                                 help="maximum gradient norm for clipping",
                                 default=1.0)

        # conventional options
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        # ====================================================================
        # LOSS WEIGHTS (可配置的损失权重，便于调参和消融实验)
        # ====================================================================
        
        # Smoothness Loss (平滑损失)
        # 作用：惩罚相邻像素的深度不连续性，使用边缘感知权重
        # 影响：⚠️ 最不利于细节保留，会平滑掉小尺度细节
        # 建议：凸显细节时使用 0.05-0.1，平衡时使用 0.1-0.2
        self.parser.add_argument("--smoothness_weight",
                                 type=float,
                                 help="smoothness loss weight (⚠️ high value smooths details)",
                                 default=0.05)  # 从0.2降低到0.05，更有利于细节
        
        # Plane Regularization (平面正则化)
        # 作用：强制4个点共面，假设场景中存在平面结构
        # 影响：⚠️ 不利于细节保留，可能错误地平滑小物体和纹理
        # 建议：凸显细节时使用 0.5-1.0，或禁用
        self.parser.add_argument("--num_plane_keysets",
                                 type=int,
                                 help="the number of keysets for plane regularization",
                                 default=512)
        self.parser.add_argument("--plane_weight",
                                 type=float,
                                 help="plane regularization weight (⚠️ high value may smooth details)",
                                 default=0.5)  # 从2.0降低到0.5，更有利于细节
        
        # Line Regularization (线段正则化)
        # 作用：强制3个点共线，假设场景中存在直线结构
        # 影响：⚠️ 中等程度不利于细节保留
        # 建议：凸显细节时使用 0.1-0.2，或禁用
        self.parser.add_argument("--num_line_keysets",
                                 type=int,
                                 help="the number of keysets for line regularization",
                                 default=128)
        self.parser.add_argument("--line_weight",
                                 type=float,
                                 help="line regularization weight (⚠️ may smooth details)",
                                 default=0.1)  # 从0.5降低到0.1，更有利于细节
        
        # Photometric Loss (光度损失)
        # 作用：重投影误差，确保深度预测与图像一致
        # 影响：✅ 有利于细节对齐
        # 注意：权重固定为1.0，作为基础损失
        self.parser.add_argument("--photometric_weight",
                                 type=float,
                                 help="photometric reprojection loss weight (base loss, usually 1.0)",
                                 default=1.0)
        
        # SSIM Loss (结构相似性损失)
        # 作用：在reprojection loss中与L1混合使用
        # 影响：⚠️ 在低对比度区域可能不够敏感，可能丢失细节
        # 建议：可以禁用(--no_ssim)仅使用L1，或降低SSIM权重
        # 注意：当前实现中SSIM权重固定为0.85，L1为0.15

        # ====================================================================
        # DIFFUSION OPTIONS (扩散模型配置)
        # ====================================================================
        # Note: Diffusion decoder is always used, this flag is kept for backward compatibility
        self.parser.add_argument("--use_diffusion",
                                help="[DEPRECATED] Diffusion is always enabled",
                                action="store_true",
                                default=True)
        
        # Diffusion Inference Steps (扩散推理步数)
        # 含义：DDIM采样时的去噪步数，从粗到细的尺度对应不同的步数
        # 作用：控制扩散过程的精细程度
        #   - 更多步数 = 更精细的去噪 = 更好的细节恢复，但计算更慢
        #   - 更少步数 = 更快的推理，但可能丢失细节
        # 格式：[scale_2, scale_1, scale_0] 或 [scale_0] (如果只有一个尺度)
        # 建议：
        #   - 基础：[5, 4, 3] (粗尺度5步，中尺度4步，细尺度3步)
        #   - 增强细节：[8, 6, 5] 或 [10, 8, 6]
        #   - 快速推理：[3, 2, 2]
        self.parser.add_argument("--diffusion_steps",
                                nargs="+",
                                type=int,
                                default=[8, 6, 5],  # 从[5,4,3]增加到[8,6,5]，增强细节
                                help="number of diffusion inference steps for each scale [coarse to fine]. "
                                     "More steps = better details but slower. "
                                     "Format: [scale_2, scale_1, scale_0] or [scale_0] for single scale")
        
        # Diffusion Training Timesteps (扩散训练时间步数)
        # 含义：训练时扩散调度器的总时间步数
        # 作用：定义噪声调度，更多步数提供更细粒度的噪声级别
        # 格式：[scale_2, scale_1, scale_0]
        # 建议：通常保持默认值，除非需要特殊调整
        self.parser.add_argument("--diffusion_timesteps",
                                nargs="+",
                                type=int,
                                default=[250, 200, 150],
                                help="number of training timesteps for each scale [coarse to fine]. "
                                     "Defines noise schedule. Format: [scale_2, scale_1, scale_0]")
        
        # Diffusion L1 Loss Weight (扩散L1损失权重)
        # 作用：学生模型与教师模型预测的一致性损失
        # 影响：✅ 有助于学生模型学习教师的知识
        # 注意：如果教师模型本身不够细节，高权重可能让学生也丢失细节
        # 建议：通常1.0-3.0，根据教师模型质量调整
        self.parser.add_argument("--diffusion_l1_weight",
                                type=float,
                                default=2.0,  # 从1.0增加到2.0，强制学生模型更好跟随教师
                                help="weight for L1 loss between teacher and student predictions. "
                                     "Higher = stronger teacher-student alignment")
        
        # Diffusion DDIM Loss Weight (扩散DDIM损失权重)
        # 作用：扩散模型的去噪损失，训练噪声预测网络
        # 影响：✅ 有利于细节生成（扩散模型本身设计用于生成细节）
        # 建议：通常0.5-2.0
        self.parser.add_argument("--diffusion_ddim_weight",
                                type=float,
                                default=1.0,
                                help="weight for DDIM diffusion loss (denoising loss). "
                                     "✅ Helps preserve details")
        
        # ====================================================================
        # MASK TRAINING OPTIONS (Mask训练配置)
        # ====================================================================
        # Mask训练：使用随机mask对特征进行遮挡，增强模型鲁棒性
        # 原理：类似dropout，但作用于特征空间，强制模型从部分信息恢复深度
        # 优势：增强泛化能力，处理遮挡情况，有助于细节保留
        
        self.parser.add_argument("--use_mask_training",
                                help="enable mask training (feature-level masking for robustness)",
                                action="store_true",
                                default=False)
        
        self.parser.add_argument("--mask_probability",
                                type=float,
                                default=0.2,
                                help="probability of masking a feature location (0.2 = 20%% masked, 80%% kept)",
                                choices=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        
        self.parser.add_argument("--mask_loss_weight",
                                type=float,
                                default=0.1,
                                help="weight for mask training loss (consistency between masked and full predictions)")
        
        # DEPTH ANYTHING V3 teacher options (always enabled)
        # Note: Depth Anything V3 is always used as teacher model
        self.parser.add_argument("--use_depth_anything_v3",
                                help="[DEPRECATED] Depth Anything V3 is always used as teacher",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--depth_anything_v3_model",
                                type=str,
                                help="Depth Anything V3 model name",
                                default="DA3Mono-Large",
                                choices=["DA3Mono-Large"])
        self.parser.add_argument("--depth_anything_v3_weights",
                                type=str,
                                help="path to Depth Anything V3 model directory (optional, will auto-download from HuggingFace if not provided)",
                                default=None)

        # ABLATION options-PLNet
        self.parser.add_argument("--disable_pixel_coordinate_modulation",
                                 help="if set, do not use pixel coordinate modulation,"
                                      "and apply a ReLU to the obtained disparity",
                                 action="store_true")
        self.parser.add_argument("--disable_plane_smoothness",
                                 help="if set, the image-edge-aware smoothness will be applied on the "
                                      "conventional disparity instead of the proposed planar coefficients",
                                 action="store_true")
        self.parser.add_argument("--disable_plane_regularization",
                                 help="if set, do not use plane regularization",
                                 action="store_true")
        self.parser.add_argument("--disable_line_regularization",
                                 help="if set, do not use line regularization",
                                 action="store_true")

        # SYSTEM options
        # ====================================================================
        # GPU/DEVICE OPTIONS (GPU设备配置)
        # ====================================================================
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA (use CPU)",
                                 action="store_true")
        
        self.parser.add_argument("--gpu_id",
                                 type=int,
                                 default=0,
                                 help="GPU device ID to use (0, 1, 2, ...). "
                                      "Default: 0. Use --gpu_id 1 to use GPU 1, etc.")
        
        self.parser.add_argument("--cuda_device",
                                 type=str,
                                 default=None,
                                 help="Alternative way to specify GPU: 'cuda:0', 'cuda:1', etc. "
                                      "If specified, overrides --gpu_id")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose", "scalenet", "regression"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)
        self.parser.add_argument("--debug_no_save",
                                 help="if set, skip writing tensorboard logs and model checkpoints (debug mode)",
                                 action="store_true",
                                 default=False)
        self.parser.add_argument("--debug_save_teacher",
                                 help="if set, save teacher disparity/depth debug visualizations",
                                 action="store_true",
                                 default=True)

        # EVALUATION options
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="nyu",
                                 choices=["nyu"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
