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

        # ====================================================================
        # PATHS (路径配置)
        # ====================================================================
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default="/oldisk/home/jingyang/monoldp/datasets/nyu_data")
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="/oldisk/home/jingyang/monoldp/temp")

        # ====================================================================
        # DATASET & DATA CONFIGURATION (数据集和数据配置)
        # ====================================================================
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="nyu")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
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
                                 default=[0,1,2])
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

        # ====================================================================
        # MODEL ARCHITECTURE (模型架构配置)
        # ====================================================================
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="monoldp_diffusion_model")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
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
        # MODEL ABLATION OPTIONS (模型消融选项)
        # ====================================================================
        self.parser.add_argument("--disable_pixel_coordinate_modulation",
                                 help="if set, do not use pixel coordinate modulation,"
                                      "and apply a ReLU to the obtained disparity",
                                 action="store_true")

        # ====================================================================
        # TRAINING CONFIGURATION (训练配置)
        # ====================================================================
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)

        # ====================================================================
        # OPTIMIZATION & SCHEDULER (优化器和调度器配置)
        # ====================================================================
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay for AdamW optimizer",
                                 default=1e-2)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=30)  # 从20增加到30，避免学习率下降过快
        self.parser.add_argument("--scheduler_gamma",
                                 type=float,
                                 help="gamma (decay factor) of the scheduler",
                                 default=0.5)  # 从0.1改为0.5，更平滑的衰减
        self.parser.add_argument("--use_cosine_scheduler",
                                 help="use cosine annealing scheduler instead of step scheduler (default: True)",
                                 action="store_true",
                                 default=True)
        self.parser.add_argument("--use_step_scheduler",
                                 help="use step scheduler instead of cosine scheduler (overrides --use_cosine_scheduler)",
                                 action="store_true",
                                 default=False)
        self.parser.add_argument("--max_grad_norm",
                                 type=float,
                                 help="maximum gradient norm for clipping",
                                 default=1.0)

        # ====================================================================
        # LOSS FUNCTION CONFIGURATION (损失函数配置)
        # ====================================================================
        # 所有损失函数默认都启用，可以通过--use_xxx或--disable_xxx控制
        # 权重可以通过--xxx_weight调整
        
        # ====================================================================
        # PHOTOMETRIC LOSS (光度损失/重投影损失)
        # ====================================================================
        # 作用：重投影误差，确保深度预测与图像一致
        # 影响：✅ 有利于细节对齐，但可能会干扰学生模型跟随教师
        # 建议：如果想让学生模型更专注于跟随教师，可以降低权重（0.1-0.5）
        # 注意：Photometric Loss包含三个子损失：
        #   - reprojection_losses_ori: 原始重投影损失（权重可配置）
        #   - reprojection_losses_vitual: 虚拟重投影损失（权重可配置）
        #   - reprojection_losses_new: 新重投影损失（权重可配置）
        
        self.parser.add_argument("--use_photometric_loss",
                                help="enable photometric reprojection loss (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--photometric_weight",
                                type=float,
                                default=0.5,
                                help="overall weight for photometric reprojection loss. "
                                     "Lower (0.1-0.5) to focus more on teacher alignment, "
                                     "Higher (0.5-1.5) to balance geometry consistency")
        
        # Reprojection loss component weights (重投影损失组件权重)
        # 作用：控制三个重投影损失之间的相对权重
        # 默认：ori=0.25, virtual=1.0, new=1.0
        
        self.parser.add_argument("--reprojection_ori_weight",
                                type=float,
                                default=0.5,
                                help="weight for original reprojection loss (reprojection_losses_ori). "
                                     "Default: 0.25")
        self.parser.add_argument("--reprojection_virtual_weight",
                                type=float,
                                default=1.0,
                                help="weight for virtual reprojection loss (reprojection_losses_vitual). "
                                     "Default: 1.0")
        self.parser.add_argument("--reprojection_new_weight",
                                type=float,
                                default=1.0,
                                help="weight for new reprojection loss (reprojection_losses_new). "
                                     "Default: 1.0")
        
        # Reprojection SSIM and L1 Loss (重投影SSIM和L1损失)
        # 作用：在reprojection loss中混合使用SSIM和L1损失
        # 注意：这是重投影损失内部的混合权重，与Teacher-Student L1损失不同
        #   - Reprojection L1: 重投影图像与原图像的L1距离（几何一致性）
        #   - Teacher-Student L1: 学生预测与教师预测的L1距离（知识蒸馏）
        # 影响：⚠️ SSIM在低对比度区域可能不够敏感，可能丢失细节
        # 建议：可以独立控制SSIM和L1的启用/禁用，调整权重比例
        #    - 仅使用L1: --use_reprojection_l1 --no-use_reprojection_ssim
        #    - 仅使用SSIM: --use_reprojection_ssim --no-use_reprojection_l1
        #    - 两者都使用: --use_reprojection_ssim --use_reprojection_l1 (默认)
        
        self.parser.add_argument("--use_reprojection_ssim",
                                help="enable SSIM in reprojection loss (default: True). "
                                     "When enabled with use_reprojection_l1, uses reprojection_ssim_weight*SSIM + reprojection_l1_weight*L1. "
                                     "When disabled, uses L1 only (if use_reprojection_l1 is enabled)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--use_reprojection_l1",
                                help="enable L1 in reprojection loss (default: True). "
                                     "When enabled with use_reprojection_ssim, uses reprojection_ssim_weight*SSIM + reprojection_l1_weight*L1. "
                                     "When disabled, uses SSIM only (if use_reprojection_ssim is enabled). "
                                     "If both are disabled, returns zero loss",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--reprojection_ssim_weight",
                                type=float,
                                default=0.85,
                                help="weight for SSIM in reprojection loss. "
                                     "Used together with reprojection_l1_weight. "
                                     "Default: 0.85. Note: weights don't need to sum to 1.0")
        self.parser.add_argument("--reprojection_l1_weight",
                                type=float,
                                default=0.15,
                                help="weight for L1 in reprojection loss (reprojection image vs original image). "
                                     "This is DIFFERENT from teacher-student L1 loss. "
                                     "Used together with reprojection_ssim_weight. "
                                     "Default: 0.15. Note: weights don't need to sum to 1.0")
        
        # ====================================================================
        # GEOMETRIC REGULARIZATION LOSSES (几何正则化损失)
        # ====================================================================
        
        # Smoothness Loss (平滑损失)
        # 作用：惩罚相邻像素的深度不连续性，使用边缘感知权重
        # 影响：⚠️ 最不利于细节保留，会平滑掉小尺度细节
        # 建议：凸显细节时使用低权重(0.01-0.05)或禁用
        
        self.parser.add_argument("--use_smoothness_loss",
                                help="enable edge-aware smoothness loss (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--smoothness_weight",
                                type=float,
                                default=0.02,
                                help="weight for smoothness loss (⚠️ high value smooths details). "
                                     "Recommended: 0.01-0.05 for detail preservation, 0.1-0.2 for balance")
        
        # Smoothness computation mode (not a loss switch, but a computation mode)
        self.parser.add_argument("--disable_plane_smoothness",
                                help="use edge-aware smoothness on disparity instead of plane coefficients",
                                action="store_true")
        
        # Plane Regularization Loss (平面正则化损失)
        # 作用：强制4个点共面，假设场景中存在平面结构
        # 影响：⚠️ 不利于细节保留，可能错误地平滑小物体和纹理
        # 建议：凸显细节时使用低权重(0.1-0.5)或禁用
        
        self.parser.add_argument("--use_plane_regularization",
                                help="enable plane regularization loss (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--num_plane_keysets",
                                type=int,
                                default=512,
                                help="number of keysets for plane regularization")
        self.parser.add_argument("--plane_weight",
                                type=float,
                                default=0.1,
                                help="weight for plane regularization loss (⚠️ high value may smooth details). "
                                     "Recommended: 0.1-0.5 for detail preservation")
        
        # Line Regularization Loss (线段正则化损失)
        # 作用：强制3个点共线，假设场景中存在直线结构
        # 影响：⚠️ 中等程度不利于细节保留
        # 建议：凸显细节时使用低权重(0.1-0.2)或禁用
        
        self.parser.add_argument("--use_line_regularization",
                                help="enable line regularization loss (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--num_line_keysets",
                                type=int,
                                default=128,
                                help="number of keysets for line regularization")
        self.parser.add_argument("--line_weight",
                                type=float,
                                default=0.1,
                                help="weight for line regularization loss (⚠️ may smooth details). "
                                     "Recommended: 0.1-0.2 for detail preservation")
        
        # ====================================================================
        # DIFFUSION MODEL CONFIGURATION (扩散模型配置)
        # ====================================================================
        # Note: Diffusion decoder is always used, this flag is kept for backward compatibility
        
        self.parser.add_argument("--use_diffusion",
                                help="[DEPRECATED] Diffusion is always enabled",
                                action="store_true",
                                default=True)
        
        # DDIM Loss (Denoising Diffusion Implicit Model Loss)
        # 作用：扩散模型的去噪损失，训练噪声预测网络
        # 影响：✅ 有利于细节生成（扩散模型本身设计用于生成细节）
        # 建议：通常0.5-2.0，如果想更专注于跟随教师，可以降低权重
        
        self.parser.add_argument("--use_ddim_loss",
                                help="enable DDIM diffusion loss (denoising loss) (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--diffusion_ddim_weight",
                                type=float,
                                default=0.5,
                                help="weight for DDIM diffusion loss (denoising loss). "
                                     "✅ Helps preserve details. "
                                     "Recommended: 0.5-2.0. Lower if focusing more on teacher alignment")
        
        # Diffusion Training Parameters (扩散训练参数)
        # Diffusion Training Timesteps (扩散训练时间步数)
        # 含义：训练时扩散调度器的总时间步数
        # 作用：定义噪声调度，更多步数提供更细粒度的噪声级别
        # 格式：[scale_2, scale_1, scale_0]
        # 建议：通常保持默认值，除非需要特殊调整
        
        self.parser.add_argument("--diffusion_timesteps",
                                nargs="+",
                                type=int,
                                default=[500, 400, 300],
                                help="number of training timesteps for each scale [coarse to fine]. "
                                     "Defines noise schedule. Format: [scale_2, scale_1, scale_0]")
        
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
                                default=[20, 15, 12],  # 从[5,4,3]增加到[8,6,5]，增强细节
                                help="number of diffusion inference steps for each scale [coarse to fine]. "
                                     "More steps = better details but slower. "
                                     "Format: [scale_2, scale_1, scale_0] or [scale_0] for single scale")
        
        # ====================================================================
        # TEACHER MODEL CONFIGURATION (教师模型配置)
        # ====================================================================
        # Note: Depth Anything V3 is always used as teacher model
        
        # Teacher Model Settings (教师模型设置)
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
        
        # Teacher-Student Alignment Losses (学生-教师对齐损失)
        # 用于知识蒸馏，使学生模型输出逼近教师模型
        
        # Teacher-Student L1 Loss (Mean Absolute Error Loss)
        # 作用：学生模型与教师模型预测的一致性损失（基础对齐损失）
        # 注意：这是知识蒸馏损失，计算学生预测与教师预测的L1距离
        #   与重投影L1损失不同（重投影L1计算重投影图像与原图像的L1距离）
        # 影响：✅ 有助于学生模型学习教师的知识
        # 建议：通常1.0-10.0，如果想让学生模型尽可能逼近教师，可以设置5.0-10.0
        
        self.parser.add_argument("--use_teacher_student_l1",
                                help="enable L1 loss between teacher and student predictions (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--teacher_student_l1_weight",
                                type=float,
                                default=5.0,
                                help="weight for L1 loss between teacher and student predictions. "
                                     "This is DIFFERENT from reprojection L1 loss. "
                                     "Higher = stronger teacher-student alignment. "
                                     "Recommended: 5.0-10.0 for maximum alignment")
        
        # MSE Loss (Mean Squared Error Loss)
        # 作用：MSE损失对大误差更敏感，有助于快速收敛
        # 影响：✅ 有助于学生模型快速逼近教师模型
        # 建议：权重通常设置为0.5-2.0
        
        self.parser.add_argument("--use_teacher_student_mse",
                                help="enable MSE loss between teacher and student predictions (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--teacher_student_mse_weight",
                                type=float,
                                default=2.0,
                                help="weight for MSE loss between teacher and student predictions. "
                                     "Recommended: 0.5-2.0. Used together with L1 loss for stronger alignment")
        
        # SSIM Loss (Structural Similarity Index Measure Loss)
        # 作用：SSIM损失关注结构相似性，有助于整体结构对齐
        # 影响：✅ 有助于学生模型在结构上与教师对齐
        # 建议：权重通常设置为0.5-2.0
        
        self.parser.add_argument("--use_teacher_student_ssim",
                                help="enable SSIM loss between teacher and student predictions (default: True)",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--teacher_student_ssim_weight",
                                type=float,
                                default=2.0,
                                help="weight for SSIM loss between teacher and student predictions. "
                                     "Recommended: 0.5-2.0. Helps align structural similarity")
        
        # ====================================================================
        # MASK TRAINING LOSS (Mask训练损失)
        # ====================================================================
        # Mask训练：使用随机mask对特征进行遮挡，增强模型鲁棒性
        # 原理：类似dropout，但作用于特征空间，强制模型从部分信息恢复深度
        # 优势：增强泛化能力，处理遮挡情况，有助于细节保留
        
        self.parser.add_argument("--use_mask_training",
                                help="enable mask training loss (feature-level masking for robustness) (default: True)",
                                action="store_true",
                                default=True)
        
        self.parser.add_argument("--mask_probability",
                                type=float,
                                default=0.2,
                                help="probability of masking a feature location (0.2 = 20%% masked, 80%% kept). "
                                     "Options: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5")
        
        self.parser.add_argument("--mask_loss_weight",
                                type=float,
                                default=0.1,
                                help="weight for mask training loss (consistency between masked and full predictions). "
                                     "Recommended: 0.1-0.2")

        # ====================================================================
        # SYSTEM & DEVICE OPTIONS (系统和设备配置)
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

        # ====================================================================
        # MODEL LOADING OPTIONS (模型加载配置)
        # ====================================================================
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose", "scalenet", "regression"])

        # ====================================================================
        # LOGGING & DEBUGGING OPTIONS (日志和调试配置)
        # ====================================================================
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

        # ====================================================================
        # EVALUATION OPTIONS (评估配置)
        # ====================================================================
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
