# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from layers import *

import datasets
import networks
from IPython import embed

import torch.nn.functional as F
import os


class Trainer:
    def __init__(self, options):
        self.opt = options
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.opt.log_dir, f"{self.opt.model_name}_{timestamp}")
        print("-> Log path: {}".format(self.log_path))
        print("-> Model name: {}".format(self.opt.model_name))
        self.debug_no_save = getattr(self.opt, "debug_no_save", False)
        if self.debug_no_save:
            print("⚠ Debug mode enabled: training artifacts will not be written to disk.")

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        # 设置设备（支持指定GPU）
        if self.opt.no_cuda:
            self.device = torch.device("cpu")
        else:
            if self.opt.cuda_device is not None:
                # 使用cuda_device参数（如 "cuda:1"）
                self.device = torch.device(self.opt.cuda_device)
            else:
                # 使用gpu_id参数
                gpu_id = getattr(self.opt, 'gpu_id', 0)
                if torch.cuda.is_available():
                    if gpu_id >= torch.cuda.device_count():
                        print(f"⚠️ Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
                        gpu_id = 0
                    self.device = torch.device(f"cuda:{gpu_id}")
                else:
                    print("⚠️ Warning: CUDA not available. Using CPU instead.")
                    self.device = torch.device("cpu")
        
        print("Using device:", self.device)
        if not self.opt.no_cuda and torch.cuda.is_available():
            print(f"  GPU ID: {self.device.index if hasattr(self.device, 'index') else 'N/A'}")
            if hasattr(self.device, 'index') and self.device.index is not None:
                print(f"  GPU Name: {torch.cuda.get_device_name(self.device.index)}")

        self.num_scales = len(self.opt.scales)
        print("Training scales:", self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        print("Input frames:", self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        print("Pose frames:", self.num_pose_frames)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models["encoder"] = networks.ResnetEncoder(
        self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        
        res_out_channel = [64, 64, 128, 256, 512]

        #scale factor prediction
        self.models["scalenet"] = networks.ScaleNetwork(res_out_channel)
        self.models["scalenet"].to(self.device)
        self.parameters_to_train += list(self.models["scalenet"].parameters())
        
        self.models["regression"] = nn.ModuleList([
                        networks.ProbabilisticScaleRegressionHead(in_channels=out_channels) for out_channels in res_out_channel])
        self.models["regression"].to(self.device)
        self.parameters_to_train += list(self.models["regression"].parameters())
        
        # 深度解码器初始化 - 使用扩散解码器 + Depth Anything V3 教师模型
        print("=" * 60)
        print("🔄 使用扩散深度解码器 + Depth Anything V3 教师模型")
        print("=" * 60)
        
        # 创建 Depth Anything V3 教师模型
        print(f"📦 加载 Depth Anything V3 教师模型: {self.opt.depth_anything_v3_model}")
        self.models["depth_anything_v3_teacher"] = networks.create_depth_anything_v3_teacher(
            model_name=self.opt.depth_anything_v3_model,
            device=self.device,
            scales=self.opt.scales,
            input_size=(self.opt.height, self.opt.width),
            model_path=self.opt.depth_anything_v3_weights
        )
        self.models["depth_anything_v3_teacher"].eval()
        print("✓ Depth Anything V3 教师模型加载成功")
        
        # 创建学生模型（扩散解码器）
        # Note: Diffusion decoder outputs single-channel disparity (num_output_channels=1)
        # and does not use PixelCoorModu (follows MonoDiffusion architecture)
        self.models["depth"] = networks.DepthDecoderDiffusion(
            self.models["encoder"].num_ch_enc, 
            self.opt.scales,
            num_output_channels=1,
            use_skips=True,
            diffusion_steps=self.opt.diffusion_steps,
            diffusion_timesteps=self.opt.diffusion_timesteps)
        print("✓ 扩散解码器初始化成功")
        print("=" * 60)
        
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # ====================================================================
        # Pose 模块初始化
        # ====================================================================
        # Pose模块采用共享编码器 + 三个独立解码器的架构：
        # 1. pose_encoder: 共享的ResNet编码器，用于提取多帧图像的特征
        #    输入: 拼接后的多帧图像 (B, 6, H, W) 或 (B, 3*N, H, W)
        #    输出: 5个多尺度特征图列表
        #
        # 2. pose_rec: 用于predict_poses_ori，预测原始输入帧之间的pose
        #    输入: pose_encoder提取的特征
        #    输出: axisangle和translation，用于生成cam_T_cam_ori变换矩阵
        #
        # 3. pose: 用于predict_poses_second，预测第一次重投影后帧之间的pose
        #    输入: pose_encoder提取的特征（基于第一次重投影后的图像）
        #    输出: axisangle和translation，用于生成cam_T_cam_second变换矩阵
        #
        # 4. pose_third: 用于predict_poses_third，预测第二次重投影后帧之间的pose
        #    输入: pose_encoder提取的特征（基于第二次重投影后的图像）
        #    输出: axisangle和translation，用于生成cam_T_cam_third变换矩阵
        #
        # 数据流: 原始图像 → pose_rec → 第一次重投影 → pose → 第二次重投影 → pose_third
        # ====================================================================
        self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers,
                                                             self.opt.weights_init == "pretrained",
                                                             num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                                   num_input_features=1,
                                                   num_frames_to_predict_for=(self.num_pose_frames-1))
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.models["pose_rec"] = networks.PoseDecoderRec(self.models["pose_encoder"].num_ch_enc,
                                                   num_input_features=1,
                                                   num_frames_to_predict_for=(self.num_pose_frames-1))
        self.models["pose_rec"].to(self.device)
        self.parameters_to_train += list(self.models["pose_rec"].parameters())

        self.models["pose_third"] = networks.PoseDecoderThird(self.models["pose_encoder"].num_ch_enc,
                                                   num_input_features=1,
                                                   num_frames_to_predict_for=(self.num_pose_frames-1))
        self.models["pose_third"].to(self.device)
        self.parameters_to_train += list(self.models["pose_third"].parameters())

        # 使用AdamW优化器（带weight decay，有助于正则化和稳定性）
        # 如果未指定weight_decay，使用默认值1e-2（参考MonoDiffusion）
        weight_decay = getattr(self.opt, 'weight_decay', 1e-2)
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate, weight_decay=weight_decay)
        
        # 改进的学习率调度器：使用更平滑的衰减策略
        # 方案1: 改进的StepLR（减小gamma，避免突然下降）
        # 方案2: 使用CosineAnnealingLR（更平滑的衰减）
        if getattr(self.opt, 'use_cosine_scheduler', False):
            # 使用CosineAnnealingLR，更平滑的衰减
            T_max = self.opt.num_epochs * self.num_total_steps // len(self.train_loader)
            self.model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.model_optimizer, T_max=T_max, eta_min=1e-6)
        else:
            # 改进的StepLR：增大step_size，减小gamma，避免突然下降
            step_size = max(self.opt.scheduler_step_size, 30)  # 至少30个epoch
            gamma = getattr(self.opt, 'scheduler_gamma', 0.5)  # 默认0.5而不是0.1
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, step_size, gamma)
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"nyu": datasets.NYUDataset}
        self.dataset = datasets_dict[self.opt.dataset]
        # print("Using dataset:", self.opt.dataset)

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        print("Using split file:", fpath)

        train_filenames = readlines(fpath.format("train"))
        # print("train_filenames:", train_filenames[0:5])
        val_filenames = readlines(fpath.format("val"))
        # print("val_filenames:", val_filenames[0:5])
        img_ext = '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        use_plane_reg = getattr(self.opt, 'use_plane_regularization', True)
        use_line_reg = getattr(self.opt, 'use_line_regularization', True)
        
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext,
            return_plane=use_plane_reg,
            num_plane_keysets = self.opt.num_plane_keysets,
            return_line=use_line_reg,
            num_line_keysets = self.opt.num_line_keysets)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext,
            return_plane=use_plane_reg,
            num_plane_keysets = self.opt.num_plane_keysets,
            return_line=use_line_reg,
            num_line_keysets = self.opt.num_line_keysets)

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        if not self.debug_no_save:
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        use_reprojection_ssim = getattr(self.opt, 'use_reprojection_ssim', True)
        if use_reprojection_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        # self.project_homo = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "de/log10","da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        # 打印所有配置项
        self.print_config()

        if not self.debug_no_save:
            self.save_opts()

    def print_config(self):
        """打印所有训练配置项，方便审查"""
        print("\n" + "=" * 80)
        print("训练配置项汇总")
        print("=" * 80)
        
        # 路径配置
        print("\n【路径配置】")
        print(f"  数据路径: {self.opt.data_path}")
        print(f"  日志目录: {self.log_path}")
        print(f"  模型名称: {self.opt.model_name}")
        print(f"  数据集: {self.opt.dataset}")
        print(f"  数据分割: {self.opt.split}")
        if self.opt.load_weights_folder:
            print(f"  加载权重文件夹: {self.opt.load_weights_folder}")
            print(f"  加载的模型: {self.opt.models_to_load}")
        
        # 模型配置
        print("\n【模型配置】")
        print(f"  ResNet层数: {self.opt.num_layers}")
        print(f"  权重初始化: {self.opt.weights_init}")
        print(f"  输入图像尺寸: {self.opt.height} x {self.opt.width}")
        print(f"  训练尺度: {self.opt.scales}")
        print(f"  输入帧ID: {self.opt.frame_ids}")
        print(f"  Pose模型输入: {self.opt.pose_model_input}")
        print(f"  深度范围: [{self.opt.min_depth}, {self.opt.max_depth}]")
        
        # 扩散模型配置
        print("\n【扩散模型配置】")
        print(f"  使用扩散解码器: True (固定)")
        print(f"  Depth Anything V3模型: {self.opt.depth_anything_v3_model}")
        if self.opt.depth_anything_v3_weights:
            print(f"  DA3权重路径: {self.opt.depth_anything_v3_weights}")
        print(f"  学生-教师L1损失权重: {self.opt.teacher_student_l1_weight}")
        print(f"  扩散DDIM损失权重: {self.opt.diffusion_ddim_weight}")
        print(f"  扩散推理步数: {self.opt.diffusion_steps}")
        print(f"  扩散训练时间步: {self.opt.diffusion_timesteps}")
        print(f"  Mask训练: {'启用' if getattr(self.opt, 'use_mask_training', True) else '禁用'}")
        if getattr(self.opt, 'use_mask_training', True):
            print(f"  Mask概率: {getattr(self.opt, 'mask_probability', 0.2)}")
            print(f"  Mask损失权重: {getattr(self.opt, 'mask_loss_weight', 0.1)}")
        
        # 训练配置
        print("\n【训练配置】")
        print(f"  批次大小: {self.opt.batch_size}")
        print(f"  训练轮数: {self.opt.num_epochs}")
        print(f"  学习率: {self.opt.learning_rate}")
        print(f"  学习率调度器步长: {self.opt.scheduler_step_size}")
        print(f"  数据加载线程数: {self.opt.num_workers}")
        print(f"  使用CUDA: {not self.opt.no_cuda}")
        
        # 损失函数配置
        print("\n【损失函数配置】")
        use_photometric = getattr(self.opt, 'use_photometric_loss', True)
        use_reprojection_ssim = getattr(self.opt, 'use_reprojection_ssim', True)
        use_smoothness = getattr(self.opt, 'use_smoothness_loss', True)
        use_plane_reg = getattr(self.opt, 'use_plane_regularization', True)
        use_line_reg = getattr(self.opt, 'use_line_regularization', True)
        use_l1 = getattr(self.opt, 'use_teacher_student_l1', True)
        use_mse = getattr(self.opt, 'use_teacher_student_mse', True)
        use_ts_ssim = getattr(self.opt, 'use_teacher_student_ssim', True)
        use_ddim = getattr(self.opt, 'use_ddim_loss', True)
        use_mask = getattr(self.opt, 'use_mask_training', True)
        
        print(f"  Photometric Loss: {'启用' if use_photometric else '禁用'} (权重: {self.opt.photometric_weight})")
        if use_photometric:
            print(f"    重投影损失组件权重: ori={getattr(self.opt, 'reprojection_ori_weight', 0.25)}, "
                  f"virtual={getattr(self.opt, 'reprojection_virtual_weight', 1.0)}, "
                  f"new={getattr(self.opt, 'reprojection_new_weight', 1.0)}")
        print(f"  Reprojection SSIM: {'启用' if use_reprojection_ssim else '禁用'}")
        print(f"  Smoothness Loss: {'启用' if use_smoothness else '禁用'} (权重: {self.opt.smoothness_weight})")
        print(f"  Plane Regularization: {'启用' if use_plane_reg else '禁用'} (权重: {self.opt.plane_weight})")
        if use_plane_reg:
            print(f"    平面keysets数量: {self.opt.num_plane_keysets}")
        print(f"  Line Regularization: {'启用' if use_line_reg else '禁用'} (权重: {self.opt.line_weight})")
        if use_line_reg:
            print(f"    线keysets数量: {self.opt.num_line_keysets}")
        print(f"  Teacher-Student L1: {'启用' if use_l1 else '禁用'} (权重: {self.opt.teacher_student_l1_weight})")
        print(f"  Reprojection SSIM权重: {getattr(self.opt, 'reprojection_ssim_weight', 0.85)}")
        print(f"  Reprojection L1权重: {getattr(self.opt, 'reprojection_l1_weight', 0.15)}")
        print(f"  Teacher-Student MSE: {'启用' if use_mse else '禁用'} (权重: {getattr(self.opt, 'teacher_student_mse_weight', 1.0)})")
        print(f"  Teacher-Student SSIM: {'启用' if use_ts_ssim else '禁用'} (权重: {getattr(self.opt, 'teacher_student_ssim_weight', 1.0)})")
        print(f"  DDIM Loss: {'启用' if use_ddim else '禁用'} (权重: {self.opt.diffusion_ddim_weight})")
        print(f"  Mask Training: {'启用' if use_mask else '禁用'} (权重: {getattr(self.opt, 'mask_loss_weight', 0.1)})")
        
        # 日志配置
        print("\n【日志配置】")
        print(f"  日志频率: {self.opt.log_frequency}")
        print(f"  保存频率: {self.opt.save_frequency}")
        print(f"  调试模式(不保存): {self.debug_no_save}")
        
        # 评估配置
        print("\n【评估配置】")
        print(f"  禁用中位数缩放: {self.opt.disable_median_scaling}")
        print(f"  预测深度缩放因子: {self.opt.pred_depth_scale_factor}")
        print(f"  后处理: {self.opt.post_process}")
        print(f"  评估分割: {self.opt.eval_split}")
        print(f"  禁用评估: {self.opt.no_eval}")
        
        # 设备信息
        print("\n【设备信息】")
        print(f"  使用设备: {self.device}")
        if not self.opt.no_cuda and torch.cuda.is_available():
            print(f"  CUDA设备数量: {torch.cuda.device_count()}")
            if hasattr(self.device, 'index') and self.device.index is not None:
                print(f"  指定GPU ID: {self.device.index}")
                print(f"  当前CUDA设备: {torch.cuda.current_device()}")
                print(f"  CUDA设备名称: {torch.cuda.get_device_name(self.device.index)}")
            else:
                print(f"  当前CUDA设备: {torch.cuda.current_device()}")
                print(f"  CUDA设备名称: {torch.cuda.get_device_name(0)}")
        
        # 数据集信息
        print("\n【数据集信息】")
        print(f"  训练样本数: {len(self.train_loader.dataset)}")
        print(f"  验证样本数: {len(self.val_loader.dataset)}")
        print(f"  总训练步数: {self.num_total_steps}")
        
        # 模型参数统计
        print("\n【模型参数统计】")
        total_params = sum(p.numel() for p in self.parameters_to_train)
        trainable_params = sum(p.numel() for p in self.parameters_to_train if p.requires_grad)
        print(f"  可训练参数总数: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  需要梯度的参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        print("\n" + "=" * 80)
        print("配置审查完成，开始训练...")
        print("=" * 80 + "\n")

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.val()
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            torch.cuda.empty_cache()
            self.run_epoch()
            if (not self.debug_no_save) and ((self.epoch + 1) % self.opt.save_frequency == 0):
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        run_step = 0
        loss_sum = 0.0

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            
            # 计算梯度范数（用于监控，在裁剪之前）
            total_grad_norm = 0.0
            param_count = 0
            for p in self.parameters_to_train:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                total_grad_norm = total_grad_norm ** (1. / 2)
                # 保存到losses中，用于后续记录
                losses['grad_norm'] = torch.tensor(total_grad_norm).to(self.device)
            
            # 添加梯度裁剪，防止梯度爆炸（280000步崩溃的可能原因）
            max_grad_norm = getattr(self.opt, 'max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=max_grad_norm)
            
            self.model_optimizer.step()
            
            # Mask训练（如果启用，默认True）
            use_mask_training = getattr(self.opt, 'use_mask_training', True)
            if use_mask_training:
                outputs_mask, losses_mask = self.process_batch_mask(inputs, outputs)
                self.model_optimizer.zero_grad()
                losses_mask.backward()
                
                # 梯度裁剪
                max_grad_norm = getattr(self.opt, 'max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, max_norm=max_grad_norm)
                
                self.model_optimizer.step()
            else:
                outputs_mask = None
                losses_mask = None

            duration = time.time() - before_op_time

            run_step += 1
            loss_sum += losses["loss"].cpu().data

            # log less frequently to save time & disk space
            # 前100步每10步写入一次，100-5000步每200步写入一次，之后每5000步写入一次
            should_log = (
                (self.step > 0 and self.step <= 100 and self.step % 10 == 0) or
                (self.step > 100 and self.step < 5000 and self.step % 200 == 0) or
                (self.step >= 5000 and self.step % 5000 == 0)
            )

            if should_log:
                self.log_time(batch_idx, duration, loss_sum/run_step)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                # 记录mask损失（如果启用）
                use_mask_training = getattr(self.opt, 'use_mask_training', True)
                if use_mask_training and losses_mask is not None:
                    losses['mask_loss'] = losses_mask
                
                self.log("train", inputs, outputs, losses, outputs_mask)
            self.step += 1

        self.model_lr_scheduler.step()

        self.val()


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses."""
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # 使用 Depth Anything V3 教师模型生成伪GT（教师模型已冻结，使用eval模式）
        with torch.no_grad():
            # Depth Anything V3 需要 RGB 图像输入 (0-1 范围)
            pre_outputs = self.models["depth_anything_v3_teacher"](inputs[("color_aug", 0, 0)])
        
        # 学生模型前向传播
        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        
        # 准备扩散的伪GT（从教师模型的预测）
        gt_for_diffusion = {}
        for scale in self.opt.scales:
            # 使用教师的 disp 作为扩散的目标
            gt_for_diffusion[("disp_diffusion", scale)] = pre_outputs[("disp", scale)].detach()
        
        # 使用扩散解码器
        # Note: Diffusion decoder does not use norm_pix_coords (follows MonoDiffusion)
        outputs = self.models["depth"](features, gt_for_diffusion)
        
        # 保存教师的预测，用于后续损失计算和可视化
        for scale in self.opt.scales:
            outputs[("predisp", scale)] = pre_outputs[("disp", scale)]

        outputs.update(self.predict_poses_ori(inputs))
        self.generate_images_pred_ori(inputs, outputs)

        outputs.update(self.predict_poses_second(inputs, outputs))
        self.generate_images_pred_second(inputs, outputs)

        outputs.update(self.predict_poses_third(inputs, outputs))
        self.generate_images_pred_third(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)
        
        # ====================================================================
        # TEACHER-STUDENT ALIGNMENT LOSSES (学生-教师对齐损失)
        # ====================================================================
        
        # Teacher-Student L1损失：学生与教师的一致性（基础对齐损失）
        # 注意：这是知识蒸馏损失，与重投影L1损失不同
        use_l1_loss = getattr(self.opt, 'use_teacher_student_l1', True)
        teacher_student_l1_loss = 0
        if use_l1_loss:
            for scale in self.opt.scales:
                teacher_student_l1_loss += F.l1_loss(outputs[("predisp", scale)], outputs[("disp", scale)])
            losses['teacher_student_l1'] = teacher_student_l1_loss / len(self.opt.scales)
        else:
            losses['teacher_student_l1'] = torch.tensor(0.0).to(self.device)
        
        # MSE损失：学生与教师的一致性（更强调大误差，有助于快速收敛）
        use_mse_loss = getattr(self.opt, 'use_teacher_student_mse', True)
        mse_loss = 0
        if use_mse_loss:
            for scale in self.opt.scales:
                mse_loss += F.mse_loss(outputs[("predisp", scale)], outputs[("disp", scale)])
            losses['mse'] = mse_loss / len(self.opt.scales)
        else:
            losses['mse'] = torch.tensor(0.0).to(self.device)
        
        # SSIM损失：学生与教师的结构相似性（关注整体结构一致性）
        use_ssim_loss = getattr(self.opt, 'use_teacher_student_ssim', True)
        ssim_loss = 0
        if use_ssim_loss and hasattr(self, 'ssim'):
            for scale in self.opt.scales:
                teacher_disp = outputs[("predisp", scale)]
                student_disp = outputs[("disp", scale)]
                # SSIM损失（1 - SSIM），值越小表示越相似
                ssim_val = self.ssim(teacher_disp * 5, student_disp * 5).mean()
                ssim_loss += ssim_val
            losses['ssim_teacher_student'] = ssim_loss / len(self.opt.scales)
        else:
            losses['ssim_teacher_student'] = torch.tensor(0.0).to(self.device)
        
        # ====================================================================
        # DIFFUSION MODEL LOSSES (扩散模型损失)
        # ====================================================================
        
        # DDIM损失：扩散模型的去噪损失
        use_ddim_loss = getattr(self.opt, 'use_ddim_loss', True)
        ddim_loss = 0
        if use_ddim_loss:
            for scale in self.opt.scales:
                if ("ddim_loss", scale) in outputs:
                    ddim_loss += outputs[("ddim_loss", scale)]
            losses['ddim'] = ddim_loss / len(self.opt.scales) if ddim_loss != 0 else torch.tensor(0.0).to(self.device)
        else:
            losses['ddim'] = torch.tensor(0.0).to(self.device)
        
        # ====================================================================
        # TOTAL LOSS COMPUTATION (总损失计算)
        # ====================================================================
        
        # 保存原始光度损失
        losses['photometric'] = losses["loss"].clone()
        
        # 总损失 = 光度损失 + L1损失 + MSE损失 + SSIM损失 + DDIM损失
        # 注意：增加对齐损失权重有助于学生模型更好跟随教师模型
        use_photometric_loss = getattr(self.opt, 'use_photometric_loss', True)
        photometric_weight = getattr(self.opt, 'photometric_weight', 0.2) if use_photometric_loss else 0.0
        l1_weight = getattr(self.opt, 'teacher_student_l1_weight', 5.0) if use_l1_loss else 0.0
        mse_weight = getattr(self.opt, 'teacher_student_mse_weight', 1.0) if use_mse_loss else 0.0
        ssim_weight = getattr(self.opt, 'teacher_student_ssim_weight', 1.0) if use_ssim_loss else 0.0
        ddim_weight = getattr(self.opt, 'diffusion_ddim_weight', 1.0) if use_ddim_loss else 0.0
        
        losses["loss"] = (photometric_weight * losses['photometric'] + 
                         l1_weight * losses['teacher_student_l1'] + 
                         mse_weight * losses['mse'] +
                         ssim_weight * losses['ssim_teacher_student'] +
                         ddim_weight * losses['ddim'])
        
        # 记录各损失项的独立值，方便分析
        losses['loss_photometric'] = losses['photometric']
        losses['loss_teacher_student_l1'] = losses['teacher_student_l1']
        losses['loss_mse'] = losses['mse']
        losses['loss_ssim_teacher_student'] = losses['ssim_teacher_student']
        losses['loss_ddim'] = losses['ddim']

        return outputs, losses

    def process_batch_mask(self, inputs, outputs):
        """Mask训练：使用随机mask对特征进行遮挡，增强模型鲁棒性
        
        原理：
        - 生成随机mask（部分特征被遮挡）
        - 使用masked特征进行深度预测
        - 损失：masked预测与完整预测的一致性
        
        优势：
        - 增强泛化能力
        - 处理遮挡情况
        - 有助于细节保留（特征级别的mask不会直接破坏细节）
        """
        # 获取编码器特征
        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        
        # 生成随机mask
        # mask_probability: 被遮挡的概率（0.2 = 20%遮挡，80%保留）
        mask_prob = getattr(self.opt, 'mask_probability', 0.2)
        b, c, h, w = features[0].shape
        mask_initial = (torch.rand(b, 1, h, w).to(self.device) > mask_prob).float()
        
        # 应用mask到特征（只mask第一个特征，其他特征通过插值）
        masked_features = []
        for i, feat in enumerate(features):
            if i == 0:
                masked_feat = feat * mask_initial
            else:
                # 将mask插值到对应尺寸
                mask_resized = F.interpolate(mask_initial, size=feat.shape[-2:], mode='nearest')
                masked_feat = feat * mask_resized
            masked_features.append(masked_feat)
        
        # 准备GT（使用完整预测作为伪GT）
        gt_for_diffusion = {}
        for scale in self.opt.scales:
            gt_for_diffusion[("disp_diffusion", scale)] = outputs[("disp", scale)].detach()
        
        # 使用masked特征进行预测
        outputs_mask = self.models["depth"](masked_features, gt_for_diffusion)
        
        # 计算mask损失：masked预测与完整预测的一致性
        mask_loss = 0
        mask_loss_weight = getattr(self.opt, 'mask_loss_weight', 0.1)
        for scale in self.opt.scales:
            mask_loss += mask_loss_weight * F.l1_loss(
                outputs_mask[("disp", scale)],
                outputs[("disp", scale)].detach()
            )
        mask_loss /= len(self.opt.scales)
        
        # 保存mask信息用于可视化
        outputs_mask[("mask", 0)] = mask_initial
        
        return outputs_mask, mask_loss

    def predict_poses_ori(self, inputs):
        """预测原始输入帧之间的pose（第一次pose预测）
        
        使用pose_encoder + pose_rec解码器
        输入: 原始输入图像帧
        输出: cam_T_cam_ori变换矩阵，用于第一次图像重投影
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            half_source_frames = len(self.opt.frame_ids[1:]) // 2

            negative_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames:0:-1]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[negative_half[i + 1]], pose_feats[negative_half[i]]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose_rec"](pose_inputs)

                outputs[("axisangle", negative_half[i + 1], negative_half[i])] = axisangle
                outputs[("translation", negative_half[i + 1], negative_half[i])] = translation

                # Invert the matrix if the frame id is negative
                if i == 0:
                    outputs[("cam_T_cam_ori", 0, negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                else:
                    outputs[("cam_T_cam_ori", negative_half[i], negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                    outputs[("cam_T_cam_ori", 0, negative_half[i + 1])] = \
                        outputs[("cam_T_cam_ori", 0, negative_half[i])] @ \
                        outputs[("cam_T_cam_ori", negative_half[i], negative_half[i + 1])]

            positive_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames + 1:]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[positive_half[i]], pose_feats[positive_half[i + 1]]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose_rec"](pose_inputs)

                outputs[("axisangle", positive_half[i], positive_half[i + 1])] = axisangle
                outputs[("translation", positive_half[i], positive_half[i + 1])] = translation

                if i == 0:
                    outputs[("cam_T_cam_ori", 0, positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                else:
                    outputs[("cam_T_cam_ori", positive_half[i], positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                    outputs[("cam_T_cam_ori", 0, positive_half[i + 1])] = \
                        outputs[("cam_T_cam_ori", 0, positive_half[i])] @ \
                        outputs[("cam_T_cam_ori", positive_half[i], positive_half[i + 1])]

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 1)

            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                outputs[("axisangle", 0, f_i)] = axisangle[:, i:i + 1]
                outputs[("translation", 0, f_i)] = translation[:, i:i + 1]
                outputs[("cam_T_cam_ori", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])

        return outputs

    def predict_poses_second(self, inputs, outputs):
        """预测第一次重投影后帧之间的pose（第二次pose预测）
        
        使用pose_encoder + pose解码器
        输入: 第一次重投影后的图像（color_aug_ori）
        输出: cam_T_cam_second变换矩阵，用于第二次图像重投影
        """

        if self.num_pose_frames == 2:
       

            # select what features the pose network takes as input
            pose_feats = {f_i: outputs[("color_aug_ori", f_i, 0)]  for f_i in self.opt.frame_ids[1:]}
            pose_feats[0] = inputs[("color_aug", 0, 0)]

            half_source_frames = len(self.opt.frame_ids[1:]) // 2  

            negative_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames:0:-1]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[negative_half[i + 1]], pose_feats[negative_half[i]]] 
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose"](pose_inputs)

                outputs[("axisangle", negative_half[i + 1], negative_half[i])] = axisangle
                outputs[("translation", negative_half[i + 1], negative_half[i])] = translation

                # Invert the matrix if the frame id is negative
                if i == 0:
                    outputs[("cam_T_cam_second", 0, negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                else:
                    outputs[("cam_T_cam_second", negative_half[i], negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                    outputs[("cam_T_cam_second", 0, negative_half[i + 1])] = \
                        outputs[("cam_T_cam_second", 0, negative_half[i])] @ \
                        outputs[("cam_T_cam_second", negative_half[i], negative_half[i + 1])]

            positive_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames + 1:]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[positive_half[i]], pose_feats[positive_half[i + 1]]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose"](pose_inputs)

                outputs[("axisangle", positive_half[i], positive_half[i + 1])] = axisangle
                outputs[("translation", positive_half[i], positive_half[i + 1])] = translation

                if i == 0:
                    outputs[("cam_T_cam_second", 0, positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                else:
                    outputs[("cam_T_cam_second", positive_half[i], positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                    outputs[("cam_T_cam_second", 0, positive_half[i + 1])] = \
                        outputs[("cam_T_cam_second", 0, positive_half[i])] @ \
                        outputs[("cam_T_cam_second", positive_half[i], positive_half[i + 1])]

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs_ref = torch.cat(
                [outputs[("color_aug_rot_trans", i, 0)] for i in self.opt.frame_ids[1:]], 1)
            pose_inputs = torch.cat((inputs[("color_aug", 0, 0)], pose_inputs_ref), 1)

            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                outputs[("axisangle", 0, f_i)] = axisangle[:, i:i + 1]#
                outputs[("translation", 0, f_i)] = translation[:, i:i + 1]
                outputs[("cam_T_cam_second", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])

        return outputs

    def predict_poses_third(self, inputs, outputs):
        """预测第二次重投影后帧之间的pose（第三次pose预测）
        
        使用pose_encoder + pose_third解码器
        输入: 第二次重投影后的图像（color_aug）
        输出: cam_T_cam_third变换矩阵，用于第三次图像重投影
        """
       
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: outputs[("color_aug", f_i, 0)]  for f_i in self.opt.frame_ids[1:]}
            pose_feats[0] = inputs[("color_aug", 0, 0)]

            half_source_frames = len(self.opt.frame_ids[1:]) // 2

            negative_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames:0:-1]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[negative_half[i + 1]], pose_feats[negative_half[i]]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose_third"](pose_inputs)

                outputs[("axisangle", negative_half[i + 1], negative_half[i])] = axisangle
                outputs[("translation", negative_half[i + 1], negative_half[i])] = translation

                # Invert the matrix if the frame id is negative
                if i == 0:
                    outputs[("cam_T_cam_third", 0, negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                else:
                    outputs[("cam_T_cam_third", negative_half[i], negative_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=True)
                    outputs[("cam_T_cam_third", 0, negative_half[i + 1])] = \
                        outputs[("cam_T_cam_third", 0, negative_half[i])] @ \
                        outputs[("cam_T_cam_third", negative_half[i], negative_half[i + 1])]

            positive_half = self.opt.frame_ids[:1] + self.opt.frame_ids[half_source_frames + 1:]

            for i in range(half_source_frames):
                pose_inputs = [pose_feats[positive_half[i]], pose_feats[positive_half[i + 1]]]
                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose_third"](pose_inputs)

                outputs[("axisangle", positive_half[i], positive_half[i + 1])] = axisangle
                outputs[("translation", positive_half[i], positive_half[i + 1])] = translation

                if i == 0:
                    outputs[("cam_T_cam_third", 0, positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                else:
                    outputs[("cam_T_cam_third", positive_half[i], positive_half[i + 1])] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=False)
                    outputs[("cam_T_cam_third", 0, positive_half[i + 1])] = \
                        outputs[("cam_T_cam_third", 0, positive_half[i])] @ \
                        outputs[("cam_T_cam_third", positive_half[i], positive_half[i + 1])]

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 1)

            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                outputs[("axisangle", 0, f_i)] = axisangle[:, i:i + 1]
                outputs[("translation", 0, f_i)] = translation[:, i:i + 1]
                outputs[("cam_T_cam_third", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])

        return outputs 

    def val(self):
        """Validate the model on the whole validation set"""
        run_step = 0
        losses_sum = {"loss": 0.0}
        losses_avg = {"loss": 0.0}

        for s in self.opt.scales:
            losses_sum["loss/" + str(s)] = 0.0
            losses_avg["loss/" + str(s)] = 0.0
            losses_sum["smooth_loss/" + str(s)] = 0.0
            losses_avg["smooth_loss/" + str(s)] = 0.0
            use_plane_reg = getattr(self.opt, 'use_plane_regularization', True)
            if use_plane_reg:
                losses_sum["plane_loss/" + str(s)] = 0.0
                losses_avg["plane_loss/" + str(s)] = 0.0
            use_line_reg = getattr(self.opt, 'use_line_regularization', True)
            if use_line_reg:
                losses_sum["line_loss/" + str(s)] = 0.0
                losses_avg["line_loss/" + str(s)] = 0.0

        for name in self.depth_metric_names:
            losses_sum[name] = 0.0
            losses_avg[name] = 0.0

        for batch_idx, inputs in enumerate(self.val_loader):
            run_step += 1
            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                for l, v in losses.items():
                    if l not in losses_sum:
                        losses_sum[l] = 0.0
                        losses_avg[l] = 0.0
                    losses_sum[l] += v

        for l, v in losses_sum.items():
            losses_avg[l] = losses_sum[l] / run_step

        self.log("val", inputs, outputs, losses_avg)

        del inputs, outputs, losses, losses_sum, losses_avg

        self.set_train()



    def generate_images_pred_third(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """    
        
        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        depth_factors = self.models["scalenet"](features)


        scale_predictions = [head(factor) for head, factor in zip(self.models["regression"], depth_factors)]
        max_depth = torch.mean(torch.stack(scale_predictions), dim=0)

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            all_depths = []
            for i in range(self.opt.batch_size):
                disp_i = disp[i:i+1]  # (1, H, W)
                max_depth_i = max_depth[i].item()  

              
                _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, max_depth_i)

           
                if not isinstance(depth_i, torch.Tensor):
                    depth_i = torch.tensor(depth_i)

         
                depth_i = depth_i.view(1, *depth_i.shape[1:])
                all_depths.append(depth_i)

            depth = torch.cat(all_depths)
            outputs[("depth_third", 0, scale)] = depth 

            outputs[("cam_points", 0, scale)] = self.backproject_depth[scale](
                depth, inputs[("norm_pix_coords", scale)])      
            all_depths = []
        
        for scale in self.opt.scales:

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam_third", 0, frame_id)]

                pix_coords = self.project_3d[scale](
                    outputs[("cam_points", 0, scale)], inputs[("K", scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords.permute(0, 2, 3, 1)

                outputs[("color_new", frame_id, scale)] = F.grid_sample(
                   outputs[("color", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                frame_features = self.models["encoder"](inputs[("color_aug", frame_id, 0)])
                frame_depth_factors = self.models["scalenet"](frame_features)
                frame_scale_predictions = [head(factor) for head, factor in zip(self.models["regression"], frame_depth_factors)]
                frame_max_depth = torch.mean(torch.stack(frame_scale_predictions), dim=0)

                frame_all_depths = []
                for j in range(self.opt.batch_size):
                    frame_disp_j = disp[j:j+1]  # (1, H, W)
                    frame_max_depth_j = frame_max_depth[j].item()  

                    
                    _, frame_depth_j = disp_to_depth(frame_disp_j, self.opt.min_depth, frame_max_depth_j)

                   
                    if not isinstance(frame_depth_j, torch.Tensor):
                        frame_depth_j = torch.tensor(frame_depth_j)

                    
                    frame_depth_j = frame_depth_j.view(1, *frame_depth_j.shape[1:])
                    frame_all_depths.append(frame_depth_j)

                frame_depth = torch.cat(frame_all_depths)
                outputs[("depth_third", frame_id, scale)] = frame_depth

    def generate_images_pred_second(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """   
        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        depth_factors = self.models["scalenet"](features)

        
        scale_predictions = [head(factor) for head, factor in zip(self.models["regression"], depth_factors)]
        max_depth = torch.mean(torch.stack(scale_predictions), dim=0)

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            all_depths = []
            for i in range(self.opt.batch_size):
                disp_i = disp[i:i+1]  # (1, H, W)
                max_depth_i = max_depth[i].item()  

                
                _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, max_depth_i)

               
                if not isinstance(depth_i, torch.Tensor):
                    depth_i = torch.tensor(depth_i)

                
                depth_i = depth_i.view(1, *depth_i.shape[1:])
                all_depths.append(depth_i)

            depth = torch.cat(all_depths)
            outputs[("depth_second", 0, scale)] = depth 

            outputs[("cam_points", 0, scale)] = self.backproject_depth[scale](
                depth, inputs[("norm_pix_coords", scale)])      
            all_depths = []
        
        for scale in self.opt.scales:

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam_second", 0, frame_id)]

                pix_coords = self.project_3d[scale](
                    outputs[("cam_points", 0, scale)], inputs[("K", scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords.permute(0, 2, 3, 1)

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    outputs[("color_ori", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                outputs[("color_aug", frame_id, scale)] = F.grid_sample(
                    outputs[("color_aug_ori", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                                
                
                frame_features = self.models["encoder"](inputs[("color_aug", frame_id, 0)])
                frame_depth_factors = self.models["scalenet"](frame_features)
                frame_scale_predictions = [head(factor) for head, factor in zip(self.models["regression"], frame_depth_factors)]
                frame_max_depth = torch.mean(torch.stack(frame_scale_predictions), dim=0)

                frame_all_depths = []
                for j in range(self.opt.batch_size):
                    frame_disp_j = disp[j:j+1]  # (1, H, W)
                    frame_max_depth_j = frame_max_depth[j].item()  

                    
                    _, frame_depth_j = disp_to_depth(frame_disp_j, self.opt.min_depth, frame_max_depth_j)

                   
                    if not isinstance(frame_depth_j, torch.Tensor):
                        frame_depth_j = torch.tensor(frame_depth_j)

                    
                    frame_depth_j = frame_depth_j.view(1, *frame_depth_j.shape[1:])
                    frame_all_depths.append(frame_depth_j)

                frame_depth = torch.cat(frame_all_depths)
                outputs[("depth_second", frame_id, scale)] = frame_depth

    def generate_images_pred_ori(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        depth_factors = self.models["scalenet"](features)

        
        scale_predictions = [head(factor) for head, factor in zip(self.models["regression"], depth_factors)]
        max_depth = torch.mean(torch.stack(scale_predictions), dim=0)

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            all_depths = []
            for i in range(self.opt.batch_size):
                disp_i = disp[i:i+1]  # (1, H, W)
                max_depth_i = max_depth[i].item()  

                
                _, depth_i = disp_to_depth(disp_i, self.opt.min_depth, max_depth_i)

                
                if not isinstance(depth_i, torch.Tensor):
                    depth_i = torch.tensor(depth_i)

                
                depth_i = depth_i.view(1, *depth_i.shape[1:])
                all_depths.append(depth_i)

            depth = torch.cat(all_depths)
            outputs[("depth_ori", 0, scale)] = depth 

            outputs[("cam_points", 0, scale)] = self.backproject_depth[scale](
                depth, inputs[("norm_pix_coords", scale)])      
            all_depths = []
            
              
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam_ori", 0, frame_id)]

                pix_coords = self.project_3d[scale](
                    outputs[("cam_points", 0, scale)], inputs[("K", scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords.permute(0, 2, 3, 1)

                outputs[("color_ori", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                outputs[("color_aug_ori", frame_id, scale)] = F.grid_sample(
                    inputs[("color_aug", frame_id, scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)
                
                frame_features = self.models["encoder"](inputs[("color_aug", frame_id, 0)])
                frame_depth_factors = self.models["scalenet"](frame_features)
                frame_scale_predictions = [head(factor) for head, factor in zip(self.models["regression"], frame_depth_factors)]
                frame_max_depth = torch.mean(torch.stack(frame_scale_predictions), dim=0)

                frame_all_depths = []
                for j in range(self.opt.batch_size):
                    frame_disp_j = disp[j:j+1]  # (1, H, W)
                    frame_max_depth_j = frame_max_depth[j].item()  

                    _, frame_depth_j = disp_to_depth(frame_disp_j, self.opt.min_depth, frame_max_depth_j)

                   
                    if not isinstance(frame_depth_j, torch.Tensor):
                        frame_depth_j = torch.tensor(frame_depth_j)
                    
                    frame_depth_j = frame_depth_j.view(1, *frame_depth_j.shape[1:])
                    frame_all_depths.append(frame_depth_j)

                frame_depth = torch.cat(frame_all_depths)
                outputs[("depth_ori", frame_id, scale)] = frame_depth 


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        
        Uses SSIM + L1 with configurable weights if SSIM is enabled,
        otherwise uses L1 only.
        
        Note: This L1 loss is different from teacher-student L1 loss:
        - Reprojection L1: |reprojected_image - original_image|
        - Teacher-Student L1: |student_prediction - teacher_prediction|
        """
        abs_diff = torch.abs(target - pred)
        reprojection_l1_loss = abs_diff.mean(1, True)

        # 检查是否使用SSIM
        use_ssim = getattr(self.opt, 'use_reprojection_ssim', True)
        
        if use_ssim and hasattr(self, 'ssim'):
            new_pred = pred * 5
            new_target = target * 5
            ssim_loss = self.ssim(new_pred, new_target).mean(1, True)
            # 使用可配置的权重（不需要和为1）
            ssim_weight = getattr(self.opt, 'reprojection_ssim_weight', 0.85)
            l1_weight = getattr(self.opt, 'reprojection_l1_weight', 0.15)
            reprojection_loss = ssim_weight * ssim_loss + l1_weight * reprojection_l1_loss
        else:
            reprojection_loss = reprojection_l1_loss

        return reprojection_loss            

    

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, scale)]

                # Calculate the multi-reprojection loss (photometric loss)
            use_photometric_loss = getattr(self.opt, 'use_photometric_loss', True)
            if use_photometric_loss:
                # 获取三个重投影损失的权重（可配置）
                reproj_ori_weight = getattr(self.opt, 'reprojection_ori_weight', 0.25)
                reproj_virtual_weight = getattr(self.opt, 'reprojection_virtual_weight', 1.0)
                reproj_new_weight = getattr(self.opt, 'reprojection_new_weight', 1.0)
                
                for frame_id in self.opt.frame_ids[1:]:
                    pred_ori = outputs[("color_ori", frame_id, scale)]
                    pred = outputs[("color", frame_id, scale)]
                    pred_new = outputs[("color_new", frame_id, scale)]
                    
                    outputs[("reprojection_losses_ori", frame_id, scale)] = self.compute_reprojection_loss(pred_ori, target)
                    losses["reprojection_losses_ori/{}_{}".format(frame_id, scale)] = outputs[("reprojection_losses_ori", frame_id, scale)].mean()
                    loss += reproj_ori_weight * outputs[("reprojection_losses_ori", frame_id, scale)].mean()

                    outputs[("reprojection_losses_vitual", frame_id, scale)] = self.compute_reprojection_loss(pred, target)
                    losses["reprojection_losses_vitual/{}_{}".format(frame_id, scale)] = outputs[("reprojection_losses_vitual", frame_id, scale)].mean()
                    loss += reproj_virtual_weight * outputs[("reprojection_losses_vitual", frame_id, scale)].mean()

                    outputs[("reprojection_losses_new", frame_id, scale)] = self.compute_reprojection_loss(pred_new, target)
                    losses["reprojection_losses_new/{}_{}".format(frame_id, scale)] = outputs[("reprojection_losses_new", frame_id, scale)].mean()
                    loss += reproj_new_weight * outputs[("reprojection_losses_new", frame_id, scale)].mean()
            else:
                # 如果禁用photometric loss，设置所有reprojection loss为0
                for frame_id in self.opt.frame_ids[1:]:
                    losses["reprojection_losses_ori/{}_{}".format(frame_id, scale)] = torch.tensor(0.0).to(self.device)
                    losses["reprojection_losses_vitual/{}_{}".format(frame_id, scale)] = torch.tensor(0.0).to(self.device)
                    losses["reprojection_losses_new/{}_{}".format(frame_id, scale)] = torch.tensor(0.0).to(self.device)


            # Smoothness loss: use coeff if available (DepthDecoder), else use disp (DiffusionDecoder)
            use_smoothness_loss = getattr(self.opt, 'use_smoothness_loss', True)
            if use_smoothness_loss:
                if self.opt.disable_plane_smoothness or ("coeff", scale) not in outputs:
                    # For diffusion decoder or when plane smoothness is disabled,
                    # compute smoothness on normalized disparity
                    mean_disp = disp.mean(2, True).mean(3, True)
                    norm_disp = disp / (mean_disp + 1e-7)
                    smooth_loss = get_smooth_loss(norm_disp, color)
                else:
                    # For original decoder with pixel coordinate modulation,
                    # compute smoothness on normalized coefficients
                    mean_coeff = outputs[("coeff", scale)].abs().mean(2, True).mean(3, True)
                    norm_coeff = outputs[("coeff", scale)] / (mean_coeff + 1e-7)
                    smooth_loss = get_smooth_loss(norm_coeff, color)
                
                loss += self.opt.smoothness_weight / (2 ** scale) * smooth_loss
                losses["smooth_loss/{}".format(scale)] = smooth_loss
            else:
                losses["smooth_loss/{}".format(scale)] = torch.tensor(0.0).to(self.device)

            # Geometric regularization losses
            point3D = outputs[("cam_points", 0, scale)][:, :3, ...]
            mean_depth = outputs[("depth_ori", 0, scale)].mean(2, True).mean(3)
            norm_point3D = point3D/(mean_depth + 1e-7)

            # Plane regularization loss
            use_plane_reg = getattr(self.opt, 'use_plane_regularization', True)
            
            if use_plane_reg:
                plane_loss = get_plane_loss(inputs[("plane_keysets", 0, scale)], norm_point3D)
                loss += self.opt.plane_weight * plane_loss
                losses["plane_loss/{}".format(scale)] = plane_loss
            else:
                losses["plane_loss/{}".format(scale)] = torch.tensor(0.0).to(self.device)

            # Line regularization loss
            use_line_reg = getattr(self.opt, 'use_line_regularization', True)
            
            if use_line_reg:
                line_loss = get_line_loss(inputs[("line_keysets", 0, scale)], norm_point3D)
                loss += self.opt.line_weight * line_loss
                losses["line_loss/{}".format(scale)] = line_loss
            else:
                losses["line_loss/{}".format(scale)] = torch.tensor(0.0).to(self.device)
            
            losses["loss/{}".format(scale)] = loss
            total_loss += loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """

        depth_pred = outputs[("depth_ori", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [self.dataset.full_res_shape[1], self.dataset.full_res_shape[0]],
            mode="bilinear", align_corners=False), self.dataset.min_depth, self.dataset.max_depth)
        
        
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, self.dataset.default_crop[2]:self.dataset.default_crop[3], \
        self.dataset.default_crop[0]:self.dataset.default_crop[1]] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=self.dataset.min_depth, max=self.dataset.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, outputs_mask=None):
        """Write an event to the tensorboard events file
        """
        if self.debug_no_save or mode not in self.writers:
            return
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        
        # 添加学习率监控（重要：用于诊断280000步崩溃问题）
        current_lr = self.model_optimizer.param_groups[0]['lr']
        writer.add_scalar("learning_rate", current_lr, self.step)
        
        # 添加梯度范数监控（用于检测梯度爆炸）
        if 'grad_norm' in losses:
            writer.add_scalar("grad_norm", losses['grad_norm'].item(), self.step)
        
        # 记录mask训练的可视化（如果启用）
        if outputs_mask is not None and ("mask", 0) in outputs_mask:
            for j in range(min(4, self.opt.batch_size)):
                writer.add_image(
                    "mask_training/mask_{}".format(j),
                    outputs_mask[("mask", 0)][j].data,
                    self.step
                )
                for s in self.opt.scales:
                    writer.add_image(
                        "mask_training/disp_masked_{}/{}".format(s, j),
                        normalize_image(outputs_mask[("disp", s)][j]),
                        self.step
                    )

        for j in range(min(12, self.opt.batch_size)):  # write a maxmimum of four images

            writer.add_image(
                "gt_depth_0/{}".format(j),
                normalize_image(inputs["depth_gt"][j]), self.step)
            writer.add_image(
                "gt_disp_0/{}".format(j),
                normalize_image(1/(inputs["depth_gt"][j]+0.01)), self.step)
                
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                # 学生模型（扩散解码器）的预测视差图
                writer.add_image(
                    "disp_student_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                
                # 教师模型（Depth Anything V3）的预测视差图
                if ("predisp", s) in outputs:
                    writer.add_image(
                        "disp_teacher_{}/{}".format(s, j),
                        normalize_image(outputs[("predisp", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        if self.debug_no_save:
            return
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        if self.debug_no_save:
            return
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # 排除教师模型（它是预训练的冻结模型，不应该被保存到检查点）
        # 教师模型：depth_anything_v3_teacher - Depth Anything V3 教师模型
        teacher_model_keys = ["depth_anything_v3_teacher"]
        
        for model_name, model in self.models.items():
            # 跳过教师模型
            if model_name in teacher_model_keys:
                print(f"→ 跳过教师模型: {model_name}（教师模型不保存到检查点）")
                continue
                
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # 排除教师模型（它是预训练的冻结模型，从各自的源加载，不从检查点加载）
        # 教师模型：depth_anything_v3_teacher - 从 HuggingFace 或本地路径加载
        teacher_model_keys = ["depth_anything_v3_teacher"]

        for n in self.opt.models_to_load:
            # 跳过教师模型
            if n in teacher_model_keys:
                print("→ 跳过教师模型: {} (教师模型从各自的源加载，不从检查点加载)".format(n))
                continue
                
            # 检查模型是否存在
            if n not in self.models:
                print("Warning: Model '{}' not found in self.models, skipping...".format(n))
                continue
                
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if not os.path.exists(path):
                print("Warning: Weight file '{}' not found, skipping...".format(path))
                continue
                
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")



