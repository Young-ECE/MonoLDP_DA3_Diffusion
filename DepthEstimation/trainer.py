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
from PIL import Image


class Trainer:
    def __init__(self, options):
        self.opt = options
        # self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.opt.log_dir, f"{self.opt.model_name}_{timestamp}")
        print("-> Log path: {}".format(self.log_path))
        print("-> Model name: {}".format(self.opt.model_name))
        self.debug_no_save = getattr(self.opt, "debug_no_save", False)
        self.debug_save_teacher = getattr(self.opt, "debug_save_teacher", False)
        if self.debug_no_save:
            print("⚠ Debug mode enabled: training artifacts will not be written to disk.")
        if self.debug_save_teacher:
            print("→ Debug teacher snapshot saving enabled.")

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        print("Using device:", self.device)

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
        
        # 深度解码器初始化
        if self.opt.use_diffusion:
            print("=" * 60)
            print("🔄 使用扩散深度解码器")
            print("=" * 60)
            
            # 1. 创建教师模型（冻结的预训练模型）
            self.models["pre_depth_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["pre_depth_decoder"] = networks.DepthDecoder(
                self.models["pre_depth_encoder"].num_ch_enc, 
                self.opt.scales,
                PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation)
            
            # 2. 加载教师模型权重（如果提供）
            if self.opt.teacher_weights_folder is not None:
                teacher_path = self.opt.teacher_weights_folder
                encoder_path = os.path.join(teacher_path, "encoder.pth")
                decoder_path = os.path.join(teacher_path, "depth.pth")
                
                if os.path.exists(encoder_path) and os.path.exists(decoder_path):
                    print(f"→ 加载教师模型: {teacher_path}")
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
                    print("✓ 教师模型加载成功")
                else:
                    print(f"⚠ 警告: 教师权重未找到于 {teacher_path}")
                    print("→ 教师模型将从头训练")
            else:
                print("⚠ 未指定教师模型路径，教师模型将使用与学生相同的初始化")
            
            # 3. 冻结教师模型
            for param in self.models["pre_depth_encoder"].parameters():
                param.requires_grad = False
            for param in self.models["pre_depth_decoder"].parameters():
                param.requires_grad = False
            
            self.models["pre_depth_encoder"].to(self.device)
            self.models["pre_depth_decoder"].to(self.device)
            self.models["pre_depth_encoder"].eval()
            self.models["pre_depth_decoder"].eval()
            
            # 4. 创建学生模型（带扩散）
            self.models["depth"] = networks.DepthDecoderDiffusion(
                self.models["encoder"].num_ch_enc, 
                self.opt.scales,
                num_output_channels=3,
                PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation)
            print("✓ 扩散解码器初始化成功")
            print("=" * 60)
        else:
            # 不使用扩散，使用原始解码器
            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, 
                self.opt.scales,
                PixelCoorModu = not self.opt.disable_pixel_coordinate_modulation)
        
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers,
                                                             self.opt.weights_init == "pretrained",
                                                             num_input_images=self.num_pose_frames)
        # print("pose_encoder weights init:", self.opt.weights_init)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,
                                                   num_input_features=1,
                                                   num_frames_to_predict_for=(self.num_pose_frames-1))
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        ####add pose_rec
        self.models["pose_rec"] = networks.PoseDecoderRec(self.models["pose_encoder"].num_ch_enc,
                                                   num_input_features=1,
                                                   num_frames_to_predict_for=(self.num_pose_frames-1))
        self.models["pose_rec"].to(self.device)
        self.parameters_to_train += list(self.models["pose_rec"].parameters())

        ####add pose_Third
        self.models["pose_third"] = networks.PoseDecoderThird(self.models["pose_encoder"].num_ch_enc,
                                                   num_input_features=1,
                                                   num_frames_to_predict_for=(self.num_pose_frames-1))
        self.models["pose_third"].to(self.device)
        self.parameters_to_train += list(self.models["pose_third"].parameters())

        # if torch.cuda.device_count() > 1:
        #     print("-> Using {} GPUs for training".format(torch.cuda.device_count()))
        #     for key in self.models.keys():
        #         # Only apply DataParallel to encoder and scalenet
        #         # Pose models have complex dependencies, keep them on single GPU
        #         if key in ["encoder", "scalenet"]:
        #             self.models[key] = nn.DataParallel(self.models[key])
        #     self.device = torch.device("cuda:0")
        #     print("-> Note: encoder and scalenet use multi-GPU; pose and depth models use single GPU")
        # else:
        #     print("-> Using single GPU")

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
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

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=True, img_ext=img_ext,
            return_plane=not self.opt.disable_plane_regularization,
            num_plane_keysets = self.opt.num_plane_keysets,
            return_line=not self.opt.disable_line_regularization,
            num_line_keysets = self.opt.num_line_keysets)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, self.num_scales, is_train=False, img_ext=img_ext,
            return_plane=not self.opt.disable_plane_regularization,
            num_plane_keysets = self.opt.num_plane_keysets,
            return_line=not self.opt.disable_line_regularization,
            num_line_keysets = self.opt.num_line_keysets)

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        if not self.debug_no_save:
            for mode in ["train", "val"]:
                self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
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

        if not self.debug_no_save:
            self.save_opts()

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
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            run_step += 1
            loss_sum += losses["loss"].cpu().data

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, loss_sum/run_step)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
            self.step += 1

        self.model_lr_scheduler.step()

        self.val()


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses."""
        """Pass a minibatch through the network and generate images and losses
        """

        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        norm_pix_coords = [inputs[("norm_pix_coords", s)] for s in self.opt.scales]

        # 如果使用扩散模块
        if self.opt.use_diffusion:
            # 1. 使用教师模型生成伪GT（教师模型已冻结，使用eval模式）
            with torch.no_grad():
                pre_features = self.models["pre_depth_encoder"](inputs[("color_aug", 0, 0)])
                pre_outputs = self.models["pre_depth_decoder"](pre_features, norm_pix_coords)
                if self.debug_save_teacher:
                    self._save_teacher_debug(pre_outputs)
            
            # 2. 学生模型前向传播
            features = self.models["encoder"](inputs[("color_aug", 0, 0)])
            
            # 3. 准备扩散的伪GT（从教师模型的预测）
            gt_for_diffusion = {}
            for scale in self.opt.scales:
                # 使用教师的 disp 作为扩散的目标
                gt_for_diffusion[("disp_diffusion", scale)] = pre_outputs[("disp", scale)].detach()
            
            # 4. 使用扩散解码器
            outputs = self.models["depth"](features, norm_pix_coords, gt_for_diffusion)
            
            # 5. 保存教师的预测，用于后续损失计算和可视化
            for scale in self.opt.scales:
                outputs[("predisp", scale)] = pre_outputs[("disp", scale)]
        else:
            # 不使用扩散，使用原始的深度解码器
            features = self.models["encoder"](inputs[("color_aug", 0, 0)])
            outputs = self.models["depth"](features, norm_pix_coords)

        outputs.update(self.predict_poses_ori(inputs))
        self.generate_images_pred_ori(inputs, outputs)

        outputs.update(self.predict_poses_second(inputs, outputs))
        self.generate_images_pred_second(inputs, outputs)

        outputs.update(self.predict_poses_third(inputs, outputs))
        self.generate_images_pred_third(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)
        
        # 如果使用扩散，添加额外的损失
        if self.opt.use_diffusion:
            # L1损失：学生与教师的一致性
            l1_loss = 0
            for scale in self.opt.scales:
                l1_loss += F.l1_loss(outputs[("predisp", scale)], outputs[("disp", scale)])
            losses['l1'] = l1_loss / len(self.opt.scales)
            
            # DDIM损失：扩散模型的去噪损失
            ddim_loss = 0
            for scale in self.opt.scales:
                if ("ddim_loss", scale) in outputs:
                    ddim_loss += outputs[("ddim_loss", scale)]
            losses['ddim'] = ddim_loss / len(self.opt.scales) if ddim_loss != 0 else torch.tensor(0.0).to(self.device)
            
            # 保存原始光度损失
            losses['photometric'] = losses["loss"].clone()
            
            # 总损失 = 光度损失 + L1损失 + DDIM损失
            losses["loss"] = (1.0 * losses['photometric'] + 
                             self.opt.diffusion_l1_weight * losses['l1'] + 
                             self.opt.diffusion_ddim_weight * losses['ddim'])

        return outputs, losses


    def predict_poses_ori(self, inputs):
        """Predict poses between input frames for monocular sequences.
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
        """Predict poses between input frames for monocular sequences.
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
        """Predict poses between input frames for monocular sequences.
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
            if not self.opt.disable_plane_regularization:
                losses_sum["plane_loss/" + str(s)] = 0.0
                losses_avg["plane_loss/" + str(s)] = 0.0
            if not self.opt.disable_line_regularization:
                losses_sum["line_loss/" + str(s)] = 0.0
                losses_avg["line_loss/" + str(s)] = 0.0
            for frame_id in self.opt.frame_ids[1:]:
                losses_sum["depth_consistency_loss/{}_{}".format(s, frame_id)] = 0.0
                losses_avg["depth_consistency_loss/{}_{}".format(s, frame_id)] = 0.0

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
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        new_pred = pred * 5
        new_target = target * 5

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(new_pred, new_target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss            

    
    
    def compute_depth_consistency_loss(self, depth_t, depth_t_prime):
        """
        Compute the depth consistency loss between two predicted depth maps.

        Args:
            depth_t (torch.Tensor): Predicted depth map of the target image.
            depth_t_prime (torch.Tensor): Predicted depth map of the source image.

        Returns:
            torch.Tensor: Depth consistency loss.
        """
        # Ensure depth tensors are three-dimensional
        if len(depth_t.shape) == 4:
            depth_t = depth_t.squeeze(1)  # Remove the channel dimension if it exists
        if len(depth_t_prime.shape) == 4:
            depth_t_prime = depth_t_prime.squeeze(1)  # Remove the channel dimension if it exists

        # Compute depth consistency loss
        depth_consistency_loss = torch.abs(depth_t - depth_t_prime) / (depth_t + depth_t_prime)
        depth_consistency_loss = torch.mean(depth_consistency_loss)

        return depth_consistency_loss

    

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

            #calculate the multi-reprojection loss
            for frame_id in self.opt.frame_ids[1:]:
                pred_ori = outputs[("color_ori", frame_id, scale)]
                pred = outputs[("color", frame_id, scale)]
                pred_new = outputs[("color_new", frame_id, scale)]
                
                outputs[("reprojection_losses_ori", frame_id, scale)] = self.compute_reprojection_loss(pred_ori, target)
                losses["reprojection_losses_ori/{}_{}".format(frame_id, scale)] = outputs[("reprojection_losses_ori", frame_id, scale)].mean()
                loss += 0.25*outputs[("reprojection_losses_ori", frame_id, scale)].mean()

                outputs[("reprojection_losses_vitual", frame_id, scale)] = self.compute_reprojection_loss(pred, target)
                losses["reprojection_losses_vitual/{}_{}".format(frame_id, scale)] = outputs[("reprojection_losses_vitual", frame_id, scale)].mean()
                loss += outputs[("reprojection_losses_vitual", frame_id, scale)].mean()

                outputs[("reprojection_losses_new", frame_id, scale)] = self.compute_reprojection_loss(pred_new, target)
                losses["reprojection_losses_new/{}_{}".format(frame_id, scale)] = outputs[("reprojection_losses_new", frame_id, scale)].mean()
                loss += outputs[("reprojection_losses_new", frame_id, scale)].mean()


            if self.opt.disable_plane_smoothness:
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)
            else:
                mean_coeff = outputs[("coeff", scale)].abs().mean(2, True).mean(3, True)

                norm_coeff = outputs[("coeff", scale)] / (mean_coeff + 1e-7)
                smooth_loss = get_smooth_loss(norm_coeff, color)

            loss += self.opt.smoothness_weight / (2 ** scale) * smooth_loss
            losses["smooth_loss/{}".format(scale)] = smooth_loss

            point3D = outputs[("cam_points", 0, scale)][:, :3, ...]
            mean_depth = outputs[("depth_ori", 0, scale)].mean(2, True).mean(3)
            norm_point3D = point3D/(mean_depth + 1e-7)

            if not self.opt.disable_plane_regularization:
                plane_loss = get_plane_loss(inputs[("plane_keysets", 0, scale)], norm_point3D)
                loss += self.opt.plane_weight * plane_loss
                losses["plane_loss/{}".format(scale)] = plane_loss

            if not self.opt.disable_line_regularization:
                line_loss = get_line_loss(inputs[("line_keysets", 0, scale)], norm_point3D)
                loss += self.opt.line_weight * line_loss
                losses["line_loss/{}".format(scale)] = line_loss
            
                # calculate the depth consistency loss
            for frame_id in self.opt.frame_ids[1:]:
                depth_ori = outputs[("depth_ori", 0, scale)].squeeze(1)  
                depth_second = outputs[("depth_second", 0, scale)].squeeze(1) 
                depth_third = outputs[("depth_third", 0, scale)].squeeze(1)  
                
                #compute the consistency among synthesized views
                depth_consistency_loss_ori_second = self.compute_depth_consistency_loss(depth_ori, depth_second)
                depth_consistency_loss_ori_third = self.compute_depth_consistency_loss(depth_ori, depth_third)
                depth_consistency_loss_second_third = self.compute_depth_consistency_loss(depth_second, depth_third)

                depth_consistency_loss = (depth_consistency_loss_ori_second +
                                          depth_consistency_loss_ori_third +
                                          depth_consistency_loss_second_third) / 3.0

                loss += self.opt.depth_consistency_weight * depth_consistency_loss
                losses["depth_consistency_loss/{}_{}".format(scale, frame_id)] = depth_consistency_loss
            
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

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        if self.debug_no_save or mode not in self.writers:
            return
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

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

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

    def _save_teacher_debug(self, pre_outputs):
        """Save teacher disparity visualizations for debugging purposes."""
        if not hasattr(self, "_teacher_debug_dir"):
            self._teacher_debug_dir = os.path.join(self.log_path, "debug_teacher_disp")
            os.makedirs(self._teacher_debug_dir, exist_ok=True)

        step = getattr(self, "step", 0)
        with torch.no_grad():
            for scale in self.opt.scales:
                key = ("disp", scale)
                if key not in pre_outputs:
                    continue
                disp = pre_outputs[key]
                if disp is None:
                    continue
                disp = disp.detach().cpu()
                if disp.ndim != 4:
                    continue
                disp_vis = normalize_image(disp)
                batch_to_save = min(2, disp_vis.shape[0])
                for idx in range(batch_to_save):
                    tensor = disp_vis[idx]
                    if tensor.shape[0] > 1:
                        tensor = tensor.mean(0, keepdim=True)
                    array = tensor.squeeze(0).numpy()
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                    save_path = os.path.join(
                        self._teacher_debug_dir,
                        f"step{step:06d}_batch{idx}_scale{scale}.png"
                    )
                    Image.fromarray(array).save(save_path)

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

        for model_name, model in self.models.items():
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

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
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



