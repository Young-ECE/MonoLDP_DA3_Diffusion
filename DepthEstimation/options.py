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
                                 choices=["nyu","sequences"],
                                 default="nyu")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="nyu",
                                 choices=["nyu","mydata"])
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
                                 default=6)
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
                                 default=20)

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

        # hyper-parameter
        self.parser.add_argument("--smoothness_weight",
                                 type=float,
                                 help="smoothness weight",
                                 default=0.2)
        self.parser.add_argument("--num_plane_keysets",
                                 type=int,
                                 help="the number of keysets for plane regularization",
                                 default=512)
        self.parser.add_argument("--plane_weight",
                                 type=float,
                                 help="plane regularization weight",
                                 default=2.0)
        self.parser.add_argument("--num_line_keysets",
                                 type=int,
                                 help="the number of keysets for line regularization",
                                 default=128)
        self.parser.add_argument("--line_weight",
                                 type=float,
                                 help="line regularization weight",
                                 default=0.5)
        self.parser.add_argument('--depth_consistency_weight', 
                             type=float, 
                             default=1.0, 
                             help='Weight for depth consistency loss')

        # DIFFUSION options
        self.parser.add_argument("--use_diffusion",
                                help="if set, use diffusion-based depth decoder",
                                action="store_true",
                                default=True)
        self.parser.add_argument("--teacher_weights_folder",
                                type=str,
                                help="path to pretrained teacher model weights",
                                default="/oldisk/home/jingyang/monoldp/temp/mdp_20251106_205450/models/weights_14")
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
        self.parser.add_argument("--diffusion_l1_weight",
                                type=float,
                                default=1.0,
                                help="weight for L1 loss between teacher and student")
        self.parser.add_argument("--diffusion_ddim_weight",
                                type=float,
                                default=1.0,
                                help="weight for DDIM diffusion loss")

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
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
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
                                 default=True)
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
