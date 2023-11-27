# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import pickle
import torch

from guided_diffusion.script_util import create_model
from argparse import Namespace
from . import util
from .ckpt_util import (
    I2SB_IMG256_UNCOND_PKL,
    I2SB_IMG256_UNCOND_CKPT,
    I2SB_IMG256_COND_PKL,
    I2SB_IMG256_COND_CKPT,
)

# from ipdb import set_trace as debug

def create_argparser():
    return Namespace(
        attention_resolutions='16,8',
        # batch_size=4,
        channel_mult='',
        class_cond=False,
        clip_denoised=True,
        # diffusion_steps=500,
        dropout=0.0,
        image_size=192,
        learn_sigma=True,
        # adm_ckpt='256x256_diffusion_uncond.pt',
        noise_schedule='linear',
        num_channels=256,
        num_head_channels=64,
        num_heads=4,
        num_heads_upsample=-1,
        num_res_blocks=2,
        num_samples=4,
        predict_xstart=False,
        resblock_updown=True,
        rescale_learned_sigmas=False,
        rescale_timesteps=False,
        timestep_respacing='250',
        use_checkpoint=False,
        use_ddim=False,
        use_fp16=True,
        use_kl=False,
        use_new_attention_order=False,
        use_scale_shift_norm=True,
        in_channels=1,
        out_channels=1,
    )
class Image256Net(torch.nn.Module):
    def __init__(self, log, noise_levels, use_fp16=False, ckpt=None):
        super(Image256Net, self).__init__()

        # initialize model
        # ckpt_pkl = os.path.join(ckpt_dir, I2SB_IMG256_COND_PKL if cond else I2SB_IMG256_UNCOND_PKL)
        # with open(ckpt_pkl, "rb") as f:
        #     kwargs = pickle.load(f)
        if ckpt is None:
            kwargs = create_argparser().__dict__
        else:
            with open(ckpt+'net.pkl', "rb") as f:
                kwargs = pickle.load(f)
        kwargs["use_fp16"] = use_fp16
        self.diffusion_model = create_model(**kwargs)
        log.info(f"[Net] Initialized network from {ckpt}! Size={util.count_parameters(self.diffusion_model)}!")


        self.diffusion_model.eval()
        self.noise_levels = noise_levels

    def forward(self, x, steps, cond=None):

        t = self.noise_levels[steps].detach()
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        return self.diffusion_model(x, t)
