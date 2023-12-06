
from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

      
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        h = self.out_layers(h)
        
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)




class VisualEncoder(nn.Module):
    """
    Encode the input map
    """

    def __init__(
        self,
        image_size=192,
        in_channels=1,
        model_channels=8,
        num_res_blocks=2,
        dropout=0,
        channel_mult=(1, 2, 4, 8, 16),
        conv_resample=True,
        dims=2,
        resblock_updown=True,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = th.float32

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [nn.Sequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            out_ch = ch
            self.input_blocks.append(
                nn.Sequential(
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(
                        ch, conv_resample, dims=dims, out_channels=out_ch
                    )
                )
            )
            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            self._feature_size += ch

        self._feature_size += ch

        print("Visual encoder total params: {}".format(
            sum(p.numel() for p in self.parameters() if p.requires_grad))
        )
            

    def get_feat_dim(self):
        x = th.zeros(1, self.in_channels, self.image_size, self.image_size)
        x = x.type(self.dtype)
        y = self.forward(x)
        
        return y.view(1, -1).shape[1]
    
    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :return: an [N x K] Tensor of outputs.
        """

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
           
        # h = h.type(x.dtype)
        # return self.out(h)
        return h.view(h.shape[0], -1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class MLPNoisePredNet(nn.Module):
    
    def __init__(self, cond_dim, time_step_emb_dim=64, hidden_dim=256) -> None:
        super().__init__()
        
        tsed = time_step_emb_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(tsed),
            nn.Linear(tsed, tsed * 4),
            nn.SiLU(),
            nn.Linear(tsed * 4, tsed),
        )

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim + tsed + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

        
    def forward(self, noisy_kp, cond, timesteps):
        time_emb = self.diffusion_step_encoder(timesteps)
        
        x = th.cat([noisy_kp, cond, time_emb], dim=1)

        return self.mlp(x)


class MLPDenoise(nn.Module):
    def __init__(self, img_size) -> None:
        super().__init__()
        self.vis_encoder = VisualEncoder(image_size=img_size)
        self.map_feat_dim = self.vis_encoder.get_feat_dim()
        self.noise_pred = MLPNoisePredNet(self.map_feat_dim)
        
    def forward(self, tra_map, noisy_kp, timesteps):
        cond = self.vis_encoder(tra_map)
        noise_pred = self.noise_pred(noisy_kp, cond, timesteps)
        return noise_pred
    
    def get_feat_dim(self):
        return self.map_feat_dim
    
    def get_vis_feat(self, tra_map):
        return self.vis_encoder(tra_map)
    
    def get_noise_pred(self, kp, cond, timesteps):
        return self.noise_pred(kp, cond, timesteps)
        
        

class DenoiseTransformer(nn.Module):
    
    def __init__(self, 
                 map_feat_dim, 
                 emb_dim=768,
                 timestep_emb_dim=64,
                 n_head=12,
                 dropout=0.1,
                 n_layers=8,
                 ) -> None:
        """
        @param map_feat_dim: visual condition embedding dimension
        @param emb_dim: token embedding dimension
        
        """
        super().__init__()

        # # learnable positional embedding for map and timestep
        # self.cond_pos_emb = nn.Parameter(th.zeros(1, 2 , emb_dim)) 
        
        # map feat embedding        
        self.map_emb = nn.Sequential(
            nn.Linear(map_feat_dim, emb_dim),
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # timestep embedding
        self.timestep_emb = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, emb_dim),
        )
        
        # input embedding
        self.kp_emb = nn.Linear(2, emb_dim)
        
        # decoder
        decoder_layer = nn.TransformerDecoderLayer(
                        d_model=emb_dim, 
                        nhead=n_head,
                        dim_feedforward=emb_dim * 4,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation='gelu',
                        )
        self.decoder = nn.TransformerDecoder(
                        decoder_layer=decoder_layer,
                        num_layers=n_layers,
                        norm=nn.LayerNorm(emb_dim),
                        )
        
        # decoder head
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, 2) # 2 for kp
        
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        map_emb_params = sum(p.numel() for p in self.map_emb.parameters())
        timestep_emb_params = sum(p.numel() for p in self.timestep_emb.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print('Denoise transformer total params: {}, map_emb_params: {}, timestep_emb_params: {}, decoder_params: {}'.format(
            total_params, map_emb_params, timestep_emb_params, decoder_params)
        )
        
    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.SiLU,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            th.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                th.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    th.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    th.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            th.nn.init.zeros_(module.bias)
            th.nn.init.ones_(module.weight)
        elif isinstance(module, DenoiseTransformer):
            pass
            # if module.cond_obs_emb is not None:
            #     th.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

            
    def forward(self,noisy_kp, map_feat, timesteps):
        """
        @param noisy_kp: [B, N, 2]
        @param map_feat: [B, map_feat_dim]
        @param timesteps: [B, N]
        """
        # map_feat = map_feat + self.cond_pos_emb[:, 0, :] # [B, emb_dim]
        map_emb = self.map_emb(map_feat).unsqueeze(1) # [B, 1, emb_dim]
        te = self.timestep_emb(timesteps).unsqueeze(1) # [B, 1, emb_dim]
        memory = th.concat([map_emb, te], dim=1) # [B, 2, emb_dim]
        memory = self.dropout(memory)
        
        x = self.kp_emb(noisy_kp) # [B, N, emb_dim]
        x = self.dropout(x)
        
        y = self.decoder(
            tgt = x, # [B, N, 2]
            memory = memory, # [B, 2, emb_dim]
        )
        
        y = self.ln_f(y)
        y = self.head(y) # [B, N, 2]
        
        return y
    
class DenoiseTransformerNet(nn.Module):
    def __init__(self, img_size) -> None:
        super().__init__()
        self.vis_encoder = VisualEncoder(image_size=img_size)
        self.map_feat_dim = self.vis_encoder.get_feat_dim()
        self.denoise = DenoiseTransformer(self.map_feat_dim)
        
    def forward(self, tra_map, noisy_kp, timesteps):
        cond = self.vis_encoder(tra_map)
        noise_pred = self.denoise(noisy_kp, cond, timesteps)
        return noise_pred
    
    def get_feat_dim(self):
        return self.map_feat_dim
    
    def get_vis_feat(self, tra_map):
        return self.vis_encoder(tra_map)
    
    def get_noise_pred(self, kp, cond, timesteps):
        return self.denoise(kp, cond, timesteps)