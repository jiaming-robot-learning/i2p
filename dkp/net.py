
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
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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
        """
        returns: (N, dim), where N is the number of features and dim is the dimensionality of each feature
        """
        x = th.zeros(1, self.in_channels, self.image_size, self.image_size)
        x = x.type(self.dtype)
        y = self.forward(x)
        
        return 1, y.view(1, -1).shape[1]
    
    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :return: an [N x K] Tensor of outputs.
        """

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h)
           
        return h.view(h.shape[0], 1, -1) # [B, 1, feat_dim], compatible with the transformer input

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
        
        
        
        
################################## Transformers ##################################


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = th.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = th.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, 
                 image_size, 
                 patch_size=16, # 16 | 32
                 dim=768, 
                 depth=8, 
                 heads=8, 
                 channels = 3, 
                 dim_head = 64, 
                 dropout = 0.1, 
                 emb_dropout = 0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # self.time_embedding = SinusoidalPosEmb(dim)
        self.pos_embedding = nn.Parameter(th.randn(1, num_patches, dim)) 
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*4, dropout)


    def forward(self, img):
        x = self.to_patch_embedding(img)
        # t = self.time_embedding(timestep)
        # x = th.cat([t, x], dim=1)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        return x
        
class PatchConv(nn.Module):
    """
    Patchify the input map, and then apply conv layers to each patch
    """
    def __init__(self,
                 image_size,
                 patch_size=16, # 16 | 32
                 in_channels=1,
                 model_channels=4,
                 num_layers=4, 
                 patch_dim=768,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        
        self.h = image_size // patch_size
        self.w = image_size // patch_size
        
        self.conv_layers = nn.ModuleList([])
        self.conv_layers.append(nn.Sequential(
            nn.Conv2d(in_channels, model_channels, 3, padding=1),
            nn.GroupNorm(2, model_channels),
            nn.SiLU(),
        ))
        for _ in range(num_layers-1):
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(model_channels, model_channels, 3, padding=1),
                nn.GroupNorm(2, model_channels),
                nn.SiLU(),
            ))
        
        self.patch_emb = nn.Linear(patch_size**2*model_channels, patch_dim)
        self.pos_emb = nn.Parameter(th.zeros(1, self.h*self.w, patch_dim))
        
    def get_feat_dim(self):
        """
        returns: (N, dim), where N is the number of features and dim is the dimensionality of each feature
        """
        x = th.zeros(1, self.in_channels, self.image_size, self.image_size)
        y = self.forward(x)
        
        return y.shape[1], y.shape[2]
     
    def forward(self, x):
        """
        x: [B, C, H, W]
        
        """
        
        patches = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = self.patch_size, p2 = self.patch_size) # [BHW, C, p1, p2]
      
        for conv_layer in self.conv_layers:
            patches = conv_layer(patches)
        
        patches = rearrange(patches, '(b h w) c p1 p2 -> b (h w) (c p1 p2)', 
                            h = self.h, w = self.w, p1 = self.patch_size, p2 = self.patch_size) 
        
        patches = self.patch_emb(patches) # [B, HW, patch_dim]
        patches = patches + self.pos_emb # [B, HW, patch_dim]
        
        return patches # [B, HW, C*p1*p2]
    
class DenoiseTransformer(nn.Module):
    """
    Transformer with encoder and decoder
    """
    
    def __init__(self, 
                 map_feat_dim, 
                 map_feat_n_tokens=1,
                 emb_dim=768,
                 n_head=12,
                 dropout=0.1,
                 n_layers=6,
                 ) -> None:
        """
        @param map_feat_dim: visual condition embedding dimension
        @param emb_dim: token embedding dimension
        
        """
        super().__init__()


        self.cond_n = 1 # 1 for timestep

        # map feat embedding        
        # self.map_emb = nn.Linear(map_feat_dim, emb_dim)

        # timestep embedding
        self.timestep_emb = SinusoidalPosEmb(emb_dim)
    
        self.pos_embedding = nn.Parameter(th.zeros(1, self.cond_n, emb_dim)) # learnable positional embedding for map and timestep
        
        # input embedding
        self.kp_emb = nn.Linear(2, emb_dim) # 2 for kp
        
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
        
        # print model params
        total_params = sum(p.numel() for p in self.parameters())
        timestep_emb_params = sum(p.numel() for p in self.timestep_emb.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print('Denoise transformer total params: {},  timestep_emb_params: {}, decoder_params: {}'.format(
            total_params, timestep_emb_params, decoder_params)
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
        @param map_feat: [B, N', emb_dim]
        @param timesteps: [B]
        """
        # map_emb = self.map_emb(map_feat) # [B, N', emb_dim]
        map_emb = map_feat
        te = self.timestep_emb(timesteps).unsqueeze(1) # [B, 1, emb_dim]
        te = te + self.pos_embedding # [B, 1, emb_dim]
        
        memory = th.concat([te, map_emb], dim=1) # [B, 1+N', emb_dim]
        # memory = memory + self.pos_embedding # [B, 1+N', emb_dim]
        
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
    def __init__(self, img_size,use_patch_encoder=True, use_line_pred=True) -> None:
        super().__init__()
        if not use_patch_encoder:
            self.vis_encoder = VisualEncoder(image_size=img_size)
        else:
            self.vis_encoder = PatchConv(image_size=img_size)
            # self.vis_encoder = ViT(image_size=img_size)
            
        if use_line_pred:
            self.line_pred = LineClassifier()
            
        self.use_line_pred = use_line_pred
        
        self.map_feat_n_tokens, self.map_feat_dim = self.vis_encoder.get_feat_dim()
        
        self.denoise = DenoiseTransformer(
            map_feat_dim=self.map_feat_dim,
            map_feat_n_tokens=self.map_feat_n_tokens,
        )
        
    def forward(self, tra_map, noisy_kp, timesteps):
        cond = self.vis_encoder(tra_map)
        noise_pred = self.denoise(noisy_kp, cond, timesteps)
        return noise_pred
    
    def get_line_pred(self,lines, memory):
        """
        args: lines [B, N, 4]
                memory [B, N', emb_dim]
        """
        assert self.use_line_pred
        return self.line_pred(lines, memory)
    
    def get_feat_dim(self):
        return self.map_feat_n_tokens, self.map_feat_dim
    
    def get_vis_feat(self, tra_map):
        return self.vis_encoder(tra_map) # [B, N|1, feat_dim]
    
    def get_noise_pred(self, kp, cond, timesteps):
        return self.denoise(kp, cond, timesteps)

        
        
##########################################################################################
# Line classifier
##########################################################################################

class LineClassifier(nn.Module):
    def __init__(self,
                emb_dim=768,
                n_head=12,
                dropout=0.1,
                n_layers=3,
                 ) -> None:
        super().__init__()
        
        # self.map_encoder = PatchConv(image_size=image_size)

        # input embedding
        self.l_emb = nn.Linear(4, emb_dim) # 2 for [x1, y1, x2, y2]
        
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
        self.head = nn.Linear(emb_dim, 1) # 1 for line classification
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,lines, memory):
        """
        @param lines [B, N, 4]
        @param mem [B, N', emb_dim]
        
        """
        
        memory = self.dropout(memory)
        
        x = self.l_emb(lines) # [B, N, emb_dim]
        x = self.dropout(x)
        
        y = self.decoder(
            tgt = x, # [B, N, embdim]
            memory = memory, # [B, N', emb_dim]
        )
        
        y = self.ln_f(y)
        y = self.head(y) # [B, N, 1]
        
        return y