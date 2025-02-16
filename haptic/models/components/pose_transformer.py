import numpy as np
from inspect import isfunction
from typing import Callable, Optional
from xformers.ops import memory_efficient_attention

from omegaconf import ListConfig
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from .t_cond_mlp import (
    AdaptiveLayerNorm1D,
    FrequencyEmbedder,
    normalization_layer,
)


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn


    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def zero_init(self, zero_qkv=False):
        if zero_qkv:
            self.to_kv.weight.data.zero_()
            self.to_kv.bias.data.zero_()
            self.to_q.weight.data.zero_()
            self.to_q.bias.data.zero_()
        self.to_out[0].weight.data.zero_()
        self.to_out[0].bias.data.zero_()


    def forward(self, x, context=None, attn_mask=None): 

        """    
        attn_mask: (B, 1, T)
        attn_mask = torch.ones(L:query, S:key, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
        attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p)
        return attn_weight @ V
        """
        import torch.nn.functional as F
        F.scaled_dot_product_attention

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(~attn_mask, -float('inf'))

        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        # MEA expects [B, N, H, D], whereas timm uses [B, H, N, D]
        out = memory_efficient_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=self.scale,
            attn_bias=attn_mask,
        )
        out = out.transpose(1, 2)


        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args):
        for attn, ff in self.layers:
            x = attn(x, *args) + x
            x = ff(x, *args) + x
        return x


class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})")

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # Zero-out the masked tokens
            x[zero_mask, :] = 0
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        emb_dropout_loc: str = "token",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        token_pe_numfreq: int = -1,
    ):
        super().__init__()
        if token_pe_numfreq > 0:
            token_dim_new = token_dim * (2 * token_pe_numfreq + 1)
            self.to_token_embedding = nn.Sequential(
                Rearrange("b n d -> (b n) d", n=num_tokens, d=token_dim),
                FrequencyEmbedder(token_pe_numfreq, token_pe_numfreq - 1),
                Rearrange("(b n) d -> b n d", n=num_tokens, d=token_dim_new),
                nn.Linear(token_dim_new, dim),
            )
        else:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        else:
            raise ValueError(f"Unknown emb_dropout_type: {emb_dropout_type}")
        self.emb_dropout_loc = emb_dropout_loc

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, norm=norm, norm_cond_dim=norm_cond_dim
        )

    def forward(self, inp: torch.Tensor, *args, **kwargs):
        x = inp

        if self.emb_dropout_loc == "input":
            x = self.dropout(x)
        x = self.to_token_embedding(x)

        if self.emb_dropout_loc == "token":
            x = self.dropout(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        if self.emb_dropout_loc == "token_afterpos":
            x = self.dropout(x)
        x = self.transformer(x, *args)
        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim = None,
        skip_token_embedding: bool = False,
        aux_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        if isinstance(context_dim, int):
            self.extra_xattn = False
            self.transformer = TransformerCrossAttn(
                dim,
                depth,
                heads,
                dim_head,
                mlp_dim,
                dropout,
                norm=norm,
                norm_cond_dim=norm_cond_dim,
                context_dim=context_dim,
            )
        elif isinstance(context_dim, list) or isinstance(context_dim, ListConfig):
            self.extra_xattn = True
            self.proj_in_context = nn.ModuleList(nn.Linear(context_dim[i], aux_dim) for i in range(1, len(context_dim)))
            self.pos_enc_f = PositionalEncoding(30, aux_dim)
            self.transformer = TransformerCrossAttnList(
                dim,
                depth,
                heads,
                dim_head,
                mlp_dim,
                dropout,
                norm=norm,
                norm_cond_dim=norm_cond_dim,
                context_dim=[context_dim[0], ] + [aux_dim for _ in range(1, len(context_dim))],
            )
        else:
            raise ValueError(f"Unknown context_dim type: {type(context_dim)}")

    def forward(self, inp: torch.Tensor, *args, context=None, context_list=None):
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        if self.extra_xattn:
            for i in range(1, len(context)):
                context[i] = self.proj_in_context[i-1](context[i])
                context[i] = self.pos_enc_f(context[i])
        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x


class TransformerDecoderVid(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim = None,
        global_context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
        aux_dim: int = 64,
        iid_view=False,
        cfg={},
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.pos_enc_f = PositionalEncoding(30, aux_dim)
        self.transformer = TransformerCrossAttnVid(
                dim,
                depth,
                heads,
                dim_head,
                mlp_dim,
                dropout,
                norm=norm,
                norm_cond_dim=norm_cond_dim,
                context_dim=context_dim,
                global_context_dim=global_context_dim,
                iid_view=iid_view,
                cfg=cfg,
        )


    def forward(self, inp: torch.Tensor, *args, context=None, temporal_context=None, num_frames=8):
        """
        :param inp: (B, F, 1)
        :param context: (BF, HW, C)
        :param temporal_context: (BF, HW, C)
        :param num_frames: F, defaults to 8
        :return: (B, F, D)
        """

        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(x, *args, context=context, temporal_context=temporal_context, num_frames=num_frames)
        return x
    

class TransformerCrossAttnList(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ca_list = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

            cross_attn_list = nn.ModuleList([])
            for i in range(1, len(context_dim)):
                ca = CrossAttention(
                    dim, context_dim=context_dim[i], heads=heads, dim_head=dim_head, dropout=dropout
                )
                ca.zero_init()
                cross_attn_list.append(PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim))
            self.ca_list.append(cross_attn_list)

            ca = CrossAttention(
                dim, context_dim=context_dim[0], heads=heads, dim_head=dim_head, dropout=dropout
            )
            ca = PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim)
                
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        ca,
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )
        self.zeros_xattn()

    def get_new_named_params(self, ):
        new_params = {}
        for i, cross_attn_list in enumerate(self.ca_list):
            for j, cross_attn in enumerate(cross_attn_list):
                assert isinstance(cross_attn, nn.Module)
                new_params.update(cross_attn.named_parameters())
        return new_params                

    def zeros_xattn(self, skip_first=True, zero_qkv=False):
        for i, cross_attn_list in enumerate(self.ca_list):
            for j, cross_attn in enumerate(cross_attn_list):
                if zero_qkv:
                    cross_attn.fn.to_kv.weight.data.zero_()
                    cross_attn.fn.to_kv.bias.data.zero_()
                    cross_attn.tfn.o_q.weight.data.zero_()
                    cross_attn.fn.to_q.bias.data.zero_()
                cross_attn.fn.to_out[0].weight.data.zero_()
                cross_attn.fn.to_out[0].bias.data.zero_()
        return 

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None, mask=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})")

        for i, (self_attn,  cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i][0]) + x
            for c, ct in enumerate(context_list[i][1:]):
                x = self.ca_list[i][c](x, *args, context=ct, attn_mask=mask) + x
            # for c, cross_attn in enumerate(self.ca_list[i]):
            #     x = cross_attn(x, *args, context=context_list[i][c+1], attn_mask=mask) + x
            x = ff(x, *args) + x
        return x




class TransformerCrossAttnVid(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        global_context_dim: Optional[int] = None,
        iid_view=False,
        cfg={}
    ):
        super().__init__()
        if global_context_dim is None:
            global_context_dim = context_dim
        self.cfg = cfg
        self.iid_view = iid_view
        self.layers = nn.ModuleList([])
        self.cross_view_layers = nn.ModuleList([])
        self.global_layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)

            self.cross_view_layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        # PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim, context_dim=global_context_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.global_layers.append(
                nn.ModuleList(
                    [
                        # PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        # PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )   
            )
        self.pos_enc_f = PositionalEncoding(30, dim)
        self.zero_init()

    def zero_init(self, skip_first=True):
        for i, xview in enumerate(self.cross_view_layers):
            xview[0].fn.to_out[0].weight.data.zero_()
            xview[0].fn.to_out[0].bias.data.zero_()
        for i, global_layer in enumerate(self.global_layers):
            global_layer[0].fn.to_out[0].weight.data.zero_()
            global_layer[0].fn.to_out[0].bias.data.zero_()
        return 
    

    def forward(self, x: torch.Tensor, *args, context=None, temporal_context=None, num_frames=1):
        """
        :param x: (B, F, D)
        :param context: (BF, HW, C)
        : param num_frames: F, defaults to 1
        :raises ValueError: _description_
        :return: _description_
        """
        T = num_frames
        BF = x.shape[0]
        B = BF // T

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = rearrange(x, 'b f d -> (b f) 1 d')
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context) + x

            x = self.cross_view_forward(x, i, T, *args)
            x = self.global_forward(x, i, temporal_context, *args)

            x = ff(x, *args) + x  # (BF, 1, D)
        return x

    def cross_view_forward(self, x, i, T, *args):
        cross_view = self.cross_view_layers[i]
        x = rearrange(x, '(b f) 1 d -> b f d', f=T)
        if self.iid_view:
            xf = x                
        else:
            xf = self.pos_enc_f(x)
        x = cross_view[0](xf, *args) + x
        return x
    
    def global_forward(self, x, i, temporal_context, *args):
        global_layer = self.global_layers[i]
        x = rearrange(x, 'b f d -> (b f) 1 d')
        x = global_layer[0](x, *args, context=temporal_context) + x
        return x

    


class PositionalEncoding(nn.Module):
    """use 1D encoding"""

    def __init__(self, max_length, hidden_size):
        super().__init__()
        self.register_buffer(
            "pos_table", torch.FloatTensor(get_1d_sincos_pos_embed(hidden_size, max_length)).unsqueeze(0)
        )

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)]


def get_1d_sincos_pos_embed(embed_dim, size, extra_tokens=0):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = np.arange(size, dtype=np.float32)  # (M,)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if extra_tokens > 0:
        emb = np.concatenate([np.zeros([extra_tokens, embed_dim]), emb], axis=0)
    return emb

