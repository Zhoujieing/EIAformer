__all__ = ['EIAformer_backbone']

# Cell

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor

import torch.nn.functional as F
import numpy as np

import pandas as pd


from statsmodels.tsa.stattools import grangercausalitytests
import random
from keras.preprocessing.sequence import pad_sequences

#from collections import OrderedDict
from layers.EIAformer_layers import *
from layers.RevIN import RevIN
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention, get_frequency_modes
from layers.Autoformer_EncDec import EncoderLayer
from layers.AutoCorrelation import AutoCorrelationLayer
from  layers.SelfAttention_Family import ProbAttention

import statsmodels.api as sm
from fastdtw import fastdtw

# Cell

class EIAformer_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, 
                 max_seq_len:Optional[int]=1024, n_layers:int=3, d_model=128, n_heads=16, 
                 d_k:Optional[int]=None, d_v:Optional[int]=None, d_ff=256, norm='BatchNorm', 
                 attn_dropout=0., dropout=0., act="gelu", key_padding_mask='auto', 
                 padding_var=None, attn_mask=None, res_attention=True, pre_norm=False, 
                 store_attn=False, pe='zeros', learn_pe=True, fc_dropout=0., head_dropout=0, 
                 padding_patch='end', pretrain_head=True, head_type='flatten', individual=False, 
                 revin=True, affine=True, subtract_last=True, verbose=True, **kwargs):
        
        super().__init__()
        
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch #
        patch_num = int(context_window  /patch_len)
        if padding_patch == 'end': 

            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) 
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        self.W = nn.Parameter(torch.randn(c_in, d_model))  
        self.U = nn.Parameter(torch.randn(d_model)) 
        self.V = nn.Parameter(torch.randn(c_in)) 


    def forward(self, x):
        # 初始分块
        patches = self.initial_patching(x)
        
        # 动态分块
        selection_column = []
        for i in range(len(patches) - 1):
            score = self.combine_condition(patches[i], patches[i + 1])
            if score >= 0.5:
                # 合并块
                combined_patch = self.merge_patches(patches[i], patches[i + 1])
                selection_column.append(combined_patch)
            else:
                selection_column.append(patches[i])

        # 处理选择列
        final_patches = self.apply_patching_strategy(selection_column)
        return final_patches

    def initial_patching(self, x):
        return x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

    def combine_condition(self, Xi, Xi_plus_1):
        combined_data = torch.cat((Xi, Xi_plus_1), dim=0)
        x_mean = combined_data.mean()
        
        x = torch.arange(len(combined_data)).float()
        A = torch.vstack([x, torch.ones(len(x))]).T
        a, b = torch.linalg.lstsq(A, combined_data, rcond=None)[0]

        x_hat = a * x + b
        score = 1 - (torch.sum((combined_data - x_hat) ** 2) / torch.sum((combined_data - x_mean) ** 2))
        
        return score.item()

    def merge_patches(self, Xi, Xi_plus_1):
        combined_patch = (Xi + Xi_plus_1) / 2
        return combined_patch

    def apply_patching_strategy(self, selection_column):
        
        if len(selection_column) == 0:
            return None  

       
        final_output = torch.stack(selection_column)  

       
        batch_size = final_output.size(0)  
        nvars = final_output.size(1)  
        patch_len = final_output.size(2) 

        seq_len = batch_size * patch_len  

        
        final_output = final_output.view(batch_size, nvars, -1)  

        

        return final_output
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )

    def channel_fusion(self, channels):
        fused_channels = []
        
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                IC = self.intrinsic_connection(channels[i], channels[j])
                EC = self.extrinsic_connection(channels[i], channels[j])
                
                if IC == 1 and EC == 1:
                    # 计算权重系数
                    alpha = channels[i].std() / (channels[i].std() + channels[j].std())
                    beta = channels[j].std() / (channels[i].std() + channels[j].std())
                    
                    # 融合通道
                    fused_channel = alpha * channels[i] + beta * channels[j]
                    fused_channels.append(fused_channel)

        return torch.stack(fused_channels)  

    def intrinsic_connection(self, Xi, Xj):
        
        max_lag = 1  
        test_result = sm.tsa.stattools.grangercausalitytests(
            np.column_stack((Xi.cpu().numpy(), Xj.cpu().numpy())), 
            max_lag, 
            verbose=False
        )
        
        
        for lag in range(1, max_lag + 1):
            p_value = test_result[lag][0]['ssr_ftest'][1]  
            if p_value < 0.05:  
                return 1 
        return 0  

    def extrinsic_connection(self, Xi, Xj):
        distance, _ = fastdtw(Xi.cpu().numpy(), Xj.cpu().numpy())
        threshold = 1.0  
        return 1 if distance < threshold else 0


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

# 改输出

    def DTWDistance(s1, s2):
        DTW = {}

        for i in range(len(s1)):
            DTW[(i, -1)] = float('inf')
        for i in range(len(s2)):
            DTW[(-1, i)] = float('inf')
        DTW[(-1, -1)] = 0

        for i in range(len(s1)):
            for j in range(len(s2)):
                dist = (s1[i] - s2[j]) ** 2
                DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

        return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

    def forward(self, x):
        if self.individual:
            x_out = []
            

            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)

            x = torch.stack(x_out, dim=1)  # x: [bs x (nvars/2) x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)

        return x
        
        
    
    
class TSTiEncoder(nn.Module):  
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        

        x = self.W_P(x)                           # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        y = self.W_pos

        u = self.dropout(u + y)                                         # u: [bs * nvars x patch_num x d_model]

        
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]
        
        return z    
            
            
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



# 普通Encoder
class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        
        self.res_attention = res_attention
        
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        
        if self.pre_norm:
            src = self.norm_attn(src)
        
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        
        src = src + self.dropout_attn(src2) 
        if not self.pre_norm:
            src = self.norm_attn(src)

        
        if self.pre_norm:
            src = self.norm_ffn(src)
        
        src2 = self.ff(src)
        
        src = src + self.dropout_ffn(src2) 
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

       
        self.res_attention = res_attention
        
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        


        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0.05, res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout =attn_dropout
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def get_frequency_modes(seq_len = 96, modes=64, mode_select_method='random'):
        """
        get modes on frequency domain:
        'random' means sampling randomly;
        'else' means sampling the lowest modes;
        """
        modes = min(modes, seq_len // 2)
        if mode_select_method == 'random':
            index = list(range(0, seq_len // 2))
            np.random.shuffle(index)
            index = index[:modes]
        else:
            index = list(range(0, modes))
        index.sort()
        return index

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        

        attn_scores = torch.matmul(q, k) * self.scale     
        if prev is not None: attn_scores = attn_scores + prev

        
        if attn_mask is not None:                    
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

