__all__ = ['PatchTST_backbone']
# Cell
from typing import Callable, Optional, Union
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForCausalLM
import warnings

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Cell
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024, 
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:Union[bool, str]='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, use_large_model:bool=False, large_model_name:str="microsoft/DialoGPT-medium",
                 text_prompt_1:str="This is a time series forecasting task. The input contains historical data patterns that need to be analyzed for future predictions.",
                 text_prompt_2:str="The following are the encoded time series features extracted from the PatchTST encoder, which represent the learned temporal patterns.",
                 text_prompt_3:str="The following are the original patch data features that provide additional context for the prediction task.",
                 use_soft_prompt:bool=True, soft_prompt_length:int=10, use_patched_input:bool=True, fusion_type="none",
                 freeze_large_model:bool=True, use_lora:bool=False, lora_r:int=8, lora_alpha:int=16, lora_dropout:float=0.05, llm_num_layers:int=-1, large_model_type="gpt2", **kwargs):
        
        super().__init__()
        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            patch_num += 1
        
        # Backbone 
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, use_large_model=use_large_model, 
                                large_model_name=large_model_name, text_prompt_1=text_prompt_1, large_model_type=large_model_type,
                                text_prompt_2=text_prompt_2, text_prompt_3=text_prompt_3, use_soft_prompt=use_soft_prompt,
                                soft_prompt_length=soft_prompt_length, fusion_type=fusion_type, use_patched_input=use_patched_input,
                                freeze_large_model=freeze_large_model, use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, llm_num_layers=llm_num_layers, **kwargs)

        # Head
        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual
        self.use_patched_input = use_patched_input

        if self.pretrain_head: 
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten': 
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        
        self.freeze_large_model = freeze_large_model
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.llm_num_layers = llm_num_layers
    
    def forward(self, z):                                                                   # z: [bs x nvars x seq_len]
        # norm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
            
        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0,1,3,2)                                                              # z: [bs x nvars x patch_len x patch_num]

        # model
        with torch.cuda.amp.autocast():
            mid = self.backbone(z)                                                                # z: [bs x nvars x d_model x patch_num]
        z = self.head(mid.float())                                                                    # z: [bs x nvars x target_window] 
        
        # denorm
        if self.revin: 
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        return z, mid
    
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


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
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x
        
        
    
    
class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, use_large_model=False, large_model_name="microsoft/DialoGPT-medium",
                 text_prompt_1="This is a time series forecasting task. The input contains historical data patterns that need to be analyzed for future predictions.",
                 text_prompt_2="The following are the encoded time series features extracted from the PatchTST encoder, which represent the learned temporal patterns.",
                 text_prompt_3="The following are the original patch data features that provide additional context for the prediction task.",
                 use_soft_prompt=True, soft_prompt_length=10, use_patched_input=True, fusion_type="none", large_model_type="gpt2",
                 freeze_large_model=True, use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05, llm_num_layers=-1, **kwargs):
        
        
        super().__init__()
        
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.use_large_model = use_large_model
        
        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)        # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        if n_layers % 2 == 0:
            first_half_layers = n_layers // 2
            second_half_layers = n_layers // 2
        else:
            first_half_layers = n_layers // 2 + 1
            second_half_layers = n_layers // 2

        self.encoder_first_half = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, 
                                            attn_dropout=attn_dropout, dropout=dropout,
                                            pre_norm=pre_norm, activation=act, res_attention=res_attention, 
                                            n_layers=first_half_layers, store_attn=store_attn)

        if self.use_large_model:
            self.large_model_integration = LargeModelIntegration(
                d_model=d_model,
                patch_len=patch_len,
                large_model_type=large_model_type,
                text_prompt_1=text_prompt_1,
                text_prompt_2=text_prompt_2,
                text_prompt_3=text_prompt_3,
                use_soft_prompt=use_soft_prompt,
                soft_prompt_length=soft_prompt_length,
                use_patched_input=use_patched_input,
                freeze_large_model=freeze_large_model,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                llm_num_layers=llm_num_layers
            )
        else:
            self.large_model_integration = None

        self.encoder_second_half = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, 
                                             attn_dropout=attn_dropout, dropout=dropout,
                                             pre_norm=pre_norm, activation=act, res_attention=res_attention, 
                                             n_layers=second_half_layers, store_attn=store_attn)
        self.fusion_type = fusion_type
        if fusion_type == "gated":
            self.gate_linear = nn.Linear(d_model * 2, 1)
        elif fusion_type == "attention":
            self.fusion_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        self.use_patched_input = bool(use_patched_input)
        
    def forward(self, x) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]
        
        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]

        z_first = self.encoder_first_half(u)                                     # z_first: [bs * nvars x patch_num x d_model]

        if self.use_large_model and self.large_model_integration is not None:
            original_patch_data = u  # [bs * nvars x patch_num x patch_len]
            z_enhanced = self.large_model_integration(z_first, original_patch_data)                 # z_enhanced: [bs * nvars x patch_num x d_model]
            if self.fusion_type == "none":
                z_fused = z_enhanced
            elif self.fusion_type == "gated":
                gate = torch.sigmoid(self.gate_linear(torch.cat([z_enhanced, z_first], dim=-1)))
                z_fused = gate * z_enhanced + (1 - gate) * z_first
            elif self.fusion_type == "attention":
                z_fused, _ = self.fusion_attn(z_enhanced, z_first, z_first)
            else:
                raise ValueError(f"Unknown fusion_type: {self.fusion_type}")
        else:
            z_fused = z_first # z_fused: [bs * nvars x patch_num x d_model]

        z = self.encoder_second_half(z_fused)                                 # z: [bs * nvars x patch_num x d_model]
        
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]        
        
        return z




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



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
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

        # Scaled Dot-Product Attention (multiple heads)
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

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

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

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class LargeModelIntegration(nn.Module):
    def __init__(self, d_model, patch_len, large_model_type="gpt2", large_model_name=None, 
                 text_prompt_1="This is a time series forecasting task. The input contains historical data patterns that need to be analyzed for future predictions.",
                 text_prompt_2="The following are the encoded time series features extracted from the PatchTST encoder, which represent the learned temporal patterns.",
                 text_prompt_3="The following are the original patch data features that provide additional context for the prediction task.",
                 use_soft_prompt=True, soft_prompt_length=10, use_patched_input=True,
                 freeze_large_model=True, use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05, llm_num_layers=-1):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.large_model_type = large_model_type
        self.text_prompt_1 = text_prompt_1
        self.text_prompt_2 = text_prompt_2
        self.text_prompt_3 = text_prompt_3
        self.use_soft_prompt = use_soft_prompt
        self.soft_prompt_length = soft_prompt_length
        self.use_patched_input = use_patched_input
        self.freeze_large_model = freeze_large_model
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.llm_num_layers = llm_num_layers


        if large_model_name is not None:
            model_path = large_model_name
        elif large_model_type == "gpt2":
            model_path = "/media/ubuntu/data/share/gpt2/"
        elif large_model_type == "qwen3-1.7b":
            model_path = "/media/ubuntu/data/share/Qwen3-1.7B/"
        elif large_model_type == "llama3-8b":
            model_path = "/media/ubuntu/data/share/Llama-3.1-8B/"
        elif large_model_type == "qwen3-8b":
            model_path =  "/media/ubuntu/data/share/Qwen3-8B/"
        elif large_model_type == "qwen2.5-3b":
            model_path =  "/media/ubuntu/data/share/Qwen2.5-3B/"
        elif large_model_type == "llama3.2-1b":
            model_path = "/media/ubuntu/data/share/Llama-3.2-1B/"
        else:
            raise ValueError(f"Unknown large_model_type: {large_model_type}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.large_model_config = AutoConfig.from_pretrained(model_path)
        if llm_num_layers > 0:
            if "gpt" in model_path.lower():
                self.large_model_config.n_layer = llm_num_layers
            else:
                self.large_model_config.num_hidden_layers = llm_num_layers
        self.large_model = AutoModelForCausalLM.from_pretrained(model_path, config=self.large_model_config, device_map="auto")
        self.large_model_dim = self.large_model.config.hidden_size
        
        self.text_tokens_1 = self.tokenizer(text_prompt_1, return_tensors="pt", padding=False, truncation=True)
        self.text_tokens_2 = self.tokenizer(text_prompt_2, return_tensors="pt", padding=False, truncation=True)
        self.text_tokens_3 = self.tokenizer(text_prompt_3, return_tensors="pt", padding=False, truncation=True)
        
        self.projection_to_large = nn.Linear(d_model, self.large_model_dim)
        
        self.projection_patch_to_large = nn.Linear(d_model, self.large_model_dim)
        
        self.projection_from_large = nn.Linear(self.large_model_dim, d_model)
        
        if self.use_soft_prompt:
            self.soft_prompts = nn.Parameter(torch.randn(1, soft_prompt_length, self.large_model_dim) * 0.01)
        
        self.fusion_layer = nn.MultiheadAttention(self.large_model_dim, num_heads=8, batch_first=True)
        
        if self.freeze_large_model:
            for param in self.large_model.parameters():
                param.requires_grad = False
            self.large_model.eval()

        if self.use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["c_attn", "q_proj", "v_proj", "k_proj"] 
            )
            self.large_model = get_peft_model(self.large_model, lora_config)
    
    def forward(self, patchtst_features, patched_input=None):
        """
        Args:
            patchtst_features: [bs * nvars x patch_num x d_model] - PatchTST前半部分编码器输出（必须）
            patched_input: [bs * nvars x patch_num x patch_len] - 原始patch数据（可选）
        Returns:
            enhanced_features: [bs * nvars x patch_num x d_model] - 增强后的特征
        """
        if self.large_model is None:
            return patchtst_features
        
        batch_size, patch_num, _ = patchtst_features.shape
        device = patchtst_features.device
        
        input_components = []
        component_info = []  
        
        if self.use_soft_prompt:
            soft_prompts = self.soft_prompts.expand(batch_size, -1, -1)  # [bs * nvars x soft_prompt_length x large_model_dim]
            input_components.append(soft_prompts)
            component_info.append(('soft_prompt', self.soft_prompt_length))
        
        text_embeddings_1 = self.large_model.get_input_embeddings()(self.text_tokens_1['input_ids'].to(device)).squeeze(0)
        text_embeddings_1 = text_embeddings_1.expand(batch_size, -1, -1)  # [bs * nvars x text_len_1 x large_model_dim]
        input_components.append(text_embeddings_1)
        component_info.append(('text_1', text_embeddings_1.shape[1]))
        
        text_embeddings_2 = self.large_model.get_input_embeddings()(self.text_tokens_2['input_ids'].to(device)).squeeze(0)
        text_embeddings_2 = text_embeddings_2.expand(batch_size, -1, -1)  # [bs * nvars x text_len_2 x large_model_dim]
        input_components.append(text_embeddings_2)
        component_info.append(('text_2', text_embeddings_2.shape[1]))
        
        projected_features = self.projection_to_large(patchtst_features)  # [bs * nvars x patch_num x large_model_dim]
        input_components.append(projected_features)
        component_info.append(('patchtst_features', patch_num))
        
        if self.use_patched_input and patched_input is not None:
            text_embeddings_3 = self.large_model.get_input_embeddings()(self.text_tokens_3['input_ids'].to(device)).squeeze(0)
            text_embeddings_3 = text_embeddings_3.expand(batch_size, -1, -1)  # [bs * nvars x text_len_3 x large_model_dim]
            input_components.append(text_embeddings_3)
            component_info.append(('text_3', text_embeddings_3.shape[1]))
        
        if self.use_patched_input and patched_input is not None:
            projected_patch_input = self.projection_patch_to_large(patched_input)  # [bs * nvars x patch_num x large_model_dim]
            input_components.append(projected_patch_input)
            component_info.append(('patched_input', patch_num))
        
        combined_input = torch.cat(input_components, dim=1)  # [bs * nvars x total_len x large_model_dim]
        
        with torch.no_grad() if self.freeze_large_model else torch.enable_grad():
            with torch.cuda.amp.autocast():
                large_model_output = self.large_model(inputs_embeds=combined_input, output_hidden_states=True)
                large_model_features = large_model_output.hidden_states[-1]
        
        start_idx = 0
        extracted_features = {}
        
        for component_type, component_len in component_info:
            end_idx = start_idx + component_len
            if component_type in ['patchtst_features', 'patched_input']:
                extracted_features[component_type] = large_model_features[:, start_idx:end_idx, :]  # [bs * nvars x patch_num x large_model_dim]
            start_idx = end_idx

        final_large_features = extracted_features['patched_input']
        
        final_features = self.projection_from_large(final_large_features)  # [bs * nvars x patch_num x d_model]
        
        return final_features

