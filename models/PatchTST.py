__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, 
                 **kwargs):
        
        super().__init__()
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size
        large_model_type = configs.large_model_type
        large_model_name = configs.large_model_name
        fusion_type = configs.fusion_type
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, use_large_model=bool(configs.use_large_model), large_model_type=large_model_type, large_model_name=large_model_name, fusion_type=fusion_type,
                                  text_prompt_1=configs.text_prompt_1, text_prompt_2=configs.text_prompt_2, text_prompt_3=configs.text_prompt_3,
                                  use_soft_prompt=bool(configs.use_soft_prompt), soft_prompt_length=configs.soft_prompt_length, use_patched_input=bool(configs.use_patched_input),
                                  freeze_large_model=bool(configs.freeze_large_model), use_lora=bool(configs.use_lora), lora_r=configs.lora_r, lora_alpha=configs.lora_alpha, lora_dropout=configs.lora_dropout, llm_num_layers=configs.llm_num_layers, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, use_large_model=bool(configs.use_large_model), large_model_type=large_model_type, large_model_name=large_model_name, fusion_type=fusion_type,
                                  text_prompt_1=configs.text_prompt_1, text_prompt_2=configs.text_prompt_2, text_prompt_3=configs.text_prompt_3,
                                  use_soft_prompt=bool(configs.use_soft_prompt), soft_prompt_length=configs.soft_prompt_length, use_patched_input=bool(configs.use_patched_input),
                                  freeze_large_model=bool(configs.freeze_large_model), use_lora=bool(configs.use_lora), lora_r=configs.lora_r, lora_alpha=configs.lora_alpha, lora_dropout=configs.lora_dropout, llm_num_layers=configs.llm_num_layers, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, use_large_model=bool(configs.use_large_model), large_model_type=large_model_type, large_model_name=large_model_name, fusion_type=fusion_type,
                                  text_prompt_1=configs.text_prompt_1, text_prompt_2=configs.text_prompt_2, text_prompt_3=configs.text_prompt_3,
                                  use_soft_prompt=bool(configs.use_soft_prompt), soft_prompt_length=configs.soft_prompt_length, use_patched_input=bool(configs.use_patched_input),
                                  freeze_large_model=bool(configs.freeze_large_model), use_lora=bool(configs.use_lora), lora_r=configs.lora_r, lora_alpha=configs.lora_alpha, lora_dropout=configs.lora_dropout, llm_num_layers=configs.llm_num_layers, **kwargs)
    
    
    def forward(self, x, output_mid=False):           # x: [Batch, Input length, nvars]
        # import pdb; pdb.set_trace()
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, nvars, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, nvars]
        else:
            x = x.permute(0,2,1)    # x: [Batch, nvars, Input length]
            x, mid = self.model(x)
            # import pdb; pdb.set_trace()
            x = x.permute(0,2,1)    # x: [Batch, Input length, nvars], mid: [bs x nvars x d_model=16 x patch_num]
        if output_mid:
            return x, mid
        else:
            return x