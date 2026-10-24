# Adapted from MegaSAM cvd_opt.py, Copyright 2025 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
"""Checkpoint the pairwise CVD loss without changing its global normalization."""
import importlib.util
from pathlib import Path
import os
import sys
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SOURCE=Path(__file__).resolve().parents[1]/'prep/MogeSAM/third_party/megasam/cvd_opt/cvd_opt.py'
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('crisp_original_cvd',SOURCE)
original=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def _pair_terms(cam,K,K_inv,disp,uncertainty,flows,masks,ii,jj,invalid):
    h,w=disp.shape[-2:]
    yy,xx=torch.meshgrid(torch.arange(h,device=disp.device),torch.arange(w,device=disp.device),indexing='ij')
    grid=torch.stack([xx,yy],dim=-1).to(disp.dtype)[None]
    relative=torch.linalg.inv(cam[jj])@cam[ii]
    pixel=grid+flows.permute(0,2,3,1)
    normalized=2*pixel/torch.tensor([w-1,h-1],device=disp.device,dtype=disp.dtype)-1
    sampled=F.grid_sample(disp[jj,None],normalized,align_corners=True)
    sampled_invalid=F.grid_sample(invalid[jj,None],normalized,align_corners=True)>1e-4
    weight=uncertainty[ii].squeeze(1)
    grid_h=torch.cat([grid,torch.ones_like(grid[...,:1])],dim=-1)[...,None]
    ref_depth=1/torch.clamp(disp[ii],1e-3,1e3)
    invalid_flow=(invalid[ii]>0)|sampled_invalid.squeeze(1)
    xyz=ref_depth[...,None,None]*(K_inv[None,None,None]@grid_h)
    target=relative[:,None,None,:3,:3]@xyz+relative[:,None,None,:3,3:4]
    target_disp=1/torch.clamp(target[:,:, :,2:3,0],.1,1e3)
    projected=K[None,None,None]@target
    valid=masks.squeeze(1)*(projected[...,2,0]>.1)*(~invalid_flow).float()
    projected=projected[...,:2,0]/torch.clamp(projected[...,2:,0],1e-3,1e3)
    sampled=torch.clamp(sampled.squeeze(1),1e-3,1e2)
    target_disp=torch.clamp(target_disp.squeeze(-1),1e-3,1e2)
    ratio=torch.maximum(sampled/target_disp,target_disp/sampled)
    # Match upstream uncertainty weighting and masks exactly.
    ratio_numerator=torch.sum(((ratio-1).abs()*weight+original.ALPHA_MOTION*torch.log(1/weight))*valid)
    flow_numerator=torch.sum(((projected-pixel).abs()*weight[...,None]
                              +original.ALPHA_MOTION*torch.log(1/weight[...,None]))*valid[...,None])
    return ratio_numerator,flow_numerator,valid.sum()


def consistency_loss(cam_c2w,K,K_inv,disp_data,init_disp,uncertainty,flows,flow_masks,
                     ii,jj,compute_normals,fg_alpha,invalid_masks,
                     w_ratio=1.,w_flow=.2,w_si=1.,w_grad=2.,w_normal=4.):
    batch=int(os.getenv('CRISP_CVD_PAIR_BATCH','32'))
    if batch<1: raise ValueError('CRISP_CVD_PAIR_BATCH must be positive')
    numer_ratio=disp_data.new_zeros(()); numer_flow=disp_data.new_zeros(()); denominator=disp_data.new_zeros(())
    invalid=invalid_masks.to(disp_data.dtype)
    for start in range(0,len(ii),batch):
        end=start+batch
        ratio,flow,denom=checkpoint(_pair_terms,cam_c2w,K,K_inv,disp_data,uncertainty,
            flows[start:end],flow_masks[start:end],ii[start:end],jj[start:end],invalid,use_reentrant=False)
        numer_ratio=numer_ratio+ratio; numer_flow=numer_flow+flow; denominator=denominator+denom
    loss_ratio=numer_ratio/(denominator+1e-8)
    loss_flow=numer_flow/(2*denominator+1e-8)
    loss_prior=original.si_loss(init_disp,disp_data,invalid_masks)
    pred_normal=compute_normals[0](1/torch.clamp(disp_data[:,None],1e-3,1e3),K_inv[None])
    init_normal=compute_normals[0](1/torch.clamp(init_disp[:,None],1e-3,1e3),K_inv[None])
    alpha=fg_alpha*(~invalid_masks).float()
    loss_normal=torch.mean(alpha*(1-torch.sum(pred_normal*init_normal,dim=1)))
    loss_grad=disp_data.new_zeros(())
    for scale in range(4):
        factor=1/(2**scale)
        pred=F.interpolate(disp_data[:,None],scale_factor=factor,mode='nearest-exact')
        init=F.interpolate(init_disp[:,None],scale_factor=factor,mode='nearest-exact')
        u=F.interpolate(uncertainty,scale_factor=factor,mode='nearest-exact')
        mask=F.interpolate(invalid_masks[:,None].to(torch.uint8),scale_factor=factor,mode='nearest-exact').bool()
        loss_grad=loss_grad+original.gradient_loss(torch.log(pred),torch.log(init),u,mask)
    return w_ratio*loss_ratio+w_flow*loss_flow+w_si*loss_prior+w_grad*loss_grad+w_normal*loss_normal
