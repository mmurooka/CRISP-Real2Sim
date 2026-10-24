#!/usr/bin/env python3
"""Track first-frame box surface points with TAPIP3D in the raw camera world."""
import argparse
import copy
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from export_hoi_human import sha256


def load_tracking_model(checkpoint):
    """Use the complete trained state instead of downloading its initialization."""
    import models
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=copy.deepcopy(state['cfg'])
    config['model']['encoder']['pretrained']=False
    model=models.from_config(config['model'],image_size=config['train_dataset']['resolution'])
    model.load_state_dict(state['weight'],strict=True)
    if hasattr(model,'eval_mode'): model.set_eval_mode('raw')
    return model.eval()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--masks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-points',type=int,default=256)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'prep/MogeSAM'))
    from utils.inference_utils import inference,resize_depth_bilinear
    checkpoint=root/'prep/MogeSAM/checkpoints/tapip3d_final.pth'
    with np.load(args.prior,allow_pickle=False) as a:
        images=a['images']; depths=a['depths'].astype(np.float32)
        intrinsic=a['intrinsic'].astype(np.float32).copy(); c2w=a['cam_c2w'].astype(np.float32)
    model=load_tracking_model(checkpoint).cuda()
    h,w=map(int,model.image_size)
    original_h,original_w=depths.shape[1:]
    intrinsic[0]*=(w-1)/(original_w-1); intrinsic[1]*=(h-1)/(original_h-1)
    rgb=np.stack([cv2.resize(image,(w,h)) for image in images])
    depth=np.stack([resize_depth_bilinear(value,(w,h)) for value in depths])
    with np.load(args.masks/'box/dyn_mask_0.npz') as a:
        mask=cv2.resize(a['dyn_mask'].squeeze().astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
    mask=cv2.erode(mask,np.ones((3,3),np.uint8)).astype(bool)
    ys,xs=np.nonzero(mask & np.isfinite(depth[0]) & (depth[0]>0))
    if len(xs)<12: raise ValueError('Too few valid first-frame box depth pixels')
    # Deterministic spatial coverage, rather than choosing only high-texture corners.
    rng=np.random.default_rng(0)
    ids=rng.choice(len(xs),min(len(xs),args.max_points),replace=False)
    xy=np.c_[xs[ids],ys[ids]]
    camera_points=(np.c_[xy,np.ones(len(xy))]@np.linalg.inv(intrinsic).T)*depth[0,ys[ids],xs[ids],None]
    world_points=camera_points@c2w[0,:3,:3].T+c2w[0,:3,3]
    queries=np.c_[np.zeros(len(xy)),world_points].astype(np.float32)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        coords,visible=inference(model=model,
            video=torch.from_numpy(rgb).permute(0,3,1,2).float().cuda()/255,
            depths=torch.from_numpy(depth).cuda(),
            intrinsics=torch.from_numpy(np.repeat(intrinsic[None],len(images),axis=0)).cuda(),
            extrinsics=torch.from_numpy(np.linalg.inv(c2w)).cuda(),
            query_point=torch.from_numpy(queries).cuda(),grid_size=16,num_iters=6,vis_threshold=.5)
    points=coords.float().cpu().numpy(); visibility=visible.cpu().numpy()
    if points.shape!=(len(images),len(xy),3) or visibility.shape!=points.shape[:2]:
        raise ValueError('Unexpected TAPIP3D result shape')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,points_raw_world=points,visibility=visibility,
        first_frame_query_pixels=xy,tracking_intrinsic=intrinsic,tracking_image_size_hw=[h,w],
        source_prior_sha256=sha256(args.prior),checkpoint_sha256=sha256(checkpoint),
        coordinate_convention='raw_megasam_world_scale_not_yet_calibrated')
    print(f'Saved {points.shape}; visible fraction={visibility.mean():.3f}; {args.output}')


if __name__=='__main__': main()
