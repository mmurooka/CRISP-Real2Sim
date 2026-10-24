#!/usr/bin/env python3
"""Calibrate depth to the human body prior while excluding the moving box."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from export_hoi_human import sha256


def robust_scale(observed_depth, body_depth, visible_person, min_pixels=20):
    observed,body=np.asarray(observed_depth),np.asarray(body_depth)
    if observed.shape!=body.shape or np.asarray(visible_person).shape!=body.shape:
        raise ValueError('Depth and mask must share a pixel grid')
    valid=(visible_person & np.isfinite(observed) & np.isfinite(body) & (observed>0) & (body>0))
    if valid.sum()<min_pixels:
        raise ValueError('Insufficient visible human depth overlap for metric scale')
    ratio=body[valid]/observed[valid]
    value=float(np.median(ratio))
    if not .1<value<20:
        raise ValueError(f'Implausible depth scale: {value}')
    return value,int(valid.sum()),float(np.median(np.abs(ratio-value)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--hmr',type=Path,required=True)
    parser.add_argument('--masks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    hmr_meta=json.loads((args.hmr/'reconstruction_metadata.json').read_text())
    if not hmr_meta['complete'] or hmr_meta['source_camera_sha256']!=sha256(args.prior):
        raise ValueError('HMR/camera provenance mismatch or incomplete body rendering')
    with np.load(args.prior,allow_pickle=False) as source:
        images,depths=source['images'],source['depths'].astype(np.float32)
        cameras,intrinsic=source['cam_c2w'],source['intrinsic']
    count,h,w=depths.shape
    if count!=hmr_meta['frame_count']: raise ValueError('Frame count mismatch')
    person_masks,box_masks,stats=[],[],[]
    for i in range(count):
        masks=[]
        for name in ['person','box']:
            with np.load(args.masks/name/f'dyn_mask_{i}.npz') as a:
                mask=a['dyn_mask'].squeeze().astype(np.uint8)
            masks.append(cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool))
        person,box=masks
        body=np.load(args.hmr/'depth_out'/f'mesh_depth_{i}.npy')
        if body.shape!=(h,w): raise ValueError('Body depth was not rendered on the prior grid')
        visible=cv2.erode((person & ~box).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        scale,pixels,mad=robust_scale(depths[i],body,visible)
        stats.append(dict(frame=i,scale=scale,pixels=pixels,ratio_mad=mad))
        person_masks.append(person); box_masks.append(box)
    scale=float(np.median([s['scale'] for s in stats]))
    person_masks,box_masks=map(np.asarray,(person_masks,box_masks))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,images=images,depths=depths*scale,
        intrinsic=intrinsic,cam_c2w=cameras,scale=scale,valid_frame_indices=np.arange(count),
        enlarged_dynamic_mask=person_masks|box_masks,person_mask=person_masks,object_mask=box_masks,
        mask_shifts=np.empty(0),source_video_sha256=hmr_meta['source_video_sha256'])
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        source_video_sha256=hmr_meta['source_video_sha256'],source_prior_sha256=sha256(args.prior),scale=scale,frames=stats,
        scale_source='GVHMR_estimated_human_shape_not_measured_height',
        camera_convention='cam_c2w translations are raw; multiply by scale once',
        dynamic_mask='union of person and moving box; excluded from static scene'),indent=2))
    print(f'Aligned {count} frames; scale={scale:.6g}; {args.output}')


if __name__=='__main__': main()
