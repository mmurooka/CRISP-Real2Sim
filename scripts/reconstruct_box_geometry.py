#!/usr/bin/env python3
"""Calibrate tracked box poses to CRISP human geometry and fit observed surfaces."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from export_hoi_human import sha256


def points_from_depth(depth, mask, intrinsic, c2w, limit=1500):
    yy,xx=np.nonzero(mask & np.isfinite(depth) & (depth>0))
    if not len(xx): return np.empty((0,3))
    ids=np.linspace(0,len(xx)-1,min(limit,len(xx)),dtype=int)
    yy,xx=yy[ids],xx[ids]
    rays=np.c_[xx,yy,np.ones(len(xx))]@np.linalg.inv(intrinsic).T
    camera=rays*depth[yy,xx,None]
    return camera@c2w[:3,:3].T+c2w[:3,3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root.parent/'PRISM'))
    from prism_reconstruction.tracking import track_rigid_object,estimate_box_geometry
    from prism_reconstruction.scene_support import fit_support_plane
    scene_meta=json.loads(args.scene.with_suffix('.json').read_text())
    with np.load(args.human,allow_pickle=False) as h:
        if str(h['source_scene_sha256'].item())!=sha256(args.scene):
            raise ValueError('Human export uses a different scene calibration')
        cameras=h['world_from_camera']; scale=float(h['raw_scene_scale'])
        times=h['timestamps_s']; world=str(h['world_frame'].item())
        video_sha=str(h['source_video_sha256'].item()); scale_source=str(h['scale_source'].item())
    with np.load(args.tracks,allow_pickle=False) as t:
        if str(t['source_prior_sha256'].item())!=scene_meta['source_prior_sha256']:
            raise ValueError('Tracking and scene use different camera priors')
        points=t['points_raw_world']*scale; visibility=t['visibility'].astype(float)
    with np.load(args.scene,allow_pickle=False) as s:
        depths=s['depths']; intrinsic=s['intrinsic']; box=s['object_mask']; person=s['person_mask']
    if len(points)!=len(times) or len(depths)!=len(times): raise ValueError('Frame count mismatch')
    # Reject visible tracks that drift into another image region.
    for i,pose in enumerate(cameras):
        camera=(points[i]-pose[:3,3])@pose[:3,:3]
        projected=camera@intrinsic.T
        valid=np.isfinite(projected).all(axis=1)&(projected[:,2]>0)
        xy=np.zeros((len(camera),2),int)
        xy[valid]=np.rint(projected[valid,:2]/projected[valid,2,None]).astype(int)
        valid &= (xy[:,0]>=0)&(xy[:,0]<box.shape[2])&(xy[:,1]>=0)&(xy[:,1]<box.shape[1])
        ids=np.flatnonzero(valid)
        valid[ids]&=cv2.dilate(box[i].astype(np.uint8),np.ones((3,3),np.uint8))[xy[ids,1],xy[ids,0]]>0
        visibility[i]*=valid
    tracking=track_rigid_object(points,visibility,threshold_m=.03)
    surfaces=[]
    for i in range(len(times)):
        mask=cv2.erode((box[i]&~person[i]).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        surfaces.append(points_from_depth(depths[i],mask,intrinsic,cameras[i]))
    geometry=estimate_box_geometry(tracking['world_from_reference'],surfaces)
    # Lower image strip is selected as a floor candidate; RANSAC must validate it.
    floor_samples=[]
    for i in np.linspace(0,len(times)-1,min(10,len(times)),dtype=int):
        mask=~(box[i]|person[i]); mask[:int(.75*mask.shape[0])]=False
        floor_samples.append(points_from_depth(depths[i],mask,intrinsic,cameras[i],limit=1000))
    floor=fit_support_plane(np.concatenate(floor_samples),cameras[:,:3,3].mean(axis=0))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,world_from_object=geometry['world_from_object'],
        box_half_extents=geometry['half_extents'],valid=tracking['valid'],
        tracking_rmse_m=tracking['rmse_m'],tracking_inlier_count=tracking['inlier_count'],
        timestamps_s=times,world_frame=world,source_video_sha256=video_sha,scale_source=scale_source,
        length_unit='m',object_id='box',shape_assumption=geometry['shape_assumption'],
        floor_normal_world=floor['normal_world'],floor_offset_m=floor['offset_m'],
        gravity_world=-9.81*floor['normal_world'])
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        valid_frames=int(tracking['valid'].sum()),total_frames=len(times),
        usable_for_full_sequence_contacts=bool(tracking['valid'].all()),
        estimated_dimensions_m=(2*geometry['half_extents']).tolist(),
        shape_assumption=geometry['shape_assumption'],floor_rmse_m=floor['rmse_m'],
        floor_inlier_fraction=floor['inlier_fraction'],
        note='Estimated observed-surface cuboid; hidden dimensions and gravity from floor remain assumptions. Failed tracking frames retain NaN poses.'),indent=2))
    print(args.output.with_suffix('.json').read_text())


if __name__=='__main__': main()
