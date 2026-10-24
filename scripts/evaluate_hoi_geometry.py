#!/usr/bin/env python3
"""Internal consistency checks; these are not independent 3D/contact accuracy."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from export_hoi_human import sha256
from refine_hoi_human import cuboid_signed_distance


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('human','object','scene','observations','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'prep/HMR'))
    from hmr4d.utils.vis.renderer import Renderer
    with np.load(args.human) as a:h=dict(a)
    with np.load(args.object) as a:b=dict(a)
    with np.load(args.scene) as a:person=a['person_mask'];box=a['object_mask'];k=a['intrinsic']
    with np.load(args.observations) as a:observations=a['keypoints_coco17']
    for key in ('world_frame','source_video_sha256'):
        if str(h[key])!=str(b[key]):raise ValueError(f'Mismatched {key}')
    if str(h['source_scene_sha256'])!=sha256(args.scene):raise ValueError('Scene provenance mismatch')
    coco=torch.load(root/'prep/HMR/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',map_location='cpu',weights_only=False).numpy()
    height,width=person.shape[1:];renderer=Renderer(width,height,faces=h['faces'],K=torch.tensor(k,dtype=torch.float32))
    records=[]
    for i,vertices in enumerate(h['human_vertices_world']):
        camera=h['world_from_camera'][i];incam=(vertices-camera[:3,3])@camera[:3,:3]
        depth=renderer.render_mesh_depth(torch.tensor(incam,dtype=torch.float32,device='cuda'),np.zeros((height,width,3),np.uint8),VI=1)
        predicted=(depth>0)&~box[i];target=person[i]&~box[i]
        iou=float((predicted&target).sum()/max(1,(predicted|target).sum()))
        joints=coco@incam;projected=joints@b['intrinsic_full'].T;uv=projected[:,:2]/projected[:,2:]
        error=np.linalg.norm(uv-observations[i,:,:2],axis=1)
        pose=b['world_from_object'][i]
        local=(vertices-pose[:3,3])@pose[:3,:3]
        gap=cuboid_signed_distance(torch.tensor(local),torch.tensor(b['box_half_extents'])).numpy()
        records.append(dict(frame=i,visible_silhouette_iou=iou,keypoint_error_px=error.tolist(),
            keypoint_visible=(observations[i,:,2]>.5).tolist(),vertices_penetrating_2cm=int((gap<-.02).sum()),
            max_penetration_m=float(max(0,-gap.min()))))
    iou=np.array([x['visible_silhouette_iou'] for x in records])
    errors=np.array([x['keypoint_error_px'] for x in records]);visible=observations[:,:,2]>.5
    summary=dict(visible_silhouette_iou_median=float(np.median(iou)),visible_silhouette_iou_p05=float(np.quantile(iou,.05)),
        visible_keypoint_error_median_px=float(np.median(errors[visible])),visible_keypoint_error_p95_px=float(np.quantile(errors[visible],.95)),
        vertices_penetrating_2cm=sum(x['vertices_penetrating_2cm'] for x in records),
        frames_penetrating_2cm=sum(x['vertices_penetrating_2cm']>0 for x in records),
        max_penetration_m=max(x['max_penetration_m'] for x in records))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary,frames=records,
        source_human_sha256=sha256(args.human),source_object_sha256=sha256(args.object),
        interpretation='Image observations and reconstructed geometry are not independent 3D/contact ground truth.'),indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
