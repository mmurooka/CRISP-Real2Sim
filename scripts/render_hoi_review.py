#!/usr/bin/env python3
"""Local review video: source crop, camera overlay and an independent 3D view."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
import torch
import trimesh

from export_hoi_human import sha256


def look_at(eye,target,up):
    z=target-eye;z/=np.linalg.norm(z)
    x=np.cross(z,up);x/=np.linalg.norm(x)
    y=np.cross(z,x)
    return np.array([x,y,z])


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human',type=Path,required=True)
    parser.add_argument('--object',type=Path,required=True)
    parser.add_argument('--crops',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--side',type=int,default=512)
    parser.add_argument('--frames',type=int,nargs='+',help='Selected source indices for diagnostic stills')
    parser.add_argument('--fixed-view',action='store_true',help='Use a fixed wide 3D camera instead of following the actors')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'prep/HMR'))
    from hmr4d.utils.vis.renderer import Renderer
    with np.load(args.human) as h:human={k:h[k] for k in h.files}
    with np.load(args.object) as b:box={k:b[k] for k in b.files}
    for key in ('world_frame','source_video_sha256'):
        if str(human[key])!=str(box[key]):raise ValueError(f'Mismatched {key}')
    crop_meta=json.loads((args.crops/'crop_metadata.json').read_text())
    if crop_meta['source_video_sha256']!=str(human['source_video_sha256']):raise ValueError('Crop video mismatch')
    if not np.allclose(human['timestamps_s'],box['timestamps_s']):raise ValueError('Timing mismatch')
    size=args.side;count=len(human['timestamps_s']);mesh=trimesh.creation.box(extents=2*box['box_half_extents'])
    faces=np.r_[human['faces'],mesh.faces+human['human_vertices_world'].shape[1]]
    colors=np.r_[np.tile([.35,.65,.95],(6890,1)),np.tile([.95,.6,.25],(8,1))]
    colors=torch.tensor(colors[None],dtype=torch.float32,device='cuda')
    up=box['floor_normal_world'];target=human['human_vertices_world'][:,::100].mean(axis=(0,1))
    eye=target+Rotation.from_rotvec(up*np.deg2rad(-55)).apply(-target)*1.25
    view=look_at(eye,target,up)
    # Fit a constant focal length; optionally follow translation to keep
    # contact details readable. This camera is only for reviewing geometry.
    box_all=np.einsum('tij,vj->tvi',box['world_from_object'][:,:3,:3],mesh.vertices)+box['world_from_object'][:,None,:3,3]
    all_points=np.concatenate([human['human_vertices_world'][:,::40],box_all],axis=1)
    if args.fixed_view:
        eyes=np.repeat(eye[None],count,axis=0)
    else:
        centers=(all_points.min(axis=1)+all_points.max(axis=1))/2
        centers=gaussian_filter1d(centers,2,axis=0,mode='nearest')
        eyes=centers+(eye-target)/np.linalg.norm(eye-target)*3.5
    view_points=(all_points-eyes[:,None])@view.T
    extent=np.abs(view_points[...,:2]/view_points[...,2:]).max()
    focal=min(size*1.4,size*.43/extent)
    k_view=np.array([[focal,0,size/2],[0,focal,size/2],[0,0,1.]])
    viewer=Renderer(size,size,faces=faces,K=torch.tensor(k_view,dtype=torch.float32))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError(args.output)
    dt=np.diff(human['timestamps_s'])
    if not np.allclose(dt,dt[0],atol=1e-6):raise ValueError('Renderer requires constant presentation rate')
    # Frame timestamps are rounded to microseconds by ffprobe. Recover the
    # rational rate from the full duration, not a single rounded interval.
    rate=Fraction(float((count-1)/(human['timestamps_s'][-1]-human['timestamps_s'][0]))).limit_denominator(1001)
    fps=float(rate);encoder=None;stills=[]
    if not args.frames:
        encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size*3}x{size}',
            '-r',str(rate),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],stdin=subprocess.PIPE)
    try:
        for i in (args.frames or range(count)):
            rgb=cv2.cvtColor(cv2.imread(str(args.crops/f'{i:05d}.jpg')),cv2.COLOR_BGR2RGB)
            rgb=cv2.resize(rgb,(size,size));pose=box['world_from_object'][i]
            vertices=np.r_[human['human_vertices_world'][i],np.asarray(mesh.vertices)@pose[:3,:3].T+pose[:3,3]]
            camera=human['world_from_camera'][i]
            camera_vertices=(vertices-camera[:3,3])@camera[:3,:3]
            k=box['intrinsic_full'].copy();crop=crop_meta['frames'][i]['crop_ltrb'];k[:2,2]-=crop[:2]
            k[:2]*=size/crop_meta['square_side_pixels']
            renderer=Renderer(size,size,faces=faces,K=torch.tensor(k,dtype=torch.float32))
            rendered=renderer.render_mesh(torch.tensor(camera_vertices,dtype=torch.float32,device='cuda'),rgb,colors,VI=1)
            overlay=cv2.addWeighted(rgb,.40,rendered,.60,0)
            side=viewer.render_mesh(torch.tensor((vertices-eyes[i])@view.T,dtype=torch.float32,device='cuda'),colors=colors,VI=1)
            panels=[rgb,overlay,side]
            for panel,label in zip(panels,['RGB','Reconstruction overlay','3D fixed view' if args.fixed_view else '3D following view']):
                cv2.rectangle(panel,(0,0),(size,32),(245,245,245),-1)
                cv2.putText(panel,label,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.57,(20,30,40),1,cv2.LINE_AA)
            combined=np.concatenate(panels,axis=1)
            cv2.putText(combined,f'frame {i:03d}  {human["timestamps_s"][i]:.2f}s  geometry only',(12,size-14),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,30,30),1,cv2.LINE_AA)
            if encoder:encoder.stdin.write(combined.tobytes())
            else:stills.append(combined)
            if i%30==0:print(f'Render {i}/{count}',flush=True)
    finally:
        if encoder:
            encoder.stdin.close()
            if encoder.wait():raise RuntimeError('Video encoding failed')
    if stills:cv2.imwrite(str(args.output),cv2.cvtColor(np.concatenate(stills,axis=0),cv2.COLOR_RGB2BGR))
    args.output.with_suffix('.json').write_text(json.dumps(dict(source_human_sha256=sha256(args.human),
        source_object_sha256=sha256(args.object),frame_indices=args.frames or list(range(count)),fps=fps,
        virtual_camera='fixed' if args.fixed_view else 'following_translation_constant_orientation_and_focal_length',
        interpretation='Estimated geometry, not contact labels or 3D ground truth'),indent=2))


if __name__=='__main__':main()
