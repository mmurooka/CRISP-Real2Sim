#!/usr/bin/env python3
"""Reuse completed human observations; preserve video timing and camera conventions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--camera-prior',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--static-camera',action='store_true')
    args=parser.parse_args()
    video,prior,output=(p.resolve() for p in (args.video,args.camera_prior,args.output))
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    from export_hoi_human import sha256
    observations=json.loads((output/'preprocess/metadata.json').read_text())
    if not observations['complete'] or observations['source_video_sha256']!=sha256(video):
        raise ValueError('Preprocessing does not match the current video')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,avg_frame_rate','-of','json',str(video)]))['streams'][0]
    length=observations['frame_count']
    metadata=dict(source_video_sha256=observations['source_video_sha256'],source_camera_sha256=sha256(prior),
        frame_count=length,frame_rate=probe['avg_frame_rate'],static_camera=args.static_camera,
        camera_rotation_convention='world_to_camera_for_angular_increment',complete=False)
    metadata_path=output/'reconstruction_metadata.json'
    prediction_path=output/'hmr4d_results.pt'
    if prediction_path.exists():
        previous=json.loads(metadata_path.read_text())
        if any(previous.get(k)!=v for k,v in metadata.items() if k!='complete'):
            raise ValueError('Saved HMR prediction belongs to a different input/configuration')
        if previous.get('complete'):
            print('Reuse completed HMR reconstruction and rendering',flush=True)
            return
    with np.load(prior,allow_pickle=False) as archive:
        camera=np.asarray(archive['cam_c2w'])
        intrinsic=np.asarray(archive['intrinsic']).copy()
        h,w=archive['images'].shape[1:3]
    if camera.shape!=(length,4,4):
        raise ValueError('Camera/video frame mismatch')
    os.chdir(root/'prep/HMR')
    from hydra import initialize_config_module,compose
    from hydra.utils import instantiate
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # Registers the inference Hydra config.
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.vis.renderer import Renderer
    with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
        register_store_gvhmr()
        cfg=compose(config_name='demo',overrides=[f'video_name={video.stem}',f'tgt_name={video.stem}'])
    full_k=intrinsic.copy(); full_k[0]*=probe['width']/w; full_k[1]*=probe['height']/h
    # compute_cam_angvel documents world-to-camera rotations, not c2w.
    rotations=torch.from_numpy(camera[:,:3,:3].transpose(0,2,1).copy()).float()
    data=dict(length=torch.tensor(length),
        bbx_xys=torch.load(output/'preprocess/bbx.pt',map_location='cpu')['bbx_xys'],
        kp2d=torch.load(output/'preprocess/vitpose.pt',map_location='cpu'),
        f_imgseq=torch.load(output/'preprocess/vit_features.pt',map_location='cpu'),
        K_fullimg=torch.from_numpy(full_k).float().repeat(length,1,1),
        cam_angvel=compute_cam_angvel(rotations))
    if prediction_path.exists():
        pred=torch.load(prediction_path,map_location='cpu')
        print('Reuse HMR prediction; resume rendering',flush=True)
    else:
        metadata_path.write_text(json.dumps(metadata,indent=2))
        model=instantiate(cfg.model,_recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model=model.eval().cuda()
        with torch.inference_mode():
            pred=detach_to_cpu(model.predict(data,static_cam=args.static_camera))
        torch.save(pred,output/'hmr4d_results.pt.tmp')
        (output/'hmr4d_results.pt.tmp').replace(prediction_path)
        del model; torch.cuda.empty_cache()
    # Render metric body depth in exactly the depth-prior image grid for scale alignment.
    body=make_smplx('supermotion').cuda().eval()
    mapping=torch.load('hmr4d/utils/body_model/smplx2smpl_sparse.pt',map_location='cuda')
    faces=make_smplx('smpl').faces
    renderer=Renderer(w,h,device='cuda',faces=faces,K=torch.from_numpy(intrinsic).float())
    capture=cv2.VideoCapture(str(video))
    depth_dir=output/'depth_out'; depth_dir.mkdir(exist_ok=True)
    encoder=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','bgr24',
        '-video_size',f'{w}x{h}','-framerate',probe['avg_frame_rate'],'-i','-','-an','-c:v','libx264',
        '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'human_incam.mp4')],stdin=subprocess.PIPE)
    try:
        with torch.inference_mode():
            for begin in range(0,length,16):
                params={k:v[begin:begin+16].cuda() for k,v in pred['smpl_params_incam'].items()}
                vertices=body(**params).vertices
                for j,verts in enumerate(vertices):
                    i=begin+j
                    verts=mapping@verts
                    ok,bgr=capture.read()
                    if not ok: raise ValueError(f'Missing video frame {i}')
                    rgb=cv2.cvtColor(cv2.resize(bgr,(w,h)),cv2.COLOR_BGR2RGB)
                    depth=renderer.render_mesh_depth(verts,rgb,[.8,.8,.8])
                    np.save(depth_dir/f'mesh_depth_{i}.npy',depth)
                    rendered=renderer.render_mesh(verts,rgb,[.7,.8,1.])
                    encoder.stdin.write(cv2.cvtColor(rendered,cv2.COLOR_RGB2BGR).tobytes())
                    if i%30==0: print(f'Render: {i+1}/{length}',flush=True)
    finally:
        capture.release(); encoder.stdin.close(); code=encoder.wait()
    if code: raise RuntimeError(f'ffmpeg exited {code}')
    metadata['complete']=True
    metadata_path.write_text(json.dumps(metadata,indent=2))


if __name__=='__main__': main()
