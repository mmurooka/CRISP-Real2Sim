#!/usr/bin/env python3
"""Headless SMPL-topology geometry export in the calibrated CRISP scene frame."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scene_cameras(scene, frame_count):
    """CRISP stores scaled depths but unscaled camera translations plus scale."""
    cameras = np.asarray(scene['cam_c2w'], dtype=np.float64).copy()
    indices = np.asarray(scene.get('valid_frame_indices', np.arange(len(cameras))))
    if not np.array_equal(indices, np.arange(frame_count)) or len(cameras) != frame_count:
        raise ValueError('Scene omits/reorders frames; align continuous segments explicitly before export')
    scale = float(scene['scale'])
    if not np.isfinite(scale) or scale <= 0 or cameras.shape != (frame_count,4,4):
        raise ValueError('Invalid scene scale or camera shape')
    if not np.isfinite(cameras).all() or not np.allclose(cameras[:,3], [0,0,0,1]):
        raise ValueError('Invalid camera transforms')
    rotation = cameras[:,:3,:3]
    if not np.allclose(rotation.transpose(0,2,1)@rotation,np.eye(3),atol=1e-4) or not np.allclose(np.linalg.det(rotation),1,atol=1e-4):
        raise ValueError('Camera rotations must be proper rotations')
    cameras[:,:3,3] *= scale
    return cameras, scale


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--hmr-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(args.video)]))['frames']
    times=np.asarray([float(f['best_effort_timestamp_time']) for f in frames])
    times-=times[0]
    if np.any(np.diff(times)<=0):
        raise ValueError('Invalid presentation timestamps')
    with np.load(args.scene,allow_pickle=False) as archive:
        cameras,scale=scene_cameras(archive,len(times))
        intrinsics=archive['intrinsic'].copy()
        image_size=np.asarray(archive['images'].shape[1:3])
    root=Path(__file__).resolve().parents[1]
    from hmr4d.utils.body_model.body_model_smplx import BodyModelSMPLX
    import smplx
    body_dir=root/'prep/HMR/inputs/checkpoints/body_models'
    model=BodyModelSMPLX(model_path=str(body_dir),model_type='smplx',gender='neutral',
                         num_pca_comps=12,flat_hand_mean=False).to(args.device).eval()
    smpl=smplx.create(str(body_dir),model_type='smpl',gender='neutral',num_betas=10)
    mapping=torch.load(root/'prep/HMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt',map_location=args.device)
    params=torch.load(args.hmr_result,map_location='cpu')['smpl_params_incam']
    if any(len(value)!=len(times) for value in params.values()):
        raise ValueError('HMR/video frame count mismatch')
    vertices,joints=[],[]
    with torch.inference_mode():
        for begin in range(0,len(times),args.batch_size):
            chunk={name:value[begin:begin+args.batch_size].to(args.device) for name,value in params.items()}
            prediction=model(**chunk)
            cam_vertices=torch.stack([torch.matmul(mapping,v) for v in prediction.vertices]).cpu().numpy()
            cam_joints=prediction.joints[:,:22].cpu().numpy()
            pose=cameras[begin:begin+len(cam_vertices)]
            vertices.append(np.einsum('tij,tvj->tvi',pose[:,:3,:3],cam_vertices)+pose[:,None,:3,3])
            joints.append(np.einsum('tij,tvj->tvi',pose[:,:3,:3],cam_joints)+pose[:,None,:3,3])
    scene_sha=sha256(args.scene)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,human_vertices_world=np.concatenate(vertices).astype(np.float32),
        human_joints_world=np.concatenate(joints).astype(np.float32),faces=smpl.faces,
        template_vertices=smpl.v_template.numpy(),skinning_weights=smpl.lbs_weights.numpy(),
        rest_joints=(smpl.J_regressor@smpl.v_template).numpy(),
        timestamps_s=times,frame_indices=np.arange(len(times)),world_from_camera=cameras,
        scene_intrinsics=intrinsics,scene_image_size_hw=image_size,raw_scene_scale=scale,
        length_unit='m',world_frame='crisp_scene_'+scene_sha,
        scale_source='CRISP_depth_alignment_to_GVHMR_estimated_body_scale_height_unmeasured',
        source_video_sha256=sha256(args.video),source_scene_sha256=scene_sha,
        source_hmr_sha256=sha256(args.hmr_result),body_topology='SMPL_6890_from_SMPLX_mapping')
    print(args.output)


if __name__=='__main__':
    main()
