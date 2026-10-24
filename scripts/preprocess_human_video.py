#!/usr/bin/env python3
"""Run small-batch ViTPose and HMR2 features using the reviewed person masks."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    video, masks, output = (p.resolve() for p in (args.video,args.masks,args.output))
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a fresh output directory: {output}')
    metadata = json.loads((masks/'segmentation.json').read_text())
    if not metadata['complete']:
        raise ValueError('Segmentation must be complete')
    frames = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(video)]))['frames']
    times = np.array([float(f['best_effort_timestamp_time']) for f in frames])
    times -= times[0]
    if len(times) != metadata['frame_count'] or np.any(np.diff(times) <= 0):
        raise ValueError('Video/mask frame mismatch or invalid presentation timestamps')
    boxes = []
    for i in range(len(times)):
        with np.load(masks/'person'/f'dyn_mask_{i}.npz') as a:
            yy, xx = np.nonzero(a['dyn_mask'].squeeze())
        if not len(xx):
            raise ValueError(f'Person mask empty at frame {i}')
        boxes.append([xx.min(), yy.min(), xx.max()+1, yy.max()+1])
    root = Path(__file__).resolve().parents[1]
    os.chdir(root/'prep/HMR')  # Upstream checkpoint paths are project-relative.
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    from hmr4d.utils.preproc.vitpose import VitPoseExtractor
    from hmr4d.utils.preproc.vitfeat_extractor import Extractor, get_batch
    box_tensor = torch.tensor(boxes, dtype=torch.float32)
    bbx_xys = get_bbx_xys_from_xyxy(box_tensor, base_enlarge=1.2).float()
    crops, bbx_xys = get_batch(str(video), bbx_xys)
    output.mkdir(parents=True)
    torch.save(dict(bbx_xyxy=box_tensor, bbx_xys=bbx_xys), output/'bbx.pt')
    with torch.inference_mode():
        extractor = VitPoseExtractor(tqdm_leave=False)
        kp = []
        for i in range(0, len(times), args.batch_size):
            kp.append(extractor.extract(crops[i:i+args.batch_size], bbx_xys[i:i+args.batch_size]))
            print(f'ViTPose: {min(i+args.batch_size,len(times))}/{len(times)}', flush=True)
        kp = torch.cat(kp)
        torch.save(kp, output/'vitpose.pt')
        del extractor
        gc.collect(); torch.cuda.empty_cache()
        # Some submodule snapshots omit this small file, though all three exact
        # buffers are present in the official HMR2 checkpoint itself.
        mean = Path('hmr4d/network/hmr2/configs/smpl_mean_params.npz')
        if not mean.exists():
            state = torch.load('inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt', map_location='cpu')['state_dict']
            np.savez(mean, pose=state['smpl_head.init_body_pose'].numpy().reshape(-1),
                     shape=state['smpl_head.init_betas'].numpy().reshape(-1),
                     cam=state['smpl_head.init_cam'].numpy().reshape(-1))
            del state
        extractor = Extractor(tqdm_leave=False)
        features = []
        for i in range(0, len(times), args.batch_size):
            features.append(extractor.extract_video_features(crops[i:i+args.batch_size],bbx_xys[i:i+args.batch_size]))
            print(f'HMR2: {min(i+args.batch_size,len(times))}/{len(times)}', flush=True)
        torch.save(torch.cat(features), output/'vit_features.pt')
    np.savez_compressed(output/'observations.npz', timestamps_s=times, keypoints_coco17=kp.numpy(),
                        bbx_xyxy=box_tensor.numpy(), bbx_xys=bbx_xys.numpy())
    (output/'metadata.json').write_text(json.dumps(dict(
        source_video=str(video), source_video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        frame_count=len(times), batch_size=args.batch_size, bbox_source='SAM2_person_masks',
        stage='2d_keypoints_and_image_features_not_3d_reconstruction', complete=True),indent=2))


if __name__ == '__main__':
    main()
