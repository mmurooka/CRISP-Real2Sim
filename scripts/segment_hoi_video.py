#!/usr/bin/env python3
"""Segment one person and one box sequentially on a 16 GB GPU."""
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from sam2.build_sam import build_sam2_video_predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--objects', nargs='+', default=['person', 'box'])
    args = parser.parse_args()
    frames = sorted(args.frames.glob('*.jpg'))
    if not frames or [p.stem for p in frames] != [f'{i:05d}' for i in range(len(frames))]:
        raise ValueError('Input must contain contiguous, zero-based 00000.jpg frames')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    args.output.mkdir(parents=True, exist_ok=True)
    image = Image.open(frames[0]).convert('RGB')
    processor = AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    detector = AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny').cuda().eval()
    detections = {}
    with torch.inference_mode():
        for name in args.objects:
            prompt = 'a cardboard box.' if name == 'box' else f'a {name}.'
            inputs = processor(images=image, text=prompt, return_tensors='pt').to('cuda')
            result = processor.post_process_grounded_object_detection(
                detector(**inputs), inputs.input_ids, threshold=.25, text_threshold=.25,
                target_sizes=[image.size[::-1]],
            )[0]
            if not len(result['scores']):
                raise RuntimeError(f'No first-frame detection for {name}; inspect input before continuing')
            index = int(result['scores'].argmax())
            detections[name] = dict(box=result['boxes'][index].cpu().tolist(),
                                    score=float(result['scores'][index]))
            print(name, detections[name], flush=True)
    del detector, processor, inputs, result
    gc.collect(); torch.cuda.empty_cache()
    metadata = dict(frame_count=len(frames), detection=detections, complete=False,
                    detector='IDEA-Research/grounding-dino-tiny', sam2_checkpoint=str(args.checkpoint.resolve()))
    (args.output/'segmentation.json').write_text(json.dumps(metadata, indent=2))
    # One object at a time avoids two simultaneous large-model processes.
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictor = build_sam2_video_predictor('sam2_hiera_l.yaml', str(args.checkpoint.resolve()), device='cuda')
        for name in args.objects:
            folder = args.output/name
            folder.mkdir(exist_ok=True)
            state = predictor.init_state(video_path=str(args.frames.resolve()),
                                         offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                             box=np.asarray(detections[name]['box'], np.float32))
            observed = set()
            for frame_index, ids, logits in predictor.propagate_in_video(state):
                mask = (logits[ids.index(1)] > 0).cpu().numpy()
                np.savez_compressed(folder/f'dyn_mask_{frame_index}.npz', dyn_mask=mask)
                observed.add(frame_index)
                if frame_index % 30 == 0:
                    print(f'{name}: {frame_index+1}/{len(frames)}; pixels={int(mask.sum())}', flush=True)
            if observed != set(range(len(frames))):
                raise RuntimeError(f'Incomplete mask sequence for {name}')
            predictor.reset_state(state)
            del state
            gc.collect(); torch.cuda.empty_cache()
    metadata['complete'] = True
    (args.output/'segmentation.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
