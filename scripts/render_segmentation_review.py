#!/usr/bin/env python3
"""Render tracked masks for visual review; this is not contact inference."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.masks/'segmentation.json').read_text())
    if not metadata['complete']:
        raise ValueError('Segmentation is incomplete')
    if args.output.exists():
        raise FileExistsError(args.output)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=avg_frame_rate', '-of', 'json', str(args.video)]))
    rate = probe['streams'][0]['avg_frame_rate']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo',
        '-pixel_format', 'bgr24', '-video_size', '1280x720', '-framerate', rate,
        '-i', '-', '-an', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(args.output)], stdin=subprocess.PIPE)
    sample_ids = np.linspace(0, metadata['frame_count']-1, 6, dtype=int)
    samples, areas = [], []
    try:
        for i in range(metadata['frame_count']):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Video ended before mask frame {i}')
            row = {'frame': i}
            for name, color in [('person', (255, 170, 25)), ('box', (60, 190, 255))]:
                with np.load(args.masks/name/f'dyn_mask_{i}.npz') as data:
                    mask = np.asarray(data['dyn_mask']).squeeze().astype(bool)
                if mask.shape != frame.shape[:2]:
                    raise ValueError('Mask/video resolution mismatch')
                row[name] = int(mask.sum())
                frame[mask] = (.65*frame[mask]+.35*np.asarray(color)).astype(np.uint8)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)
            frame = cv2.resize(frame, (1280, 720))
            cv2.rectangle(frame, (0,0), (1280,70), (20,20,20), -1)
            cv2.putText(frame, 'RGB mask tracking only | blue: person | orange: box',
                        (18,28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Frame {i} / {metadata["frame_count"]-1} | Contact has NOT been inferred',
                        (18,55), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
            encoder.stdin.write(frame.tobytes())
            areas.append(row)
            if i in sample_ids:
                samples.append(cv2.resize(frame, (640,360)))
        if capture.read()[0]:
            raise ValueError('Video has more frames than segmentation')
    finally:
        capture.release()
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f'ffmpeg exited {code}')
    cv2.imwrite(str(args.output.with_suffix('.jpg')), np.vstack([
        np.hstack(samples[:3]), np.hstack(samples[3:])]))
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        type='segmentation_review_not_contact_estimation', frame_rate=rate, areas=areas), indent=2))


if __name__ == '__main__':
    main()
