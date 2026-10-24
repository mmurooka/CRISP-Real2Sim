#!/usr/bin/env python3
"""Square HOI crops retain both full person and object through CLIP center cropping."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from export_hoi_human import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--masks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--padding', type=float, default=.08)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.padding <= .5:
        raise ValueError('padding must be between 0 and .5')
    segmentation = json.loads((args.masks/'segmentation.json').read_text())
    video_sha = sha256(args.video)
    frames = sorted(args.frames.glob('*.jpg'))
    if not segmentation['complete'] or segmentation['frame_count'] != len(frames):
        raise ValueError('Masks are incomplete or have a different frame count')
    if segmentation.get('source_video_sha256', video_sha) != video_sha:
        raise ValueError('Masks do not match the input video')
    bounds = []
    for i, frame in enumerate(frames):
        union = None
        for name in ('person', 'box'):
            with np.load(args.masks/name/f'dyn_mask_{i}.npz') as source:
                mask = source['dyn_mask'].squeeze().astype(bool)
            union = mask if union is None else union | mask
        y, x = np.nonzero(union)
        if not len(x):
            raise ValueError(f'Empty person/object mask: {frame}')
        bounds.append([x.min(), y.min(), x.max()+1, y.max()+1])
    bounds = np.asarray(bounds)
    side = int(np.ceil(np.max(bounds[:, 2:]-bounds[:, :2])*(1+2*args.padding)))
    centers = gaussian_filter1d((bounds[:, :2]+bounds[:, 2:])/2, 3, axis=0)
    # Smoothing must never cut off either instance.
    centers = np.maximum(centers, bounds[:, 2:]-side/2+1)
    centers = np.minimum(centers, bounds[:, :2]+side/2-1)
    args.output.mkdir(parents=True)
    records = []
    for frame, center, bound in zip(frames, centers, bounds):
        image = cv2.imread(str(frame))
        h, w = image.shape[:2]
        left, top = np.floor(center-side/2).astype(int)
        right, bottom = left+side, top+side
        x0, y0, x1, y1 = max(0,left), max(0,top), min(w,right), min(h,bottom)
        crop = np.full((side,side,3),114,np.uint8)
        crop[y0-top:y1-top,x0-left:x1-left] = image[y0:y1,x0:x1]
        if not (left <= bound[0] and top <= bound[1] and right >= bound[2] and bottom >= bound[3]):
            raise ValueError(f'Crop cuts a mask: {frame}')
        if not cv2.imwrite(str(args.output/frame.name),crop,[cv2.IMWRITE_JPEG_QUALITY,95]):
            raise IOError(frame)
        records.append(dict(frame=frame.name,crop_ltrb=[int(v) for v in (left,top,right,bottom)]))
    (args.output/'crop_metadata.json').write_text(json.dumps(dict(
        source_video_sha256=video_sha,source_masks=str(args.masks.resolve()),frame_count=len(frames),
        source_mask_metadata_sha256=sha256(args.masks/'segmentation.json'),
        source_frame_sha256=[sha256(f) for f in frames],
        square_side_pixels=side,padding_fraction=args.padding,frames=records,complete=True,
        purpose='Full human and moving object remain visible after CLIP center crop; SMPL vertex coordinates are unchanged.'),indent=2))
    print(f'{len(frames)} square crops, side={side}, all instance masks contained')


if __name__ == '__main__':
    main()
