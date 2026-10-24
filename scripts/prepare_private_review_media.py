#!/usr/bin/env python3
"""Make local face-pixelated video/crops using the reconstructed head projection.

Original research inputs are preserved. Output metadata keeps the original video
identity for geometry alignment and separately records the altered media hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pixelate(image, rectangle, cells=10):
    left, top, right, bottom = np.asarray(rectangle, int)
    left, right = np.clip([left, right], 0, image.shape[1])
    top, bottom = np.clip([top, bottom], 0, image.shape[0])
    if right <= left or bottom <= top:
        raise ValueError('Empty head rectangle')
    patch = image[top:bottom, left:right]
    small = cv2.resize(patch, (min(cells, right-left), min(cells, bottom-top)), interpolation=cv2.INTER_AREA)
    image[top:bottom, left:right] = cv2.resize(small, (right-left, bottom-top), interpolation=cv2.INTER_NEAREST)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('video','human','object','crops','output-video','output-crops','metadata'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cells', type=int, default=10)
    args = parser.parse_args()
    if args.cells < 2:
        raise ValueError('At least two cells per axis required')
    for path in (args.output_video, args.output_crops, args.metadata):
        if path.exists():
            raise FileExistsError(path)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'PRISM'))
    from prism_reconstruction.geometry import smpl_vertex_regions
    with np.load(args.human, allow_pickle=False) as archive:
        human = dict(archive)
    with np.load(args.object, allow_pickle=False) as archive:
        box = dict(archive)
    crop_meta = json.loads((args.crops/'crop_metadata.json').read_text())
    source_sha = sha(args.video)
    if source_sha != str(human['source_video_sha256']) or source_sha != crop_meta['source_video_sha256']:
        raise ValueError('Video/geometry/crop provenance mismatch')
    labels = smpl_vertex_regions(human['template_vertices'], human['skinning_weights'], human['rest_joints'])
    bounds = []
    for frame, camera in enumerate(human['world_from_camera']):
        vertices = (human['human_vertices_world'][frame, labels=='head']-camera[:3,3])@camera[:3,:3]
        if np.any(vertices[:,2] <= 0):
            raise ValueError('Head projection behind the camera')
        uv = vertices@box['intrinsic_full'].T
        uv = uv[:,:2]/uv[:,2,None]
        low, high = uv.min(0), uv.max(0)
        margin = np.maximum((high-low)*.25, 12.)
        bounds.append(np.r_[low-margin, high+margin])
    bounds = np.asarray(bounds)
    # A five-frame envelope covers small reconstruction jitter without lagging
    # behind a moving face. It also covers the back of the head on turned frames.
    bounds[:,:2] = minimum_filter1d(bounds[:,:2], 5, axis=0, mode='nearest')
    bounds[:,2:] = maximum_filter1d(bounds[:,2:], 5, axis=0, mode='nearest')
    bounds = np.column_stack([np.floor(bounds[:,:2]), np.ceil(bounds[:,2:])]).astype(int)
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate','-of','json',str(args.video)]))['streams'][0]
    width, height = probe['width'], probe['height']
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_crops.mkdir(parents=True)
    cap = cv2.VideoCapture(str(args.video))
    encoder = subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s',f'{width}x{height}','-r',probe['r_frame_rate'],'-i','pipe:0','-i',str(args.video),
        '-map','0:v:0','-map','1:a?','-c:v','libx264','-threads','2','-preset','fast','-crf','20',
        '-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart','-shortest',str(args.output_video)], stdin=subprocess.PIPE)
    hashes = []
    try:
        for frame, rectangle in enumerate(bounds):
            ok, image = cap.read()
            if not ok:
                raise ValueError('Video ended before reconstruction')
            pixelate(image, rectangle, args.cells)
            encoder.stdin.write(image.tobytes())
            # Use the exact original JPEG crop outside the privacy rectangle;
            # this avoids unnecessary video-decoding changes to contact pixels.
            crop = cv2.imread(str(args.crops/f'{frame:05d}.jpg'))
            left, top, _, _ = crop_meta['frames'][frame]['crop_ltrb']
            local = rectangle-np.array([left,top,left,top])
            pixelate(crop, local, args.cells)
            path = args.output_crops/f'{frame:05d}.jpg'
            if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(path)
            hashes.append(sha(path))
        if cap.read()[0]:
            raise ValueError('Video has additional frames')
    finally:
        cap.release()
        encoder.stdin.close()
        status = encoder.wait()
    if status:
        raise RuntimeError('ffmpeg encoding failed')
    metadata = dict(complete=True, source_video_sha256=source_sha, original_video=str(args.video.resolve()),
        output_video=str(args.output_video.resolve()), output_video_sha256=sha(args.output_video),
        source_human_sha256=sha(args.human), source_object_sha256=sha(args.object),
        source_crop_metadata_sha256=sha(args.crops/'crop_metadata.json'),
        code_sha256=sha(__file__), frames=len(bounds), head_rectangles_ltrb=bounds.tolist(),
        pixel_cells_per_axis=args.cells, method='Projected SMPL head, 25%/12px margin and five-frame spatial envelope; local area-downsample/nearest-neighbor pixelation.',
        limitations='Light privacy pixelation, not a formal anonymization guarantee. Coverage must be visually checked; original inputs are preserved locally.')
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2))
    crop_meta.update(privacy_pixelated=True, privacy_metadata=str(args.metadata.resolve()),
        privacy_metadata_sha256=sha(args.metadata), pixelated_frame_sha256=hashes,
        source_original_crops=str(args.crops.resolve()),
        purpose='Face-pixelated default review/inference copy; same geometry and original source identity.')
    (args.output_crops/'crop_metadata.json').write_text(json.dumps(crop_meta, indent=2))
    print(json.dumps({k:metadata[k] for k in ['complete','frames','output_video','output_video_sha256']}))


if __name__ == '__main__':
    main()
