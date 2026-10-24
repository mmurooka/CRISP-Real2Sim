#!/usr/bin/env python3
"""Run MegaSAM stages with durable outputs, failure propagation and resume markers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolution',type=int,default=196608)
    parser.add_argument('--checkpointed-cvd',action='store_true')
    args=parser.parse_args()
    frames,work,output=(p.resolve() for p in (args.frames,args.work,args.output))
    files=sorted(frames.glob('*.jpg'))
    if not files: raise ValueError('No frames')
    digest=hashlib.sha256()
    for p in files: digest.update(p.name.encode()); digest.update(p.read_bytes())
    metadata=dict(frames=str(frames),frame_count=len(files),frames_sha256=digest.hexdigest(),
                  resolution=args.resolution,checkpointed_cvd=args.checkpointed_cvd)
    work.mkdir(parents=True,exist_ok=True)
    manifest=work/'input.json'
    if manifest.exists() and json.loads(manifest.read_text())!=metadata:
        raise ValueError('Work directory belongs to another input/configuration; use a fresh directory')
    manifest.write_text(json.dumps(metadata,indent=2))
    (work/'images').mkdir(exist_ok=True)
    if not (work/'images/test').exists(): (work/'images/test').symlink_to(frames,target_is_directory=True)
    root=Path(__file__).resolve().parents[1]
    mega=root/'prep/MogeSAM/third_party/megasam'
    upstream=runpy.run_path(str(mega/'inference.py'))
    stages=[
        ('depth_anything',upstream['DEPTHANYTHING_V1_CMD'],mega/'Depth-Anything','Depth-Anything/video_visualization/test/*.npy',len(files)),
        ('unidepth',upstream['UNIDEPTH_CMD'],mega/'UniDepth','UniDepth/outputs/test/*.npz',len(files)),
        ('moge',upstream['MOGE_CMD'].format(fov_x=''),mega/'MoGe','MoGe/video_visualization/test/*.npy',len(files)),
        ('camera',upstream['CAMERA_TRACKING_CMD'].format(fov_x='',depth_for_cvd='MoGe/video_visualization',resolution=args.resolution)+f' --buffer {len(files)+32}',mega,'reconstructions/test/*.npy',6),
        ('flow',upstream['PREPROCESS_FLOW_CMD'].format(resolution=args.resolution),mega,'cache/test/*.npy',3),
        ('cvd',upstream['CVD_OPT_MOGE_CMD'],mega,'outputs_cvd/test_sgd_cvd_hr.npz',1),
    ]
    env=os.environ.copy()
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    env['PYTHONUNBUFFERED']='1'
    for name,command,pythonpath,pattern,count in stages:
        marker=work/f'{name}.done.json'
        if marker.exists():
            saved=json.loads(marker.read_text())
            if not all((work/p).is_file() for p in saved['outputs']):
                raise ValueError(f'Completed stage {name} has missing outputs')
            print(f'Reuse stage: {name}',flush=True)
            continue
        argv=shlex.split(command)
        argv[0]=sys.executable
        if name=='cvd' and args.checkpointed_cvd:
            argv[1]=str(root/'scripts/run_cvd_checkpointed.py')
        env['PYTHONPATH']=str(pythonpath)
        print(f'Start stage: {name}',flush=True)
        with (work/f'{name}.log').open('w') as log:
            subprocess.run(argv,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        found=sorted(work.glob(pattern))
        # The upstream flow script uses a different fixed folder name.
        if name=='flow' and len(found)<count:
            found=[p for p in work.glob('**/*.npy') if p.name in ('flows.npy','flows_masks.npy','ii-jj.npy')]
        if len(found)<count:
            raise RuntimeError(f'{name}: expected {count} outputs, found {len(found)}')
        marker.write_text(json.dumps(dict(outputs=[str(p.relative_to(work)) for p in found]),indent=2))
        print(f'Completed stage: {name}',flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    shutil.copyfile(work/'outputs_cvd/test_sgd_cvd_hr.npz',output)
    print(f'Saved: {output}',flush=True)


if __name__=='__main__': main()
