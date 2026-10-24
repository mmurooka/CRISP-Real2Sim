#!/usr/bin/env python3
"""Render a local kinematic G1 contact trajectory with source video crops."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

import cv2
import mujoco
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--crops',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--side',type=int,default=512)
    parser.add_argument('--frames',type=int,nargs='+')
    parser.add_argument('--speed',type=float,default=1.,help='Presentation speed; source timestamps stay unchanged')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=args.trajectory;meta=json.loads((root/'metadata.json').read_text())
    if not np.isfinite(args.speed) or args.speed<=0:raise ValueError('Positive presentation speed required')
    physics=meta.get('schema')=='g1_unassisted_physics_rollout_v1'
    records=json.loads((root/'diagnostics.json').read_text())
    cropmeta=json.loads((args.crops/'crop_metadata.json').read_text())
    if meta['source_video_sha256']!=cropmeta['source_video_sha256']:raise ValueError('Video provenance mismatch')
    with np.load(root/'trajectory.npz') as a:trajectory=dict(a)
    model=mujoco.MjModel.from_xml_path(str(root/'scene.xml'));data=mujoco.MjData(model)
    model.vis.global_.offwidth=args.side;model.vis.global_.offheight=args.side
    renderer=mujoco.Renderer(model,height=args.side,width=args.side)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=3.1;camera.elevation=-15
    box=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'hoi_box')
    frames=trajectory['frame_indices'];times=trajectory['timestamps_s'];size=args.side
    selected=np.arange(len(frames)) if args.frames is None else np.array([int(np.flatnonzero(frames==f)[0]) for f in args.frames])
    process=None;stills=[];args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.frames is None:
        if meta['independent_diagnostic_frames']:raise ValueError('Independent frames cannot be rendered as continuous motion')
        rate=Fraction(float((len(times)-1)/(times[-1]-times[0])*args.speed)).limit_denominator(1001)
        process=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size*3}x{size}',
            '-r',str(rate),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)],stdin=subprocess.PIPE)
    try:
        for index in selected:
            frame=int(frames[index]);data.qpos[:]=trajectory['qpos'][index];mujoco.mj_forward(model,data)
            rgb=cv2.cvtColor(cv2.imread(str(args.crops/f'{frame:05d}.jpg')),cv2.COLOR_BGR2RGB)
            panels=[cv2.resize(rgb,(size,size))]
            camera.lookat[:]=(data.xpos[1]+data.xpos[box])/2;camera.lookat[2]=.8
            for azimuth in (135,90):
                camera.azimuth=azimuth;renderer.update_scene(data,camera)
                for target in records[index]['targets']:
                    body=target['robot_body_id']
                    points=[data.xpos[body]+data.xmat[body].reshape(3,3)@target['robot_point_local'],
                            data.xpos[box]+data.xmat[box].reshape(3,3)@target['point_object']]
                    for point,color in zip(points,([.1,.85,.4,1],[1,.15,.1,1])):
                        geom=renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.012,.012,.012]),point,np.eye(3).reshape(-1),np.array(color,dtype=np.float32))
                        renderer.scene.ngeom+=1
                panels.append(renderer.render().copy())
            title='G1 physics rollout' if physics else 'G1 contact IK'
            for panel,label in zip(panels,('Source RGB',title+' - oblique',title+' - side')):
                cv2.rectangle(panel,(0,0),(size,32),(245,245,245),-1)
                cv2.putText(panel,label,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.54,(20,30,40),1,cv2.LINE_AA)
            combined=np.concatenate(panels,axis=1)
            status=(meta.get('controller','physics')+' - simulated, no root/object assistance') if physics else 'KINEMATIC CANDIDATE - not a controller'
            cv2.putText(combined,f'frame {frame:03d}  {times[index]:.2f}s  {status}',
                (12,size-14),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,60,40),1,cv2.LINE_AA)
            if process:process.stdin.write(combined.tobytes())
            else:stills.append(combined)
    finally:
        renderer.close()
        if process:
            process.stdin.close()
            if process.wait():raise RuntimeError('Video encoding failed')
    if stills:cv2.imwrite(str(args.output),cv2.cvtColor(np.concatenate(stills,axis=0),cv2.COLOR_RGB2BGR))


if __name__=='__main__':main()
