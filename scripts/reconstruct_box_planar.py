#!/usr/bin/env python3
"""Rigid cuboid initialization from floor-supported silhouettes and planar RGB tracks.

The box is assumed to rest flat on the reconstructed floor during stationary
endpoint frames. Its visible front is planar. No human contact labels are used.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from export_hoi_human import sha256


def polygon(mask, fraction=.009):
    contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    hull=cv2.convexHull(max(contours,key=cv2.contourArea))
    return cv2.approxPolyDP(hull,fraction*cv2.arcLength(hull,True),True).reshape(-1,2).astype(float)


def floor_points(pixels,k,camera,normal,offset):
    rays=np.c_[pixels,np.ones(len(pixels))]@np.linalg.inv(k).T@camera[:3,:3].T
    distance=(offset-camera[:3,3]@normal)/(rays@normal)
    if np.any(distance<=0): raise ValueError('Floor intersection behind camera')
    return camera[:3,3]+distance[:,None]*rays


def top_height(pixel,base,k,camera,normal):
    ray=camera[:3,:3]@np.linalg.solve(k,np.r_[pixel,1.])
    depth,height=np.linalg.lstsq(np.c_[ray,-normal],base-camera[:3,3],rcond=None)[0]
    error=np.linalg.norm(camera[:3,3]+depth*ray-base-height*normal)
    return float(height),float(error)


def calibrate_dimensions(polygons,centers,k,cameras,normal,offset):
    records=[]
    # Endpoint support is an explicit assumption, checked by image stationarity
    # and the two perpendicular base edges when both are visible.
    ids=np.r_[np.arange(min(30,len(polygons))),np.arange(max(0,len(polygons)-30),len(polygons))]
    for i in ids:
        lo,hi=max(0,i-3),min(len(centers),i+4)
        if np.max(np.linalg.norm(np.diff(centers[lo:hi],axis=0),axis=1),initial=0)>2.:
            continue
        poly=polygons[i]; middle=(poly[:,1].max()+poly[:,1].min())/2
        top=poly[poly[:,1]<middle]; bottom=poly[poly[:,1]>middle]
        top=top[np.argsort(top[:,0])]; bottom=bottom[np.argsort(bottom[:,0])]
        if len(top)<2 or len(bottom) not in (2,3): continue
        base=floor_points(bottom,k,cameras[i],normal,offset)
        heights=[top_height(top[j],base[j],k,cameras[i],normal) for j in (0,-1)]
        if max(v[1] for v in heights)>.06: continue
        height=float(np.mean([v[0] for v in heights]))
        if len(bottom)==2:
            width=float(np.linalg.norm(base[1]-base[0])); thickness=None
        else:
            edges=np.diff(base,axis=0); lengths=np.linalg.norm(edges,axis=1)
            cosine=float(edges[0]@edges[1]/np.prod(lengths))
            if abs(cosine)>.25: continue
            front=int(np.linalg.norm(np.diff(bottom,axis=0),axis=1).argmax())
            width=float(lengths[front]); thickness=float(lengths[1-front])
        if not .15<width<1.5 or not .4<height<2.5: continue
        records.append(dict(frame=int(i),width_m=width,height_m=height,thickness_m=thickness,
                            vertical_ray_residual_m=max(v[1] for v in heights)))
    if len(records)<3: raise ValueError('Insufficient stationary floor-supported endpoint views')
    thickness=[v['thickness_m'] for v in records if v['thickness_m'] is not None]
    if len(thickness)<3: raise ValueError('Box thickness not observable in endpoint silhouettes')
    dimensions=np.array([np.median([v['width_m'] for v in records]),
                         np.median([v['height_m'] for v in records]),np.median(thickness)])
    return dimensions,records


def project_homography(h,points):
    return cv2.perspectiveTransform(np.asarray(points,np.float32)[None],h)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=Path,required=True)
    parser.add_argument('--masks',type=Path,required=True)
    parser.add_argument('--human',type=Path,required=True)
    parser.add_argument('--floor',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    files=sorted(args.frames.glob('*.jpg')); count=len(files)
    with np.load(args.human) as human:
        cameras=human['world_from_camera'];times=human['timestamps_s'];k=human['scene_intrinsics'].copy()
        grid=human['scene_image_size_hw'];world=human['world_frame'];video_sha=human['source_video_sha256'];scale_source=human['scale_source']
    floor=json.loads(args.floor.read_text());normal=np.asarray(floor['normal_world']);offset=floor['offset_m']
    image=cv2.imread(str(files[0]));h,w=image.shape[:2];k[0]*=w/grid[1];k[1]*=h/grid[0]
    if len(cameras)!=count: raise ValueError('Frame/camera count mismatch')
    def mask_at(i,name='box'):
        with np.load(args.masks/name/f'dyn_mask_{i}.npz') as data: return data['dyn_mask'].squeeze().astype(np.uint8)
    polygons=[polygon(mask_at(i)) for i in range(count)]
    centers=np.array([p.mean(axis=0) for p in polygons])
    # Centroids from moments are less sensitive to polygon vertex count changes.
    for i in range(count):
        moments=cv2.moments(mask_at(i));centers[i]=[moments['m10']/moments['m00'],moments['m01']/moments['m00']]
    dimensions,calibration=calibrate_dimensions(polygons,centers,k,cameras,normal,offset)
    print('Estimated dimensions [width,height,thickness] m:',dimensions,flush=True)
    quad=polygon(mask_at(0),.02)
    if len(quad)!=4: raise ValueError('Initial front face must have four dominant silhouette corners')
    order=np.argsort(quad[:,1]);top=quad[order[:2]];bottom=quad[order[2:]]
    quad=np.array([top[np.argmin(top[:,0])],top[np.argmax(top[:,0])],bottom[np.argmax(bottom[:,0])],bottom[np.argmin(bottom[:,0])]],np.float32)
    width,height,thickness=dimensions
    front=np.array([[-width/2,height/2,thickness/2],[width/2,height/2,thickness/2],
                    [width/2,-height/2,thickness/2],[-width/2,-height/2,thickness/2]],np.float32)
    front_plane=front.copy();front_plane[:,2]=0
    reference_to_plane=cv2.getPerspectiveTransform(quad,front[:,:2].copy())
    reference=points=None;previous=None;H=np.eye(3);last_pose=None
    poses=np.full((count,4,4),np.nan);valid=np.zeros(count,bool)
    homographies=np.full((count,3,3),np.nan);errors=np.full(count,np.nan);inliers=np.zeros(count,int)
    observations=[]
    for i,path in enumerate(files):
        rgb=cv2.imread(str(path));gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        allowed=(mask_at(i)>0)&~(mask_at(i,'person')>0)
        if i:
            tracked,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,points[:,None],None,winSize=(25,25),maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
            back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,previous,tracked,None,winSize=(25,25),maxLevel=3)
            tracked=tracked[:,0];fb=np.linalg.norm(back[:,0]-points,axis=1)
            xy=np.rint(tracked).astype(int)
            keep=(status[:,0]>0)&(back_status[:,0]>0)&(fb<.8)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            ids=np.flatnonzero(keep);keep[ids]&=allowed[xy[ids,1],xy[ids,0]]
            reference,points=reference[keep],tracked[keep]
            if len(points)<8: raise RuntimeError(f'Insufficient front-face RGB tracks at frame {i}')
            H,keep=cv2.findHomography(reference,points,cv2.RANSAC,1.5,maxIters=3000,confidence=.999)
            if H is None or keep.sum()<8: raise RuntimeError(f'No consistent front-face homography at {i}')
            keep=keep[:,0].astype(bool);reference,points=reference[keep],points[keep]
        projected=project_homography(H,quad)
        # IPPE's canonical plane must have z=0. Translate the recovered plane
        # origin to the cuboid center only after solving the planar pose.
        pose_options=cv2.solvePnPGeneric(front_plane,projected,k,None,flags=cv2.SOLVEPNP_IPPE)
        candidates=[]
        for rvec,tvec in zip(pose_options[1],pose_options[2]):
            r=cv2.Rodrigues(rvec)[0];tvec=tvec-r[:,2:3]*thickness/2
            ok,rvec,tvec=cv2.solvePnP(front,projected,k,None,rvec.copy(),tvec.copy(),
                                      useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok: continue
            r=cv2.Rodrigues(rvec)[0]
            pose=np.eye(4);pose[:3,:3]=r;pose[:3,3]=tvec[:,0]
            if tvec[2,0]<=0 or r[:,2]@tvec[:,0]>=0: continue
            uv=cv2.projectPoints(front,rvec,tvec,k,None)[0][:,0]
            error=float(np.sqrt(np.mean((uv-projected)**2)))
            if last_pose is None:
                up=cameras[i,:3,:3]@r[:,1]
                score=error+10*(1-up@normal)
            else:
                angle=Rotation.from_matrix(last_pose[:3,:3].T@r).magnitude()
                score=error+5*angle+2*np.linalg.norm(pose[:3,3]-last_pose[:3,3])
            candidates.append((score,error,pose,rvec,tvec))
        if not candidates: raise RuntimeError(f'No positive-depth front-facing PnP solution at {i}')
        _,error,pose,rvec,tvec=min(candidates,key=lambda c:c[0])
        if error>30: raise RuntimeError(f'Catastrophic planar pose reprojection error {error:.2f}px at frame {i}')
        poses[i]=cameras[i]@pose;valid[i]=error<=5;last_pose=pose;homographies[i]=H;errors[i]=error
        if points is not None:
            inliers[i]=len(points)
            observations.append(dict(frame=i,reference=reference.tolist(),pixels=points.tolist()))
        # Replenish inside the tracked FRONT, excluding human occlusion and borders.
        region=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(region,np.rint(projected).astype(int),255)
        region=cv2.erode(region,np.ones((9,9),np.uint8));region[~allowed]=0
        if points is not None:
            for point in points: cv2.circle(region,tuple(np.rint(point).astype(int)),7,0,-1)
        new=cv2.goodFeaturesToTrack(gray,maxCorners=max(1,200-(0 if points is None else len(points))),
                                    qualityLevel=.004,minDistance=7,mask=region,blockSize=5)
        if new is not None:
            new=new[:,0];ref=project_homography(np.linalg.inv(H),new)
            points=new if points is None else np.r_[points,new]
            reference=ref if reference is None else np.r_[reference,ref]
        if points is None or len(points)<8: raise RuntimeError(f'Insufficient visible texture at frame {i}')
        previous=gray
        if i%30==0: print(f'Frame {i}/{count}: {len(points)} tracks, PnP RMS={error:.3f}px',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,world_from_object=poses,box_half_extents=dimensions/2,valid=valid,
        timestamps_s=times,world_frame=world,source_video_sha256=video_sha,scale_source=scale_source,
        length_unit='m',object_id='box',shape_assumption='rigid_cuboid_calibrated_from_floor_supported_endpoint_silhouettes',
        floor_normal_world=normal,floor_offset_m=offset,gravity_world=-9.81*normal,
        front_homography=homographies,front_reference_quad_pixels=quad,front_object_corners=front,
        pose_reprojection_rmse_px=errors,front_inlier_count=inliers,intrinsic_full=k)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        calibration=calibration,dimensions_m=dimensions.tolist(),frame_count=count,valid_frames=int(valid.sum()),
        assumptions=['Endpoint stationary box rests flat on the estimated floor','Box is rigid cuboid with planar textured front',
                     'Metric scale comes from estimated human body, not measured height'],
        source_human_sha256=sha256(args.human),median_pose_reprojection_rmse_px=float(np.median(errors)),
        note='Planar RGB initialization. Small reprojection error does not establish hidden-face accuracy or contact correctness.'),indent=2))
    args.output.with_suffix('.tracks.json').write_text(json.dumps(observations))
    print('Saved:',args.output,flush=True)


if __name__=='__main__': main()
