#!/usr/bin/env python3
"""Refine a fixed-size cuboid using visible silhouettes, depth and floor support.

Contact labels are deliberately absent: this geometric stage can be evaluated
independently of the later contact estimator. Depth is a soft monocular prior.
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from export_hoi_human import sha256


def boundary_samples(poly,count=96):
    delta=np.roll(poly,-1,axis=0)-poly
    lengths=np.linalg.norm(delta,axis=1)
    cumulative=np.r_[0,np.cumsum(lengths)]
    locations=np.arange(count)/count*cumulative[-1]
    ids=np.minimum(np.searchsorted(cumulative,locations,side='right')-1,len(poly)-1)
    fraction=(locations-cumulative[ids])/np.maximum(lengths[ids],1e-9)
    return poly[ids]+fraction[:,None]*delta[ids]


def polygon_distance(points,polygon):
    delta=np.roll(polygon,-1,axis=0)-polygon
    fraction=np.sum((points[:,None]-polygon)*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),1e-9)
    nearest=polygon+np.clip(fraction,0,1)[...,None]*delta
    return np.linalg.norm(points[:,None]-nearest,axis=-1).min(axis=1)


def ray_box_depth(rays,rotation,translation,half):
    origin=-translation@rotation
    direction=rays@rotation
    safe=np.where(np.abs(direction)>1e-9,direction,np.where(direction>=0,1e-9,-1e-9))
    a=(-half-origin)/safe;b=(half-origin)/safe
    near=np.minimum(a,b).max(axis=1);far=np.maximum(a,b).min(axis=1)
    return near,(far>=near)&(near>0)


def bounded_depth_residual(error,scale=.15,cutoff=3.):
    """Geman-McClure residual: gross monocular-depth errors cannot dominate RGB."""
    normalized=np.asarray(error)/scale
    return normalized/np.sqrt(1+(normalized/cutoff)**2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--human',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--maximum-speed-mps',type=float,default=4.,
        help='Conservative translation trust region; prevents monocular depth spikes from teleporting the box')
    parser.add_argument('--maximum-angular-speed-rad-s',type=float,default=3.5,
        help='Local rotation trust region; a cuboid silhouette alone has 180-degree symmetries')
    args=parser.parse_args()
    if not (np.isfinite(args.maximum_speed_mps) and args.maximum_speed_mps>0 and
            np.isfinite(args.maximum_angular_speed_rad_s) and args.maximum_angular_speed_rad_s>0):
        raise ValueError('Motion trust-region speeds must be finite and positive')
    if args.output.exists():raise FileExistsError(args.output)
    with np.load(args.initial) as a:data={k:a[k] for k in a.files}
    initial=data['world_from_object'];half=data['box_half_extents'];normal=data['floor_normal_world'];offset=float(data['floor_offset_m'])
    with np.load(args.scene) as a:depths=a['depths'];masks=a['object_mask'];persons=a['person_mask'];k=a['intrinsic']
    with np.load(args.human) as a:cameras=a['world_from_camera']
    meta=json.loads(args.initial.with_suffix('.json').read_text())
    floor_supported={v['frame'] for v in meta['calibration']}
    local=np.array(list(itertools.product([-1,1],repeat=3)))*half
    rotations=[];translations=[];poses=[];quality=[]
    for i,(mask,person,camera) in enumerate(zip(masks,persons,cameras)):
        h,w=mask.shape
        contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        target_poly=cv2.convexHull(max(contours,key=cv2.contourArea)).reshape(-1,2).astype(float)
        target_boundary=boundary_samples(target_poly)
        person_dilated=cv2.dilate(person.astype(np.uint8),np.ones((5,5),np.uint8)).astype(float)
        visible_weights=1-map_coordinates(person_dilated,target_boundary[:,::-1].T,order=1,mode='nearest')
        filled=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(filled,target_poly.astype(int),1)
        sdf=cv2.distanceTransform(1-filled,cv2.DIST_L2,5)-cv2.distanceTransform(filled,cv2.DIST_L2,5)
        eroded=cv2.erode((mask&~person).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        yy,xx=np.nonzero(eroded&(depths[i]>0)&np.isfinite(depths[i]))
        ids=np.linspace(0,len(xx)-1,min(96,len(xx)),dtype=int);yy,xx=yy[ids],xx[ids]
        rays=np.c_[xx,yy,np.ones(len(xx))]@np.linalg.inv(k).T
        observed_depth=depths[i,yy,xx]
        initial_camera=np.linalg.inv(camera)@initial[i]
        initial_rotation=initial_camera[:3,:3];initial_translation=initial_camera[:3,3]
        if i:
            predicted_translation=translations[-1]+(translations[-1]-translations[-2] if i>1 else 0)
            start=np.r_[np.zeros(3),predicted_translation]
        else:
            predicted_translation=initial_translation
            start=np.r_[np.zeros(3),initial_translation]
        rotation_base=rotations[-1] if i else initial_rotation
        def evaluate(x,details=False):
            rotation=rotation_base@Rotation.from_rotvec(x[:3]).as_matrix();translation=x[3:]
            corners=local@rotation.T+translation
            homogeneous=corners@k.T
            uv=homogeneous[:,:2]/np.maximum(homogeneous[:,2:],.05)
            hull_ids=cv2.convexHull(uv.astype(np.float32),returnPoints=False)[:,0]
            # Keep projection precision for finite-difference pose Jacobians.
            hull=uv[hull_ids]
            projected_boundary=boundary_samples(hull)
            observed_error=polygon_distance(target_boundary,hull)*visible_weights
            projected_occluded=map_coordinates(person_dilated,projected_boundary[:,::-1].T,order=1,mode='constant',cval=0)
            projected_error=map_coordinates(sdf,projected_boundary[:,::-1].T,order=1,mode='nearest')*(1-projected_occluded)
            predicted_depth,hit=ray_box_depth(rays,rotation,translation,half)
            depth_error=np.where(hit,bounded_depth_residual(predicted_depth-observed_depth),0)
            world_corners=corners@camera[:3,:3].T+camera[:3,3]
            height=world_corners@normal-offset
            penetration=np.minimum(height+.005,0)/.01
            support=(height[local[:,1]<0]/.015 if i in floor_supported else np.zeros(4))
            # Weak rotation initialization; stronger local continuity after frame 0.
            rot_reference=rotations[-1] if i else initial_rotation
            rotation_error=Rotation.from_matrix(rot_reference.T@rotation).as_rotvec()/.12
            translation_error=(translation-predicted_translation)/(.04 if i else .3)
            residual=np.r_[observed_error/1.5,projected_error/1.5,depth_error,
                           penetration,support,rotation_error,translation_error,np.minimum(corners[:,2]-.2,0)*100]
            if details:
                return dict(visible_boundary_rmse_px=float(np.sqrt(np.mean(observed_error[visible_weights>.5]**2))),
                    visible_depth_median_error_m=float(np.median(np.abs(predicted_depth[hit]-observed_depth[hit]))) if hit.any() else None,
                    depth_ray_hit_fraction=float(hit.mean()),minimum_floor_height_m=float(height.min()),
                    depth_outlier_fraction=float(np.mean(np.abs(predicted_depth-observed_depth)>.45)),
                    floor_supported_assumption=i in floor_supported)
            return residual
        lower=np.full(6,-np.inf);upper=np.full(6,np.inf)
        if i:
            dt=float(data['timestamps_s'][i]-data['timestamps_s'][i-1])
            step=args.maximum_speed_mps*dt
            # Bound each translation component relative to the last pose, not
            # an extrapolation that could amplify an earlier depth failure.
            lower[3:]=translations[-1]-step;upper[3:]=translations[-1]+step
            angle_step=args.maximum_angular_speed_rad_s*dt/np.sqrt(3)
            lower[:3]=-angle_step;upper[:3]=angle_step
        start=np.clip(start,lower+1e-8,upper-1e-8)
        fit=least_squares(evaluate,start,bounds=(lower,upper),loss='soft_l1',f_scale=1.,max_nfev=80,diff_step=1e-4,
                          xtol=1e-6,ftol=1e-6,gtol=1e-6)
        # A second, independently tracked image initialization handles local
        # minima at changes of visible faces. Select using the same objective.
        candidate=np.clip(np.r_[Rotation.from_matrix(rotation_base.T@initial_rotation).as_rotvec(),initial_translation],lower+1e-8,upper-1e-8)
        alternative=least_squares(evaluate,candidate,bounds=(lower,upper),loss='soft_l1',f_scale=1.,max_nfev=80,diff_step=1e-4,
                                 xtol=1e-6,ftol=1e-6,gtol=1e-6)
        if alternative.cost<fit.cost:fit=alternative
        rotation=rotation_base@Rotation.from_rotvec(fit.x[:3]).as_matrix();translation=fit.x[3:]
        pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=translation
        poses.append(camera@pose);rotations.append(rotation);translations.append(translation)
        q=evaluate(fit.x,True);q.update(frame=i,optimizer_success=bool(fit.success),cost=float(fit.cost),nfev=fit.nfev)
        quality.append(q)
        if i%20==0: print(i,q,flush=True)
    data['world_from_object']=np.array(poses)
    data['valid']=np.array([q['visible_boundary_rmse_px']<3 and q['depth_ray_hit_fraction']>.8 and q['minimum_floor_height_m']>-.04 for q in quality])
    data['shape_assumption']=np.asarray('rigid_cuboid_floor_calibration_and_visible_silhouette_depth_refinement')
    data.pop('pose_reprojection_rmse_px',None)
    data['visible_boundary_rmse_px']=np.array([q['visible_boundary_rmse_px'] for q in quality])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**data)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        source_initial_sha256=sha256(args.initial),source_scene_sha256=sha256(args.scene),
        dimensions_m=(2*half).tolist(),maximum_translation_component_speed_mps=args.maximum_speed_mps,
        maximum_angular_speed_rad_s=args.maximum_angular_speed_rad_s,
        depth_loss='Geman-McClure residual, 0.15 m scale, 3 sigma saturation',
        valid_frames=int(data['valid'].sum()),frame_count=len(poses),frames=quality,
        interpretation='Geometric initialization without human contact constraints. Monocular depth, floor support, rigid cuboid and temporal continuity are assumptions; image agreement is not independent 3D ground truth.'),indent=2))
    print('Saved:',args.output,flush=True)


if __name__=='__main__':main()
