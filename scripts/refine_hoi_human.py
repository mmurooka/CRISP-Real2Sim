#!/usr/bin/env python3
"""Refine SMPL-X pose against observed keypoints and a fixed reconstructed box.

This is conditional human refinement, not a joint human/object optimizer. The
geometry-only ablation adds nonpenetration but never invents contact attraction.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from export_hoi_human import sha256


def cuboid_signed_distance(points,half):
    q=points.abs()-half
    return torch.linalg.vector_norm(q.clamp_min(0),dim=-1)+q.amax(dim=-1).clamp_max(0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('human','object','hmr-result','observations','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    prior=parser.add_mutually_exclusive_group(required=True)
    prior.add_argument('--contacts',type=Path)
    prior.add_argument('--geometry-only',action='store_true')
    parser.add_argument('--prism-root',type=Path,default=Path(__file__).resolve().parents[2]/'PRISM')
    parser.add_argument('--iterations',type=int,default=160)
    parser.add_argument('--batch-size',type=int,default=12)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'prep/HMR'));sys.path.insert(0,str(args.prism_root))
    from hmr4d.utils.body_model.body_model_smplx import BodyModelSMPLX
    from prism_reconstruction.adapters import load_interactvlm_contacts
    with np.load(args.human) as a:human=dict(a)
    with np.load(args.object) as a:box=dict(a)
    for field in ('world_frame','source_video_sha256','length_unit'):
        if str(human[field])!=str(box[field]):raise ValueError(f'Mismatched {field}')
    if not np.all(box['valid']):raise ValueError('Object image/floor checks have failed frames')
    if not np.allclose(human['timestamps_s'],box['timestamps_s']):raise ValueError('Mismatched timestamps')
    if sha256(args.hmr_result)!=str(human['source_hmr_sha256']):raise ValueError('HMR parameter provenance mismatch')
    obs_meta=json.loads((args.observations.parent/'metadata.json').read_text())
    if obs_meta['source_video_sha256']!=str(human['source_video_sha256']):raise ValueError('Observation provenance mismatch')
    with np.load(args.observations) as a:observations=a['keypoints_coco17'].copy()
    device='cuda';count=len(human['timestamps_s'])
    def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device=device)
    params={k:v.to(device) for k,v in torch.load(args.hmr_result,map_location='cpu',weights_only=False)['smpl_params_incam'].items()}
    if len(observations)!=count:raise ValueError('Observation count mismatch')
    body=BodyModelSMPLX(model_path=str(root/'prep/HMR/inputs/checkpoints/body_models'),model_type='smplx',
        gender='neutral',num_pca_comps=12,flat_hand_mean=False).to(device).eval()
    body.requires_grad_(False)
    mapping=torch.load(root/'prep/HMR/hmr4d/utils/body_model/smplx2smpl_sparse.pt',map_location=device,weights_only=False)
    coco=torch.load(root/'prep/HMR/hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',map_location=device,weights_only=False)
    camera=tensor(human['world_from_camera']);poses=tensor(box['world_from_object']);half=tensor(box['box_half_extents'])
    normal=tensor(box['floor_normal_world']);floor=float(box['floor_offset_m'])
    k=tensor(box['intrinsic_full']);keypoints=tensor(observations)
    scores=torch.zeros((count,6890),device=device)
    if args.contacts:
        scores=tensor(load_interactvlm_contacts(args.contacts,human['frame_indices'],source_video_sha256=str(human['source_video_sha256'])))
        # Learned contacts far from the independently estimated object are
        # unsupported, not permission to drag a limb across the whole scene.
        original=tensor(human['human_vertices_world'])
        local=(original-poses[:,None,:3,3])@poses[:,:3,:3]
        scores*=cuboid_signed_distance(local,half).abs()<.30
        del original,local
    delta_pose=torch.nn.Parameter(torch.zeros_like(params['body_pose']))
    delta_orient=torch.nn.Parameter(torch.zeros_like(params['global_orient']))
    delta_trans=torch.nn.Parameter(torch.zeros_like(params['transl']))
    optimizer=torch.optim.Adam([{'params':[delta_pose],'lr':.008},
        {'params':[delta_orient],'lr':.004},{'params':[delta_trans],'lr':.002}])
    history=[];best=None;best_loss=float('inf')

    def forward(begin,end):
        prediction=body(betas=params['betas'][begin:end],body_pose=params['body_pose'][begin:end]+delta_pose[begin:end],
            global_orient=params['global_orient'][begin:end]+delta_orient[begin:end],transl=params['transl'][begin:end]+delta_trans[begin:end])
        # One sparse multiplication over batch/coordinate columns.
        raw=prediction.vertices
        vertices=torch.sparse.mm(mapping,raw.permute(1,0,2).reshape(raw.shape[1],-1)).reshape(6890,end-begin,3).permute(1,0,2)
        world=vertices@camera[begin:end,:3,:3].transpose(1,2)+camera[begin:end,None,:3,3]
        joints=prediction.joints[:,:22]@camera[begin:end,:3,:3].transpose(1,2)+camera[begin:end,None,:3,3]
        coco_joints=torch.einsum('jv,bvc->bjc',coco,vertices)
        projected=coco_joints@k.T
        uv=projected[...,:2]/projected[...,2:].clamp_min(.2)
        gap=cuboid_signed_distance((world-poses[begin:end,None,:3,3])@poses[begin:end,:3,:3],half)
        return world,joints,uv,gap

    for iteration in range(args.iterations):
        optimizer.zero_grad();metrics=dict(reprojection=0.,penetration=0.,floor=0.,contact=0.)
        for begin in range(0,count,args.batch_size):
            end=min(count,begin+args.batch_size);weight=(end-begin)/count
            world,joints,uv,gap=forward(begin,end)
            confidence=keypoints[begin:end,:,2].clamp(0,1)
            reprojection=(((uv-keypoints[begin:end,:,:2])/8).square().sum(-1)*confidence).mean()
            penetration=(torch.relu(-gap-.005)/.015).square().mean()
            floor_loss=(torch.relu(floor-world@normal-.015)/.015).square().mean()
            contact_weight=scores[begin:end]
            contact=((gap/.03).square().clamp_max(100)*contact_weight).sum()/contact_weight.sum().clamp_min(1)
            loss=weight*(reprojection+8*penetration+2*floor_loss+contact)
            loss.backward()
            for name,value in zip(metrics,(reprojection,penetration,floor_loss,contact)):
                metrics[name]+=weight*float(value.detach())
        regularizer=.10*(delta_pose/.15).square().mean()+.10*(delta_orient/.10).square().mean()+.10*(delta_trans/.08).square().mean()
        regularizer+=.15*(torch.diff(delta_pose,dim=0)/.05).square().mean()+.15*(torch.diff(delta_trans,dim=0)/.01).square().mean()
        regularizer.backward()
        total=metrics['reprojection']+8*metrics['penetration']+2*metrics['floor']+metrics['contact']+float(regularizer.detach())
        metrics.update(iteration=iteration,total=total,regularizer=float(regularizer.detach()))
        history.append(metrics)
        if not np.isfinite(total):raise FloatingPointError('Nonfinite refinement loss')
        if total<best_loss:
            best_loss=total;best=[x.detach().clone() for x in (delta_pose,delta_orient,delta_trans)]
        torch.nn.utils.clip_grad_norm_([delta_pose,delta_orient,delta_trans],10.)
        optimizer.step()
        with torch.no_grad():
            delta_pose.clamp_(-.6,.6);delta_orient.clamp_(-.3,.3);delta_trans.clamp_(-.25,.25)
        if iteration%10==0:print(json.dumps(metrics),flush=True)
    with torch.no_grad():
        for parameter,value in zip((delta_pose,delta_orient,delta_trans),best):parameter.copy_(value)
        vertices=[];joints=[];errors=[];gaps=[]
        for begin in range(0,count,args.batch_size):
            end=min(count,begin+args.batch_size);v,j,uv,gap=forward(begin,end)
            vertices.append(v.cpu().numpy());joints.append(j.cpu().numpy());gaps.append(gap.cpu().numpy())
            errors.append(torch.linalg.vector_norm(uv-keypoints[begin:end,:,:2],dim=-1).cpu().numpy())
        human['human_vertices_world']=np.concatenate(vertices);human['human_joints_world']=np.concatenate(joints)
        optimized={k:v.detach().cpu() for k,v in params.items()}
        for key,delta in [('body_pose',delta_pose),('global_orient',delta_orient),('transl',delta_trans)]:optimized[key]+=delta.cpu()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    parameters_file=args.output.with_suffix('.pt');torch.save(dict(smpl_params_incam=optimized),parameters_file)
    human['source_original_hmr_sha256']=human['source_hmr_sha256'];human['source_hmr_sha256']=np.asarray(sha256(parameters_file))
    human['refinement_stage']=np.asarray('human_pose_given_fixed_object')
    human['contact_prior_source']=np.asarray('interactvlm' if args.contacts else 'none_nonpenetration_only')
    np.savez_compressed(args.output,**human)
    gap=np.concatenate(gaps);error=np.concatenate(errors);visible=observations[:,:,2]>.5
    metadata=dict(source_human_sha256=sha256(args.human),source_object_sha256=sha256(args.object),
        source_observations_sha256=sha256(args.observations),iterations=args.iterations,best_loss=best_loss,
        contact_prior_source=str(human['contact_prior_source']),history=history,
        visible_keypoint_error_median_px=float(np.median(error[visible])),visible_keypoint_error_p95_px=float(np.quantile(error[visible],.95)),
        vertices_penetrating_2cm=int((gap<-.02).sum()),frames_penetrating_2cm=int((gap<-.02).any(axis=1).sum()),
        max_penetration_m=float(max(0,-gap.min())),
        interpretation='Conditional human-pose fit with fixed object geometry. Reduced penetration is an optimization objective, not independent evidence of correct contact.')
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
