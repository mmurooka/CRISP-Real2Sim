#!/usr/bin/env python3
"""Export a validated, local HOI result bundle without merging evidence types."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def export_bundle(human,objects,contact_input,contacts,image_cues,g1,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    h=np.load(human);o=np.load(objects);inp=np.load(contact_input);c=np.load(contacts/'contacts.npz');i=np.load(image_cues/'cues.npz');r=np.load(g1/'trajectory.npz')
    cm=json.loads((contacts/'metadata.json').read_text());im=json.loads((image_cues/'metadata.json').read_text());rm=json.loads((g1/'metadata.json').read_text())
    video=str(h['source_video_sha256']);times=h['timestamps_s'];frames=len(times)
    if len(times)<3 or not np.all(np.diff(times)>0):raise ValueError('Strictly increasing time samples required')
    if any(str(x)!=video for x in (o['source_video_sha256'],inp['source_video_sha256'],cm['source_video_sha256'],im['source_video_sha256'],rm['source_video_sha256'])):raise ValueError('Source videos differ')
    for archive in (o,inp,c,i,r):
        if not np.array_equal(times,archive['timestamps_s']):raise ValueError('Frame timestamps differ')
    if not np.array_equal(h['frame_indices'],r['frame_indices']):raise ValueError('Human and robot source frames differ')
    if cm['input_sha256']!=digest(contact_input) or im['source_contact_input_sha256']!=digest(contact_input):raise ValueError('Contact input provenance differs')
    if im['source_human_sha256']!=digest(human):raise ValueError('Image cue human provenance differs')
    for key,source in (('source_human_sha256',human),('source_object_sha256',objects),('source_contacts_sha256',contacts/'contacts.npz')):
        if rm[key]!=digest(source):raise ValueError('G1 source reconstruction provenance differs')
    if not np.array_equal(h['human_vertices_world'],inp['human_vertices_world']):raise ValueError('Human vertices differ')
    for archive in (inp,c):
        for name in ('world_from_object','box_half_extents'):
            if not np.array_equal(o[name],archive[name]):raise ValueError('Object geometry differs')
    if str(h['world_frame'])!=str(o['world_frame']) or str(h['world_frame'])!=cm['world_frame']:raise ValueError('Reconstruction world frames differ')
    if c['active'].shape!=h['human_vertices_world'].shape[:2] or i['corroborated_candidate'].shape!=c['active'].shape:raise ValueError('Human surface indexing differs')
    if len(r['joint_names'])!=29 or r['qpos'].shape!=(frames,43):raise ValueError('Expected free G1 29DoF plus free box')
    source_files={
        'human_geometry.npz':human,'object_geometry.npz':objects,'contact_input.npz':contact_input,
        'contacts.npz':contacts/'contacts.npz','contact_metadata.json':contacts/'metadata.json','force_evidence.json':contacts/'force_evidence.json',
        'image_cues.npz':image_cues/'cues.npz','image_cue_metadata.json':image_cues/'metadata.json',
        'g1_kinematic_trajectory.npz':g1/'trajectory.npz','g1_contact_targets.json':g1/'diagnostics.json','g1_metadata.json':g1/'metadata.json'}
    manifest=dict(schema='local_whole_body_hoi_bundle_v1',source_video_sha256=video,frames=frames,
        start_time_s=float(times[0]),last_time_s=float(times[-1]),human_surface_vertices=h['human_vertices_world'].shape[1],
        human_topology=str(h['body_topology']),human_world_frame=str(h['world_frame']),length_unit='m',
        scale_source=str(h['scale_source']),measured_person_height=False,measured_box_size=False,
        force_unit=cm['force_unit'],torque_unit=cm['torque_unit'],object_mass_measured=False,
        evidence_code_bits=cm['evidence_code_bits'],contact_mode_codes=cm['contact_mode_codes'],
        g1_joint_names=r['joint_names'].tolist(),g1_frame_description='Separate floor-aligned robot frame; transform/body_scale and preserved box extents are stored in the trajectory.',
        g1_qpos_order=['robot position xyz','robot quaternion wxyz','29 joint positions in joint_names order','box position xyz','box quaternion wxyz'],
        g1_is_kinematic_reference=True,validated_full_clip_physics_controller=False,safe_for_robot_execution=False,
        limitations=['Single monocular video, unknown metric scale and box mass.',
            'Contact evidence is not measured ground truth. Force feasibility does not uniquely identify body parts.',
            'Image boundary cues may arise from occlusion and remain separate from contact labels.',
            'G1 trajectory is a geometric reference; no trained policy or execution command is included.',
            'Robot mesh assets and licensed foundation models are not bundled.'],
        export_code_sha256=digest(__file__),files={})
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hoi-export-',dir=output.parent) as temp:
        staging=Path(temp)/'bundle';staging.mkdir()
        for name,source in source_files.items():
            shutil.copyfile(source,staging/name);checksum=digest(staging/name)
            if checksum!=digest(source):raise IOError('Copy checksum mismatch')
            manifest['files'][name]=dict(source=str(source.resolve()),sha256=checksum,bytes=(staging/name).stat().st_size)
        (staging/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
        (staging/'README.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><title>全身接触HOI結果データ</title><style>body{max-width:900px;margin:40px auto;font:17px/1.8 sans-serif}code{background:#eee}</style><h1>全身接触HOI・結果データ</h1>
<p>人・箱・接触候補を保存したローカル研究用データです。<a href="manifest.json">manifest.json</a>にフレーム数、ファイルのSHA256、座標系、単位、制限を記録しています。</p>
<ul><li><code>human_geometry.npz</code>：SMPL対応の人体頂点・関節・カメラ。</li><li><code>object_geometry.npz</code>：箱の6DoF軌道、半寸法、床面。</li><li><code>contacts.npz</code>：頂点ごとの接触候補、距離、物体座標の接触位置、根拠ビット。</li><li><code>force_evidence.json</code>：力の説明と、仮定の下での必要性。実測力ではありません。</li><li><code>image_cues.npz</code>：画像輪郭による補助候補。遮蔽との区別は保証しません。</li><li><code>g1_kinematic_trajectory.npz</code>：29関節と自由なロボット・箱の参照軌道。</li><li><code>g1_contact_targets.json</code>：物体座標の接触目標とロボット表面の対応、元の接触位置からの調整量。</li></ul>
<p>NPZは <code>numpy.load(path, allow_pickle=False)</code> で読めます。時刻は <code>timestamps_s</code>、人体の表面番号は全ファイルで共通です。<code>active</code>だけを見るとVLMが見逃した脚候補は含まれないため、<code>contact_evidence_code</code>と力・画像の補助情報を区別して参照してください。</p><p><b>G1軌道は実機実行用ではありません。動画全体を接触維持しながら再現する物理制御器は未検証です。</b></p></html>''')
        staging.rename(output)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('human','objects','contact-input','contacts','image-cues','g1','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();m=export_bundle(args.human,args.objects,args.contact_input,args.contacts,args.image_cues,args.g1,args.output)
    print(json.dumps(dict(output=str(args.output),frames=m['frames'],files=len(m['files']),source_video_sha256=m['source_video_sha256'])))

if __name__=='__main__':main()
