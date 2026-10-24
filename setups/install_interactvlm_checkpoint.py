#!/usr/bin/env python3
"""Install a completed, user-downloaded InteractVLM ZIP without network access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    archive=args.archive.resolve()
    if not archive.is_file() or archive.suffix!='.zip':raise ValueError('A completed .zip file is required')
    staging=root/'data/downloads/extracted'/archive.stem
    manifest=staging/'installation.json'
    model_name='interactvlm-3d-hcontact-wScene-damon-lemon-rich'
    model=staging/'trained_models'/model_name
    digest=hashlib.sha256()
    with archive.open('rb') as handle:
        for block in iter(lambda:handle.read(16*1024*1024),b''):digest.update(block)
    checksum=digest.hexdigest()
    if manifest.exists():
        if json.loads(manifest.read_text())['archive_sha256']!=checksum:raise ValueError('Different archive already installed')
    else:
        with zipfile.ZipFile(archive) as z:
            members=z.infolist()
            for member in members:
                destination=(staging/member.filename).resolve()
                if not destination.is_relative_to(staging.resolve()):raise ValueError('Unsafe ZIP path')
                if stat.S_ISLNK(member.external_attr>>16):raise ValueError('ZIP symlinks are not accepted')
            size=sum(m.file_size for m in members)
            if shutil.disk_usage(root).free<size+1024**3:raise RuntimeError('Insufficient space for checkpoint extraction')
            staging.mkdir(parents=True,exist_ok=True)
            for member in members:
                z.extract(member,staging)  # ZipFile checks each file CRC while extracting.
            if not (model/'config.json').is_file():raise ValueError('Expected InteractVLM model configuration missing')
            for index in model.glob('*.index.json'):
                names=set(json.loads(index.read_text()).get('weight_map',{}).values())
                if any(not (model/name).is_file() for name in names):raise ValueError('Incomplete checkpoint shards')
            manifest.write_text(json.dumps(dict(archive_sha256=checksum,uncompressed_bytes=size,file_count=len(members),
                installed_model=str(model),complete=True),indent=2))
    destination=root/'prep/Contact-Predictor/trained_models'/model_name
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_symlink() and destination.resolve()==model.resolve():pass
    elif destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    else:destination.symlink_to(model,target_is_directory=True)
    print(destination,flush=True)


if __name__=='__main__':main()
