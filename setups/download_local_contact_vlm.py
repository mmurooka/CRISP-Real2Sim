#!/usr/bin/env python3
"""Download pinned public Qwen3-VL weights; never upload images or use an account."""
import argparse
import fnmatch
import json
import os
from pathlib import Path

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
REPO = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "README.md", "LICENSE*"]


def write_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/downloads/qwen3-vl-4b-instruct"))
    args = parser.parse_args()
    manifest_path = args.output / "download_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["repo_id"] != REPO or manifest["revision"] != REVISION:
            raise ValueError("Output belongs to a different model/revision")
        if manifest.get("complete") and all(
                (args.output / item["name"]).is_file() and
                (args.output / item["name"]).stat().st_size == item["size"]
                for item in manifest["files"]):
            print("Pinned model already complete; no network or duplicate download.")
            return
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(REPO, revision=REVISION, files_metadata=True)
    files = [dict(name=item.rfilename, size=item.size) for item in info.siblings
             if any(fnmatch.fnmatch(item.rfilename, pattern) for pattern in PATTERNS)]
    if not files or any(item["size"] is None for item in files):
        raise ValueError("Missing model file sizes")
    manifest = dict(repo_id=REPO, revision=info.sha, files=files,
                    total_bytes=sum(item["size"] for item in files), complete=False)
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, manifest)
    print(f"Public model: {REPO}, revision {info.sha}, {manifest['total_bytes']/1e9:.2f} GB", flush=True)
    snapshot_download(REPO, revision=info.sha, local_dir=str(args.output),
                      allow_patterns=PATTERNS, max_workers=2, token=False)
    for item in files:
        if (args.output / item["name"]).stat().st_size != item["size"]:
            raise ValueError(f"Incomplete file: {item['name']}")
    manifest["complete"] = True
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
