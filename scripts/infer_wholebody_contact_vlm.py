#!/usr/bin/env python3
"""Offline image-level whole-body contact hypotheses; never contact ground truth.

Uses a local Qwen3-VL checkpoint. Raw language output and image/model provenance
are retained independently of the existing InteractVLM vertex predictions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# Set before importing any Hugging Face packages. Images stay on this machine.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

PROMPT = """Examine the person's WHOLE BODY and the cardboard box in this image.
Identify body regions that appear to physically touch this box. Consider all
visible regions, not only hands. Do not confuse touching the floor, 2D overlap,
or merely being near the box with touching the box. Hidden contact is uncertain.
Use anatomical left/right only if visible evidence establishes it; otherwise
use unknown and describe the limb's pose. Do not assume any particular contact.
Return only JSON with this structure:
{"contacts": [{"region": "hand|forearm|upper_arm|chest|abdomen|pelvis|thigh|knee|shin|foot|head",
"side": "left|right|both|midline|unknown", "state": "likely|uncertain",
"body_location": "short description", "box_location": "short description",
"visual_evidence": "short description of visible touch or ambiguity"}],
"clearly_separated_regions": ["short descriptions"],
"limitations": "brief description of occlusion or ambiguity"}
Keep the response concise. If a boundary between body regions is unclear, say
so rather than inventing a precise location. These are image hypotheses, not
measurements of contact force.
"""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def parse_response(raw):
    """Parse without repairing or silently changing model assertions."""
    value = raw.strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("contacts"), list):
            raise ValueError("Missing contact list")
        regions = {"hand", "forearm", "upper_arm", "chest", "abdomen", "pelvis",
                   "thigh", "knee", "shin", "foot", "head"}
        for contact in parsed["contacts"]:
            if (contact.get("region") not in regions or
                    contact.get("side") not in {"left", "right", "both", "midline", "unknown"} or
                    contact.get("state") not in {"likely", "uncertain"}):
                raise ValueError("Unexpected contact category")
        return parsed, None
    except (ValueError, TypeError, AttributeError) as error:
        return None, str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=672 * 672)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (min(args.frame_indices) < 0 or args.max_pixels < 32 * 32 or
            args.max_new_tokens <= 0 or args.batch_size < 1):
        raise ValueError("Invalid frame or generation bounds")
    if len(set(args.frame_indices)) != len(args.frame_indices):
        raise ValueError("Frame indices must be unique")
    model_path = args.model.resolve(strict=True)
    manifest_path = model_path / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise ValueError("Model download is not complete")
    for item in manifest["files"]:
        if (model_path / item["name"]).stat().st_size != item["size"]:
            raise ValueError(f"Incomplete model file: {item['name']}")
    paths = [args.frames / f"{index:05d}.jpg" for index in args.frame_indices]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local inference")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    args.output.mkdir(parents=True)
    metadata = dict(
        complete=False, model_path=str(model_path), model_manifest=manifest,
        model_manifest_sha256=sha256(manifest_path),
        model_weight_sha256={p.name: sha256(p) for p in sorted(model_path.glob("*.safetensors"))},
        script_sha256=sha256(__file__), prompt=PROMPT, frame_indices=args.frame_indices,
        frames=str(args.frames.resolve()), max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        do_sample=False, dtype="bfloat16",
        attention="sdpa", torch_version=torch.__version__, transformers_version=transformers.__version__,
        gpu=torch.cuda.get_device_name(), network_inference=False,
        interpretation="Uncalibrated image-level hypotheses; no force, vertex correspondence, or ground-truth claim.")
    crop_metadata = args.frames / "crop_metadata.json"
    if crop_metadata.exists():
        metadata["crop_metadata_sha256"] = sha256(crop_metadata)
        metadata["source_video_sha256"] = json.loads(crop_metadata.read_text())["source_video_sha256"]
    write_json(args.output / "metadata.json", metadata)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False)
    # AutoProcessor in transformers 4.57 does not forward max_pixels to this
    # fast image processor. Set its actual resize configuration explicitly.
    processor.image_processor.size = dict(
        shortest_edge=min(65536, args.max_pixels), longest_edge=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    metadata["image_processor_size"] = processor.image_processor.size.copy()
    write_json(args.output / "metadata.json", metadata)
    torch.cuda.reset_peak_memory_stats()
    for offset in range(0, len(paths), args.batch_size):
        started = time.monotonic()
        batch_paths = paths[offset:offset+args.batch_size]
        batch_indices = args.frame_indices[offset:offset+args.batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        messages = [[{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
            for image in images]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            padding=True, return_dict=True, return_tensors="pt").to(model.device)
        grid = inputs.image_grid_thw.cpu().numpy()
        patch_size = processor.image_processor.patch_size
        processed_pixels = int(max(row[1] * row[2] * patch_size**2 for row in grid))
        if processed_pixels > args.max_pixels:
            raise RuntimeError("Image processor ignored the requested pixel limit")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        ids = generated[:, inputs.input_ids.shape[1]:]
        responses = processor.batch_decode(ids, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        elapsed = time.monotonic()-started
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for row, (index, path, image, raw) in enumerate(zip(batch_indices, batch_paths, images, responses)):
            tokens = ids[row].tolist()
            stop = next((i for i, token in enumerate(tokens) if token in eos), None)
            token_count = len(tokens) if stop is None else stop+1
            parsed, error = parse_response(raw)
            result = dict(frame_index=index, image_path=str(path.resolve()), image_sha256=sha256(path),
                          image_size=list(image.size), raw_response=raw, parsed=parsed,
                          parse_error=error, generated_tokens=token_count,
                          reached_token_limit=stop is None and token_count >= args.max_new_tokens,
                          elapsed_s=elapsed, batch_frame_indices=batch_indices,
                          amortized_elapsed_s=elapsed/len(batch_indices),
                          image_grid_thw=[grid[row].tolist()],
                          processed_image_pixels=int(grid[row, 1]*grid[row, 2]*patch_size**2),
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
            write_json(args.output / f"{index:05d}.json", result)
            print(json.dumps(dict(frame=index, batch_elapsed_s=elapsed,
                                  contacts=parsed["contacts"] if parsed else None,
                                  parse_error=error), ensure_ascii=False), flush=True)
        del inputs, generated, ids
    metadata.update(complete=True, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(args.output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
