#!/usr/bin/env python3
"""Intersect offline language hypotheses with image/3D contact candidates.

This is a separate evidence supplement. It does not replace InteractVLM vertex
predictions, identify contact forces, or turn language confidence into probability.
"""
import argparse
import copy
import json
from pathlib import Path
import re

import numpy as np

from infer_wholebody_contact_vlm import parse_response, sha256, write_json

NEGATED_CONTACT = ("not touching", "no contact", "no clear contact",
                   "no direct contact", "contact is ambiguous", "contact is uncertain")


def conservative_contacts(contacts):
    """Downgrade self-contradictory likely labels; never upgrade a label.

    This conservative English-output consistency check is not a second visual
    classifier. For example, a 'likely' knee label with 'not touching the box'
    in its own explanation cannot count as positive contact evidence.
    """
    guarded = copy.deepcopy(contacts)
    flags = []
    for i, contact in enumerate(guarded):
        evidence = str(contact.get("visual_evidence", "")).lower()
        location = str(contact.get("box_location", "")).strip().lower()
        reasons = [phrase for phrase in NEGATED_CONTACT if phrase in evidence]
        if re.match(r"^(?:the )?(?:floor|ground)\b", location):
            reasons.append("contact target described as floor/ground")
        if contact["state"] == "likely" and reasons:
            contact["state"] = "uncertain"
            flags.append(dict(contact_index=i, reasons=reasons))
    return guarded, flags


def semantic_states(contacts, regions):
    """Laterality comes from geometry; raw language laterality remains in JSON.

    A language mention of either thigh proposes both anatomical thighs, after
    which the independent per-side image/geometry gate decides which survives.
    This deliberately does not claim that the VLM identified the correct side.
    """
    states = np.zeros(len(regions), np.int8)
    for contact in contacts:
        part = contact["region"]
        state = {"uncertain": 1, "likely": 2}[contact["state"]]
        names = ({"torso"} if part in {"chest", "abdomen"} else
                 {part} if part in {"pelvis", "head"} else
                 {"left_"+part, "right_"+part})
        for i, region in enumerate(regions):
            if region in names:
                states[i] = max(states[i], state)
    return states


def persistent_candidates(states, image_cues, minimum_frames):
    if states.shape != image_cues.shape or states.ndim != 2:
        raise ValueError("Frame/region arrays must match")
    if minimum_frames < 1:
        raise ValueError("Positive persistence length required")
    raw = (states == 2) & image_cues
    persistent = np.zeros_like(raw)
    for region in range(raw.shape[1]):
        changes = np.diff(np.r_[False, raw[:, region], False].astype(int))
        for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
            if last-first >= minimum_frames:
                persistent[first:last, region] = True
    return raw, persistent


def inference_image_path(metadata, frame, record):
    """Resolve an explicit per-frame input when combining documented runs."""
    if 'frame_image_paths' in metadata:
        path = Path(metadata['frame_image_paths'][str(frame)])
    else:
        path = Path(metadata['frames']) / f'{frame:05d}.jpg'
    if 'image_path' in record and Path(record['image_path']).resolve() != path.resolve():
        raise ValueError('Inference record/image manifest mismatch')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", type=Path, required=True)
    parser.add_argument("--image-cues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-frames", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    language_meta = json.loads((args.language / "metadata.json").read_text())
    cue_meta = json.loads((args.image_cues / "metadata.json").read_text())
    if not language_meta.get("complete"):
        raise ValueError("Language inference is incomplete")
    video_sha = cue_meta["source_video_sha256"]
    if language_meta.get("source_video_sha256") != video_sha:
        raise ValueError("Language/image cues refer to different videos")
    with np.load(args.image_cues / "cues.npz", allow_pickle=False) as source:
        times = source["timestamps_s"].copy()
        regions = source["region_names"].copy()
        vertex_regions = source["human_vertex_regions"].copy()
        image_regions = source["persistent_region_cue"].copy()
        image_vertices = source["corroborated_candidate"].copy()
    if sorted(language_meta["frame_indices"]) != list(range(len(times))):
        raise ValueError("Every image frame must have an inference record; sparse frames cannot establish persistence")
    states = np.full(image_regions.shape, -1, np.int8)
    raw_states = states.copy()
    valid = np.zeros(len(times), bool)
    records = []
    for frame in range(len(times)):
        path = args.language / f"{frame:05d}.json"
        record = json.loads(path.read_text())
        if record["frame_index"] != frame:
            raise ValueError("Language frame index mismatch")
        image_path = inference_image_path(language_meta, frame, record)
        if record["image_sha256"] != sha256(image_path):
            raise ValueError(f"Language image has changed: {image_path}")
        parsed, error = parse_response(record["raw_response"])
        flags = []
        if error is None and not record["reached_token_limit"]:
            raw_states[frame] = semantic_states(parsed["contacts"], regions)
            guarded, flags = conservative_contacts(parsed["contacts"])
            states[frame] = semantic_states(guarded, regions)
            valid[frame] = True
        records.append(dict(frame=frame, response_sha256=sha256(path), valid=bool(valid[frame]),
                            parse_error=error, token_limit=record["reached_token_limit"],
                            consistency_downgrades=flags))
    raw, persistent = persistent_candidates(states, image_regions, args.minimum_frames)
    vertices = np.zeros_like(image_vertices)
    for i, region in enumerate(regions):
        selected = vertex_regions == region
        vertices[:, selected] = image_vertices[:, selected] & persistent[:, i, None]
    summaries = {}
    for i, name in enumerate(regions):
        changes = np.diff(np.r_[False, persistent[:, i], False].astype(int))
        intervals = [dict(first_frame=int(first), last_frame=int(last-1),
                          start_s=float(times[first]), end_s=float(times[last-1]))
                     for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]
        summaries[str(name)] = dict(raw_language_likely_frames=int((raw_states[:, i] == 2).sum()),
                                   language_likely_frames=int((states[:, i] == 2).sum()),
                                   language_uncertain_frames=int((states[:, i] == 1).sum()),
                                   gated_frames=int(raw[:, i].sum()),
                                   persistent_frames=int(persistent[:, i].sum()), intervals=intervals)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "cues.npz", timestamps_s=times, region_names=regions,
                        human_vertex_regions=vertex_regions, valid_language_frame=valid,
                        raw_semantic_state=raw_states, semantic_state=states,
                        image_geometry_region_cue=image_regions,
                        raw_fused_region_candidate=raw, persistent_region_candidate=persistent,
                        fused_vertex_candidate=vertices, source_video_sha256=video_sha)
    metadata = dict(schema="language_image_geometry_contact_candidates_v1", complete=True,
                    source_video_sha256=video_sha, language=str(args.language.resolve()),
                    source_human_sha256=cue_meta["source_human_sha256"],
                    source_contact_input_sha256=cue_meta["source_contact_input_sha256"],
                    language_metadata_sha256=sha256(args.language / "metadata.json"),
                    language_input_provenance=language_meta.get('combined_run_provenance'),
                    image_cues=str(args.image_cues.resolve()),
                    image_cue_metadata_sha256=sha256(args.image_cues / "metadata.json"),
                    image_cues_sha256=sha256(args.image_cues / "cues.npz"),
                    code_sha256={Path(__file__).name: sha256(__file__),
                                 "infer_wholebody_contact_vlm.py": sha256(Path(__file__).with_name("infer_wholebody_contact_vlm.py"))},
                    minimum_frames=args.minimum_frames, frame_count=len(times),
                    valid_language_frames=int(valid.sum()), regions=summaries,
                    semantic_state_codes={"-1": "unusable response", "0": "not mentioned", "1": "uncertain", "2": "likely"},
                    laterality="Language left/right is ignored for gating; reconstructed anatomy and image/3D proximity choose sides. Raw responses retain the original side claims.",
                    anatomical_mapping="Chest/abdomen map to torso. Knee and thigh remain distinct.",
                    consistency_guard=dict(negated_contact_phrases=NEGATED_CONTACT,
                        floor_location_pattern=r"^(?:the )?(?:floor|ground)\b",
                        action="Downgrade likely to uncertain; preserve raw state and response. Conservative English-text heuristic, not another visual model."),
                    image_geometry_thresholds=cue_meta["thresholds"], response_records=records,
                    limitations=["Correlated image and language evidence, not independent measurements or calibrated probabilities.",
                                 "Silhouette overlap, occlusion and monocular reconstruction errors can still create false contacts.",
                                 "No force necessity claim; original InteractVLM and dynamics labels are unchanged."])
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps(dict(valid_frames=int(valid.sum()), regions=summaries), ensure_ascii=False))


if __name__ == "__main__":
    main()
