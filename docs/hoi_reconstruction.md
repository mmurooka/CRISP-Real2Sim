# Monocular whole-body HOI reconstruction

The scripts below are an independent moving-cuboid extension; they are not the unpublished PRISM reconstruction pipeline. Metric scale comes from the estimated human body. Box dimensions, object mass and human height have not been measured.

## Stages

1. SAM2 person/object masks and cached human observations.
2. `run_megasam_persistent.py --checkpointed-cvd`: persistent depth/camera inference with exact checkpointed CVD losses.
3. `run_cached_gvhmr.py`: human reconstruction using cached observations and the original presentation rate.
4. `align_hoi_scene.py`, then `export_hoi_human.py`: shared metric coordinate frame, explicit camera scale and provenance.
5. `reconstruct_box_planar.py`: estimate cuboid dimensions from stationary floor-supported endpoint silhouettes, then planar RGB tracking for pose initialization.
6. `refine_box_silhouette.py`: visible silhouette/depth fitting, person-occlusion handling, floor nonpenetration, temporal translation/rotation trust regions. A bounded monocular-depth residual prevents large depth outliers from dominating RGB. Rotation increments are local to prevent cuboid symmetry flips.
7. `crop_hoi_contact_frames.py`: square person-plus-object crops for local InteractVLM inference. The original wide image would lose the person under CLIP center cropping.
8. PRISM `scripts/prepare_whole_body_input.py` and `scripts/reconstruct_whole_body_contacts.py`: visual/geometric contact persistence, gravity-aware Newton–Euler fitting and region-removal tests.
9. `render_hoi_review.py`: local source/overlay/independent-view comparison video at the original frame rate.

`refine_hoi_human.py` optionally refines SMPL-X pose and translation against COCO17 image observations, the original motion prior, temporal continuity, and box/floor nonpenetration. The box is fixed in this stage. `--geometry-only` adds no contact attraction; `--contacts` requires complete, provenance-matched InteractVLM masks and `--scene`. Attraction is limited to learned regions within 30 cm of the independently estimated object and within 3 pixels of its observed mask on the scene-prior grid. Each predicted semantic region attracts at most its 20 nearest eligible vertices, so a whole-hand mask does not flatten the closed hand surface onto the box. This prevents a positive language-model prediction from pulling a hand across a visible image gap. It is not a joint object/human optimization or an anatomical joint-limit certificate. `evaluate_hoi_geometry.py` compares image agreement and intersections using the same procedure before and after refinement.

`setups/install_interactvlm_checkpoint.py` installs a completed user-downloaded wScene ZIP locally, checks archive CRCs and shard completeness, rejects unsafe archive paths, and records the archive SHA-256. It does not download models or handle credentials.

All generated videos, data, models and reports belong in ignored directories. Changes to code are committed separately from those outputs. Source video hashes, timestamp arrays, world-frame identifiers and body topology must match before contacts are assembled.

### Optional local whole-body language hypotheses

`infer_wholebody_contact_vlm.py` compares a local Qwen3-VL-4B-Instruct model
against the existing InteractVLM contact predictions. It examines the whole
body without presupposing a leg contact, retains the unmodified response and
its parsed categories, and records image/model hashes and generation settings.
Model files must already be present with a completed `download_manifest.json`
containing their pinned Hub revision, filenames and byte sizes. Inference uses
offline mode and local files only; it does not send images to an inference API.
`setups/download_local_contact_vlm.py` obtains the pinned public weights (about
8.9 GB) without credentials and skips an already complete local download.

```bash
.venv/bin/python setups/download_local_contact_vlm.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/infer_wholebody_contact_vlm.py \
  --model data/downloads/qwen3-vl-4b-instruct \
  --frames data/move_box_contact_img/move_box_adjusted_by_leg \
  --frame-indices 50 111 120 143 150 163 200 300 \
  --output results/move_box_adjusted_by_leg/qwen_contact_samples_v1
```

These are uncalibrated image-level hypotheses, not SMPL vertex masks or measured
contact forces. Anatomical left/right can remain unknown. Hidden contact and
2D overlap remain ambiguous. The script does not overwrite InteractVLM output
or automatically change reconstruction, retargeting, or RL targets.

After inference on **every** frame, `fuse_semantic_contact_cues.py --language DIR
--image-cues DIR --output DIR` intersects likely region mentions with the separate
image-boundary/3D-proximity candidates and requires three consecutive frames.
It ignores unreliable language left/right claims when proposing a body-part
family; the image/geometry gate selects anatomical sides. Chest/abdomen map to
torso; knee and thigh remain distinct. Missing or uncertain responses cannot
become positive contact labels. A conservative English-text check downgrades
"likely" labels whose own explanation denies contact or identifies the floor
as the contact target; this check can introduce false negatives and is not a
second visual classifier. Raw responses, uncertainty, rejected proposals,
and source hashes are retained. The output is a separate candidate supplement,
not a replacement for learned vertex masks or a measured contact assignment.
`render_hoi_review.py --contacts CONTACTS_NPZ --semantic-cues FUSED_DIR` can
show the supplement in cyan on vertices that were not already marked by the
original visual prior. It checks the human hash, video, times and vertex indexing;
the original force and visual evidence archives are not changed.

## Interpretation and current limitations

- Silhouette agreement is an image-space check, not independent 3D validation. A cuboid can fit its silhouette while a recovered hand penetrates it or lies behind it.
- `valid` in box geometry means the stated image/floor checks passed; it does not certify true contacts or physical motion. Review rotation increments, penetration and force residuals separately.
- TAPIP3D tracking failed during lifting in the current video. Its saved result is a diagnostic baseline, not the selected object trajectory.
- `--geometry-only` in PRISM explicitly removes the learned visual prior. Its output must not be presented as InteractVLM predictions. Near-surface false positives and intersections are expected when human and object reconstructions disagree.
- Forces are explanatory and mass-normalized when mass is unknown. A region is conditionally necessary only if the full wrench fit is feasible and removing that entire region makes it infeasible. Neither force fitting nor a small image residual proves the true contact label.
- A single video and synthetic analytic tests do not establish generalization, novelty or real-robot transfer. Human/object joint refinement, contact uncertainty, independent labels, retargeting and RL evaluation remain required.

### Local result bundle

`export_hoi_bundle.py` collects a selected human/object reconstruction, contact
provenance, separate image cues and a G1 kinematic reference into a local ignored
directory. It validates source-video hashes, timestamps, vertex indexing, object
poses and retargeting provenance before writing atomically. The bundle includes
an HTML data guide and SHA256 manifest; it does not include an executable robot
controller or promote candidate contacts to ground truth.

```bash
.venv/bin/python scripts/export_hoi_bundle.py \
  --human results/move_box_adjusted_by_leg/human_contact_refined_v2.npz \
  --objects results/move_box_adjusted_by_leg/box_silhouette_geometry_v5.npz \
  --contact-input results/move_box_adjusted_by_leg/contact_input_refined_v2.npz \
  --contacts results/move_box_adjusted_by_leg/contacts_refined_v2 \
  --image-cues results/move_box_adjusted_by_leg/image_contact_cues_v1 \
  --g1 results/move_box_adjusted_by_leg/g1_wrench_optimized_v2 \
  --output results/exports/move_box_hoi_v1
```
