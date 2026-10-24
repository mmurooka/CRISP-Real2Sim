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

`refine_hoi_human.py` optionally refines SMPL-X pose and translation against COCO17 image observations, the original motion prior, temporal continuity, and box/floor nonpenetration. The box is fixed in this stage. `--geometry-only` adds no contact attraction; `--contacts` requires complete, provenance-matched InteractVLM masks and `--scene`. Attraction is limited to learned regions within 30 cm of the independently estimated object and within 3 pixels of its observed mask on the scene-prior grid. This prevents a positive language-model prediction from pulling a hand across a visible image gap. It is not a joint object/human optimization or an anatomical joint-limit certificate. `evaluate_hoi_geometry.py` compares image agreement and intersections using the same procedure before and after refinement.

`setups/install_interactvlm_checkpoint.py` installs a completed user-downloaded wScene ZIP locally, checks archive CRCs and shard completeness, rejects unsafe archive paths, and records the archive SHA-256. It does not download models or handle credentials.

All generated videos, data, models and reports belong in ignored directories. Changes to code are committed separately from those outputs. Source video hashes, timestamp arrays, world-frame identifiers and body topology must match before contacts are assembled.

## Interpretation and current limitations

- Silhouette agreement is an image-space check, not independent 3D validation. A cuboid can fit its silhouette while a recovered hand penetrates it or lies behind it.
- `valid` in box geometry means the stated image/floor checks passed; it does not certify true contacts or physical motion. Review rotation increments, penetration and force residuals separately.
- TAPIP3D tracking failed during lifting in the current video. Its saved result is a diagnostic baseline, not the selected object trajectory.
- `--geometry-only` in PRISM explicitly removes the learned visual prior. Its output must not be presented as InteractVLM predictions. Near-surface false positives and intersections are expected when human and object reconstructions disagree.
- Forces are explanatory and mass-normalized when mass is unknown. A region is conditionally necessary only if the full wrench fit is feasible and removing that entire region makes it infeasible. Neither force fitting nor a small image residual proves the true contact label.
- A single video and synthetic analytic tests do not establish generalization, novelty or real-robot transfer. Human/object joint refinement, contact uncertainty, independent labels, retargeting and RL evaluation remain required.
