"""Map per-frame InteractVLM labels to colors on the reconstructed SMPL mesh."""

from pathlib import Path

import numpy as np


def contact_vertex_colors(contact_dir, frame_index, vertex_count, base_color, threshold=0.5):
    """Require the original frame's labels and preserve the SMPL vertex order."""
    contact_file = Path(contact_dir) / f"{frame_index:05d}.npz"
    with np.load(contact_file, allow_pickle=False) as data:
        scores = np.asarray(data["pred_contact_3d_smplh"]).reshape(-1)
    if scores.size != vertex_count:
        raise ValueError(
            f"Contact topology mismatch in {contact_file}: "
            f"{scores.size} labels for {vertex_count} vertices"
        )
    if not np.isfinite(scores).all():
        raise ValueError(f"Non-finite contact scores in {contact_file}")
    colors = np.tile(np.array([*base_color, 255], dtype=np.uint8), (vertex_count, 1))
    colors[scores > threshold] = [255, 45, 45, 255]
    return colors
