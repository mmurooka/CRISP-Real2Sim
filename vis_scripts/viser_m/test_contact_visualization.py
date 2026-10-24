import tempfile
import io
import unittest
from pathlib import Path

import numpy as np

from contact_visualization import contact_vertex_colors


class ContactVisualizationTest(unittest.TestCase):
    def test_original_frame_and_vertex_order(self):
        with tempfile.TemporaryDirectory() as directory:
            # The second rendered frame may be original video frame 7, not 1.
            np.savez(Path(directory) / "00001.npz", pred_contact_3d_smplh=[1, 0, 0])
            np.savez(Path(directory) / "00007.npz", pred_contact_3d_smplh=[[0, 0.9, 0.5]])
            colors = contact_vertex_colors(directory, 7, 3, [152, 251, 152])
            np.testing.assert_array_equal(colors, [
                [152, 251, 152, 255], [255, 45, 45, 255], [152, 251, 152, 255],
            ])

    def test_bad_topology_and_missing_frames_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "00000.npz", pred_contact_3d_smplh=[0, 1])
            with self.assertRaisesRegex(ValueError, "topology mismatch"):
                contact_vertex_colors(directory, 0, 3, [0, 0, 0])
            with self.assertRaises(FileNotFoundError):
                contact_vertex_colors(directory, 1, 2, [0, 0, 0])

    def test_nonfinite_predictions_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "00000.npz", pred_contact_3d_smplh=[np.nan])
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                contact_vertex_colors(directory, 0, 1, [0, 0, 0])

    def test_colors_survive_viewer_glb_serialization(self):
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "00000.npz", pred_contact_3d_smplh=[0, 1, 0])
            colors = contact_vertex_colors(directory, 0, 3, [152, 251, 152])
            mesh = trimesh.Trimesh(
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 2]], vertex_colors=colors, process=False,
            )
            scene = trimesh.load(
                io.BytesIO(mesh.export(file_type="glb")), file_type="glb", process=False,
            )
            restored = next(iter(scene.geometry.values()))
            np.testing.assert_array_equal(restored.visual.vertex_colors, colors)


if __name__ == "__main__":
    unittest.main()
