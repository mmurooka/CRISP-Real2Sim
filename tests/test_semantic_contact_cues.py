import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fuse_semantic_contact_cues import conservative_contacts, semantic_states, persistent_candidates, inference_image_path


class SemanticContactCueTests(unittest.TestCase):
    def test_mixed_run_manifest_keeps_each_actual_input(self):
        metadata = {'frame_image_paths': {'0': '/tmp/original/00000.jpg', '1': '/tmp/mosaic/00001.jpg'}}
        self.assertEqual(inference_image_path(metadata, 1, {'image_path': '/tmp/mosaic/00001.jpg'}),
                         Path('/tmp/mosaic/00001.jpg'))
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
            inference_image_path(metadata, 1, {'image_path': '/tmp/original/00001.jpg'})

    def test_geometry_selects_side_and_rejects_distant_foot(self):
        regions = ["left_thigh", "right_thigh", "left_foot", "right_foot"]
        language = [{"region": "thigh", "side": "left", "state": "likely"},
                    {"region": "foot", "side": "right", "state": "likely"}]
        states = np.tile(semantic_states(language, regions), (3, 1))
        geometry = np.tile([False, True, False, False], (3, 1))
        _, fused = persistent_candidates(states, geometry, 3)
        np.testing.assert_array_equal(fused, geometry)

    def test_uncertain_missing_and_short_runs_are_not_promoted(self):
        states = np.array([[2], [2], [-1], [2], [1], [2], [2], [2]], np.int8)
        raw, fused = persistent_candidates(states, np.ones_like(states, bool), 3)
        np.testing.assert_array_equal(fused[:, 0], [0, 0, 0, 0, 0, 1, 1, 1])
        self.assertFalse(raw[4, 0])

    def test_knee_is_not_silently_expanded_to_thigh(self):
        states = semantic_states([{"region": "knee", "side": "unknown", "state": "likely"}],
                                 ["left_knee", "right_knee", "right_thigh"])
        np.testing.assert_array_equal(states, [2, 2, 0])

    def test_self_contradiction_cannot_create_positive_evidence(self):
        original = [{"region": "knee", "side": "right", "state": "likely",
                     "box_location": "floor adjacent to the box",
                     "visual_evidence": "close to box, but not touching it"},
                    {"region": "thigh", "side": "right", "state": "likely",
                     "box_location": "bottom edge of box",
                     "visual_evidence": "thigh against box with no visible gap"}]
        guarded, flags = conservative_contacts(original)
        self.assertEqual([x["state"] for x in guarded], ["uncertain", "likely"])
        self.assertEqual(original[0]["state"], "likely")
        self.assertEqual(len(flags), 1)


if __name__ == "__main__":
    unittest.main()
