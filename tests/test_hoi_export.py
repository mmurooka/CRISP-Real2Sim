import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_hoi_human import scene_cameras


class SceneCameraTests(unittest.TestCase):
    def test_depth_scale_applies_only_to_camera_translation(self):
        cameras=np.tile(np.eye(4),(3,1,1)); cameras[:,0,3]=[0,1,2]
        transformed,scale=scene_cameras(dict(cam_c2w=cameras,scale=2),3)
        np.testing.assert_allclose(transformed[:,0,3],[0,2,4])
        np.testing.assert_allclose(transformed[:,:3,:3],cameras[:,:3,:3])
        np.testing.assert_allclose(cameras[:,0,3],[0,1,2])

    def test_missing_frames_and_nonrigid_camera_fail(self):
        cameras=np.tile(np.eye(4),(3,1,1))
        with self.assertRaises(ValueError):
            scene_cameras(dict(cam_c2w=cameras,scale=1,valid_frame_indices=[0,2,3]),3)
        cameras[1,0,0]=2
        with self.assertRaises(ValueError):
            scene_cameras(dict(cam_c2w=cameras,scale=1),3)


if __name__=='__main__':
    unittest.main()
