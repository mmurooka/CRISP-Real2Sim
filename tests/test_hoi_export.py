import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from export_hoi_human import scene_cameras
from align_hoi_scene import robust_scale


class SceneCameraTests(unittest.TestCase):
    def test_box_occlusion_does_not_corrupt_body_scale(self):
        observed=np.ones((10,10)); body=np.full((10,10),2.)
        body[:5]=100
        visible=np.ones((10,10),bool); visible[:5]=False
        scale,count,mad=robust_scale(observed,body,visible)
        self.assertEqual((scale,count,mad),(2.,50,0.))
        with self.assertRaises(ValueError): robust_scale(observed,body,np.zeros((10,10),bool))

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
