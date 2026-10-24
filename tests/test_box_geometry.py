"""Physical scale and camera-convention checks for cuboid initialization."""
import itertools
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reconstruct_box_planar import calibrate_dimensions
from refine_box_silhouette import ray_box_depth


class BoxGeometryTests(unittest.TestCase):
    def test_floor_silhouette_recovers_metric_dimensions(self):
        dimensions=np.array([.4,1.,.3])
        local=np.array(list(itertools.product([-1,1],repeat=3)))*dimensions/2
        rotation=Rotation.from_euler('y',-25,degrees=True).as_matrix()@np.diag([1,-1,-1])
        points=local@rotation.T+np.array([.3,.5,3.5])
        k=np.array([[500.,0,300],[0,500,150],[0,0,1]])
        projected=points@k.T;projected=projected[:,:2]/projected[:,2:]
        indices=cv2.convexHull(projected.astype(np.float32),returnPoints=False)[:,0]
        poly=projected[indices]
        estimated,records=calibrate_dimensions([poly]*5,np.repeat(poly.mean(axis=0)[None],5,axis=0),
            k,np.repeat(np.eye(4)[None],5,axis=0),np.array([0,-1.,0]),-1.)
        np.testing.assert_allclose(estimated,dimensions,atol=1e-6)
        self.assertGreaterEqual(len(records),3)

    def test_ray_box_depth_is_camera_z_and_rejects_misses(self):
        depth,hit=ray_box_depth(np.array([[0.,0,1],[.01,0,1],[1.,0,1]]),
            np.diag([1.,-1,-1]),np.array([0.,0,3]),np.array([.2,.5,.3]))
        np.testing.assert_array_equal(hit,[True,True,False])
        np.testing.assert_allclose(depth[:2],2.7,atol=1e-8)


if __name__=='__main__':unittest.main()
