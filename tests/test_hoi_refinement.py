"""Independent distance and derivative checks for differentiable refinement."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_hoi_human import cuboid_signed_distance


class RefinementTests(unittest.TestCase):
    def test_signed_distance_and_gradient_at_faces_edges_and_inside(self):
        points=torch.tensor([[.6,.1,.1],[.6,.7,0.],[0.,.45,0.]],dtype=torch.float64,requires_grad=True)
        half=torch.tensor([.5,.5,.5],dtype=torch.float64)
        distance=cuboid_signed_distance(points,half)
        np.testing.assert_allclose(distance.detach(),[.1,np.sqrt(.05),-.05],atol=1e-10)
        distance.sum().backward()
        expected=np.array([[1,0,0],[1/np.sqrt(5),2/np.sqrt(5),0],[0,1,0]])
        np.testing.assert_allclose(points.grad,expected,atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(lambda p:cuboid_signed_distance(p,half),(points,),eps=1e-6,atol=1e-5))


if __name__=='__main__':unittest.main()
