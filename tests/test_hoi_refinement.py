"""Independent distance and derivative checks for differentiable refinement."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from refine_hoi_human import cuboid_signed_distance,visual_contact_gate,region_patch_loss


class RefinementTests(unittest.TestCase):
    def test_two_contact_regions_cannot_ignore_the_more_distant_one(self):
        gap=torch.tensor([0.,.1,.3,.4],dtype=torch.float64,requires_grad=True)
        value=region_patch_loss(gap,[torch.tensor([0,1]),torch.tensor([2,3])],maximum_vertices=1)
        self.assertAlmostEqual(float(value),.045)
        value.backward()
        np.testing.assert_allclose(gap.grad,[0,0,.3,0],atol=1e-12)
        self.assertEqual(float(region_patch_loss(gap,[])),0)

    def test_visual_gap_rejects_false_contact_attraction(self):
        camera=np.eye(4);camera[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];camera[:3,3]=[1.,2.,3.]
        points=np.array([[0,0,2],[1,0,2],[0,0,-1],[.4,0,2]])
        world=points@camera[:3,:3].T+camera[:3,3]
        mask=np.zeros((20,20),bool);mask[9:12,9:12]=True
        k=np.array([[10.,0,10],[0,10,10],[0,0,1]])
        actual=visual_contact_gate(world[None],camera[None],k,mask[None],max_distance_px=1.)
        np.testing.assert_array_equal(actual,[[True,False,False,True]])

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
