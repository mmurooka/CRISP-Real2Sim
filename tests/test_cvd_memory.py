"""Ensure memory reduction preserves the upstream objective and its gradients."""
import os
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


@unittest.skipUnless(torch.cuda.is_available(), 'Original CVD requires CUDA')
class CVDMemoryTest(unittest.TestCase):
    def test_loss_and_gradients_with_masks(self):
        import cvd_memory as cvd
        from geometry_utils import NormalGenerator
        torch.manual_seed(7)
        device = 'cuda'
        n, h, w = 4, 32, 40
        cam = torch.eye(4, device=device).repeat(n, 1, 1)
        cam[:, :3, 3] = torch.randn(n, 3, device=device) * .01
        k = torch.tensor([[32., 0., 20.], [0., 32., 16.], [0., 0., 1.]], device=device)
        disp = .6 + torch.rand(n, h, w, device=device) * .5
        init = .6 + torch.rand_like(disp) * .5
        uncertainty = .3 + torch.rand(n, 1, h, w, device=device) * .6
        ii = torch.tensor([0, 0, 1, 1, 2, 3], device=device)
        jj = torch.tensor([1, 2, 2, 3, 3, 0], device=device)
        flows = torch.randn(len(ii), 2, h, w, device=device) * .3
        masks = (torch.rand(len(ii), 1, h, w, device=device) > .2).float()
        invalid = torch.rand(n, h, w, device=device) < .1
        normals = [NormalGenerator(h, w).to(device)]
        alpha = torch.rand(n, h, w, device=device)

        def evaluate(function):
            d = disp.clone().requires_grad_()
            u = uncertainty.clone().requires_grad_()
            loss = function(cam, k, k.inverse(), d, init, u, flows, masks,
                            ii, jj, normals, alpha, invalid)
            return (loss.detach(), *torch.autograd.grad(loss, (d, u)))

        expected = evaluate(cvd.original.consistency_loss)
        previous = os.environ.get('CRISP_CVD_PAIR_BATCH')
        try:
            for batch in (1, 2, 5, 32):
                os.environ['CRISP_CVD_PAIR_BATCH'] = str(batch)
                actual = evaluate(cvd.consistency_loss)
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-7)
        finally:
            if previous is None:
                os.environ.pop('CRISP_CVD_PAIR_BATCH', None)
            else:
                os.environ['CRISP_CVD_PAIR_BATCH'] = previous


if __name__ == '__main__':
    unittest.main()
