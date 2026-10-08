"""Numerical regression tests for the web laboratory (no web browser required)."""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

import server


class LaboratoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab=server.Lab((16,3,8,8))

    def setUp(self):
        self.lab.reset((16,3,8,8),42)

    def job(self):
        return dict(id='test',revision=self.lab.revision,started=time.time(),progress=0)

    def test_edit_changes_exactly_one_scalar_and_rejects_stale_revision(self):
        before=self.lab.z.copy();revision=self.lab.revision
        edit=dict(revision=revision,c=4,t=1,y=3,x=2,value=1.25)
        self.lab.edit(edit)
        self.assertEqual(np.count_nonzero(before!=self.lab.z),1)
        self.assertEqual(self.lab.z[4,1,3,2],1.25)
        with self.assertRaises(ValueError):self.lab.edit(edit)
        self.assertTrue(np.array_equal(before,self.lab.baseline))

    def test_region_channel_time_masks_and_bounds(self):
        for mode,extent,count in [('region',[2,3,4],24),('channel',[1,1,1],192),('time',[2,1,1],2048)]:
            self.lab.reset((16,3,8,8),42)
            before=self.lab.z.copy()
            self.lab.edit(dict(revision=self.lab.revision,mode=mode,c=0,t=1,y=1,x=1,value=.25,extent=extent))
            self.assertEqual(np.count_nonzero(before!=self.lab.z),count)
        with self.assertRaises(ValueError):
            self.lab.edit(dict(revision=self.lab.revision,mode='region',c=0,t=1,y=7,x=7,value=1,extent=[1,2,2]))

    def test_streamed_rgb_equals_upstream_and_has_correct_trim(self):
        old=server.RUNS
        with tempfile.TemporaryDirectory() as tmp,torch.inference_mode():
            server.RUNS=Path(tmp)
            try:
                z=self.lab.z.copy()
                reference=self.lab.model.decode_video(torch.from_numpy(z).permute(1,0,2,3).unsqueeze(0),parallel=False,show_progress_bar=False)[0].numpy()
                self.lab.decode(self.job(),z)
                actual=np.load(Path(tmp)/'test.npy')
                self.assertEqual(actual.shape,(9,3,64,64))
                np.testing.assert_array_equal(actual,reference)
                self.assertGreater((Path(tmp)/'test.mp4').stat().st_size,100)
                result=self.lab.results[-1]
                records={r['name']:r for group in self.lab.graph for r in [group,*group['children']]}
                for name,meta in result['activations'].items():
                    saved=np.load(Path(tmp)/('test.'+name+'.npy'))
                    shape=records[name]['shape']
                    self.assertEqual(list(saved.shape),[shape[1] if name.count('.')==1 else 1,*shape[2:]])
                    self.assertTrue(np.isfinite(saved).all())
                for name,t in [('decoder.1',1),('decoder.9',1),('decoder.13',3),('decoder.22',7),('decoder.3.conv.0',0)]:
                    self.lab.capture(self.job(),z,dict(target=records[name],time=t))
                    saved=np.load(Path(tmp)/('test.'+name+'.npy'))
                    np.testing.assert_array_equal(saved[t],self.lab.trace['array'][0])
            finally:
                server.RUNS=old
                self.lab.results=[]

    def test_intermediate_hooks_clone_before_inplace_activation(self):
        records={r['name']:r for group in self.lab.graph for r in [group,*group['children']]}
        # Include a pre-ReLU convolution, a memory-block interior and temporal reshape.
        for name,t in [('decoder.1',1),('decoder.3.conv.0',1),('decoder.13.conv',2),('decoder.13',3),('decoder.22',7)]:
            captured=[]
            handle=self.lab.model.get_submodule(name).register_forward_hook(lambda m,a,o:captured.append(o.detach().clone()))
            with torch.inference_mode():
                self.lab.model.decode_video(torch.from_numpy(self.lab.z).permute(1,0,2,3).unsqueeze(0),parallel=False,show_progress_bar=False)
            handle.remove()
            expected=torch.cat(captured,dim=0)[t].numpy()
            with torch.inference_mode():
                self.lab.capture(self.job(),self.lab.z.copy(),dict(target=records[name],time=t))
            np.testing.assert_array_equal(self.lab.trace['array'],expected)
            math=self.lab.trace['meta'].get('arithmetic')
            if math:
                np.testing.assert_allclose(np.array(math['input_patch'])*np.array(math['kernel']),math['products'],atol=1e-7)
                self.assertAlmostEqual(math['all_channel_sum']+math['bias'],float(expected[0,0,0]),places=5)
        self.assertEqual(len(self.lab.parameters),64)
        self.assertEqual(len(self.lab.graph),23)


if __name__=='__main__':unittest.main(verbosity=2)
