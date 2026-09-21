import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F
from overlap_loss import LossMaskIndex, compact_ranges, digest_ids, digest_text


class MaskSemantics(unittest.TestCase):
    def test_original_inputs_eos_and_unmasked_dev(self):
        target = "example"
        ids = torch.tensor([10, 11, 2, 3, 4, 5, 9])
        labels = torch.tensor([-100, -100, 2, 3, 4, 5, 9])
        original_ids, original_labels = ids.clone(), labels.clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"mask.json"
            path.write_text(json.dumps(dict(
                ready_for_training=True, unmasked_dev_audio=["dev.wav"],
                records={"train.wav":dict(target_sha256=digest_text(target),
                    target_ids_sha256=digest_ids(ids[2:].tolist()), target_tokens_with_eos=5,
                    ignored_tokens=2, ignore_ranges=[[1,3]])})), encoding="utf-8")
            mask = LossMaskIndex(path)
            actual = mask.apply("train.wav",target,ids,labels)
            self.assertEqual(actual.tolist(),[-100,-100,2,-100,-100,5,9])
            self.assertTrue(torch.equal(ids,original_ids))
            self.assertTrue(torch.equal(labels,original_labels))
            self.assertTrue(torch.equal(mask.apply("dev.wav",target,ids,labels),labels))
            with self.assertRaises(ValueError):
                mask.apply("unknown.wav",target,ids,labels)

    def test_shifted_ce_zero_direct_gradient_and_sp_global_reduction(self):
        # Four sequence shards of the same shifted label vector, including -100.
        torch.manual_seed(11)
        logits=torch.randn(1,12,19,requires_grad=True)
        labels=torch.tensor([[-100,-100,3,-100,4,5,-100,7,-100,8,9,10]])
        shifted=torch.cat([labels[:,1:],torch.full((1,1),-100)],dim=1)
        full=F.cross_entropy(logits.view(-1,19),shifted.view(-1),ignore_index=-100)
        pieces=[]
        for l,y in zip(logits.chunk(4,dim=1),shifted.chunk(4,dim=1)):
            pieces.append(F.cross_entropy(l.reshape(-1,19),y.reshape(-1),
                                         ignore_index=-100,reduction="sum"))
        sharded=sum(pieces)/shifted.ne(-100).sum()
        self.assertTrue(torch.allclose(full,sharded,atol=1e-7))
        full.backward()
        self.assertTrue(bool(logits.grad[shifted.eq(-100)].eq(0).all()))
        self.assertTrue(bool(logits.grad[shifted.ne(-100)].abs().sum()>0))
        self.assertEqual(compact_ranges([False,True,True,False,True]),[[1,3],[4,5]])


if __name__=="__main__":
    unittest.main()
