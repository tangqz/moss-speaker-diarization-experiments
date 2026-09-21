"""Exercise the real Trainer/Accelerate shuffled loader and every resume boundary.

Dataset indices replace audio only in this sampler test. Formal training checks
its actual full-audio batch indices against the frozen result after every phase.
"""
import argparse
from pathlib import Path
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import Trainer,TrainingArguments,set_seed
from accelerate import skip_first_batches
from common import write,emit


class Indices(Dataset):
    def __len__(self):return 536
    def __getitem__(self,index):return {'index':index}


class Placeholder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
    def forward(self,index):return {'loss':self.weight.sum()*0}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    set_seed(0)
    args=TrainingArguments(output_dir=str(a.run/'sampler-test'),per_device_train_batch_size=1,
        gradient_accumulation_steps=1,max_steps=402,num_train_epochs=3,bf16=True,seed=0,data_seed=0,
        remove_unused_columns=False,dataloader_num_workers=0,train_sampling_strategy='random',
        report_to='none',ddp_find_unused_parameters=False)
    trainer=Trainer(model=Placeholder(),args=args,train_dataset=Indices())
    assert args.world_size==4
    epochs=[]
    for epoch in range(3):
        loader=trainer.get_train_dataloader();loader.set_epoch(epoch)
        local=[int(batch['index'].item()) for batch in loader]
        assert len(local)==134
        gathered=[None]*4
        dist.all_gather_object(gathered,local)
        assert sorted(i for row in gathered for i in row)==list(range(536))
        epochs.append(gathered)
    boundaries=sorted(set(range(0,402,25))|{10,134,268})
    for step in boundaries:
        epoch,offset=divmod(step,134)
        loader=trainer.get_train_dataloader();loader.set_epoch(epoch)
        batch=next(iter(skip_first_batches(loader,offset)))
        actual=int(batch['index'].item())
        expected=epochs[epoch][args.process_index][offset]
        assert actual==expected,(step,args.process_index,actual,expected)
    if args.process_index==0:
        write(a.run/'data_order.json',dict(seed=0,data_seed=0,epochs=epochs,
            layout='epochs[epoch][rank][step_in_epoch]',records_per_epoch=536,
            verified_resume_steps=boundaries,world_size=4,all_resume_first_batches_equal=True))
        emit('shuffled_sampler_resume_gate_pass',boundaries=boundaries)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
