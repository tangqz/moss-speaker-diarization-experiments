"""Logging and bounded phase control through Swift's public callback registry."""
import json,os
from pathlib import Path
import torch
from swift.callbacks import callbacks_map
from swift.callbacks.base import TrainerCallback

class PhaseCallback(TrainerCallback):
    def on_train_begin(self,args,state,control,model=None,**kwargs):
        params=list(model.parameters())
        assert params and all(p.requires_grad for p in params), 'Full SFT requires every parameter trainable.'
        assert all(p.dtype==torch.float32 for p in params), 'Keep FP32 master parameters.'
        if args.process_index==0:
            Path(args.output_dir,'native_runtime.json').write_text(json.dumps(dict(
                trainer_class=type(self.trainer).__module__+'.'+type(self.trainer).__name__,
                model_class=type(model).__module__+'.'+type(model).__name__,
                parameter_dtype='float32',local_trainable_parameter_elements=sum(p.numel() for p in params),
                fsdp_config=getattr(args,'fsdp_config',{}),
                implicit_full_causal=os.environ.get('MOSS_IMPLICIT_CAUSAL')=='1',
                native_ce_chunk_size=int(os.environ.get('CELOSS_PARALLEL_SIZE','0')),
                optimizer_lr_at_start=[g['lr'] for g in self.trainer.optimizer.param_groups],
                starting_global_step=state.global_step,
                effective_eval_steps=state.eval_steps,
                sequence_parallel_size=self.trainer.template.sequence_parallel_size,
                custom_forward=False,custom_loss=False,custom_optimizer=False),indent=2))

    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step>=int(os.environ.get('MOSS_STOP_STEP','402')):
            control.should_training_stop=True
            control.should_save=True
        return control

callbacks_map['moss_phase']=PhaseCallback
