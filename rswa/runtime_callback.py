"""Native training stage control and explicit attention runtime evidence."""
import json,os,time
from pathlib import Path
import torch
from swift.callbacks import callbacks_map
from swift.callbacks.base import TrainerCallback

class RswaPhase(TrainerCallback):
    def on_step_begin(self,args,state,control,**kwargs):
        os.environ['MOSS_RSWA_OPTIMIZER_STEP']=str(state.global_step+1)
        self._step_started=time.monotonic()
        torch.cuda.reset_peak_memory_stats()

    def on_train_begin(self,args,state,control,model=None,**kwargs):
        params=list(model.parameters())
        assert params and all(p.requires_grad for p in params)
        assert all(p.dtype==torch.float32 for p in params)
        restore_checkpoint=os.environ.get('MOSS_RESUME_AUDIT_CHECKPOINT')
        if restore_checkpoint:
            from restore_audit import audit,watch_rng_restore
            assert state.global_step==1
            audit(self.trainer,restore_checkpoint,Path(args.output_dir,'restore_exact.json'))
            watch_rng_restore(self.trainer,restore_checkpoint,args.output_dir)
        if args.process_index==0:
            Path(args.output_dir,'rswa_runtime.json').write_text(json.dumps(dict(
                trainer_class=type(self.trainer).__module__+'.'+type(self.trainer).__name__,
                full_parameter=True,parameter_dtype='float32',custom_attention=True,
                custom_loss=False,custom_optimizer=False,window=os.environ['MOSS_RSWA_WINDOW'],
                initial_lr=args.learning_rate,sequence_parallel_size=self.trainer.template.sequence_parallel_size,
                starting_global_step=state.global_step,
                optimizer_lr_at_start=[g['lr'] for g in self.trainer.optimizer.param_groups]),indent=2))

    def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients,'No trainable gradients'
        norms=torch.stack(torch._foreach_norm(gradients))
        assert bool(torch.isfinite(norms).all()),'Nonfinite gradients: refusing optimizer update'

    def on_step_end(self,args,state,control,model=None,**kwargs):
        calls=[getattr(m,'rswa_forward_calls',0) for m in model.modules() if type(m).__name__=='Qwen3Attention']
        assert len(calls)==28 and all(c>0 for c in calls),('attention not active in all layers',calls)
        rank=args.process_index
        order=Path(args.output_dir,'sample_order.jsonl' if rank==0 else f'sample_order.rank{rank}.jsonl')
        meetings=[json.loads(line) for line in order.read_text().splitlines()[-4:]]
        assert len(meetings)==4 and all(r['optimizer_step']==state.global_step for r in meetings)
        with Path(args.output_dir,f'update_metrics.rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=state.global_step,rank=rank,
                seconds=time.monotonic()-self._step_started,
                complete_meetings=4,input_tokens=sum(r['length'] for r in meetings),
                supervised_tokens=sum(r['length']-r['prefix'] for r in meetings),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                semantics='rank-local memory/time; sequence-parallel ranks share these same four meetings'))+'\n')
        if state.global_step==5 or (state.global_step==1 and os.environ.get('MOSS_SAVE_STEP_ONE')=='1'):
            control.should_save=True
        if state.global_step>=int(os.environ.get('MOSS_STOP_STEP','402')):
            control.should_training_stop=True;control.should_save=True
        return control

callbacks_map['rswa_phase']=RswaPhase
