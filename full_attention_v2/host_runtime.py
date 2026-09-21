"""Release unused host activation caches between otherwise unchanged updates.

The numerically qualified train.py and memory_sft.py remain byte-identical.
This wrapper only adds allocator maintenance, host telemetry and early saves.
"""
import gc
import os
from pathlib import Path
import time

import torch


def host_snapshot():
    stats = torch.cuda.host_memory_stats()
    rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    result = dict(rss_gib=rss / 2**30,
                  pinned_owned_gib=stats.get('allocated_bytes.current', 0) / 2**30,
                  pinned_active_gib=stats.get('active_bytes.current', 0) / 2**30,
                  pinned_peak_gib=stats.get('allocated_bytes.peak', 0) / 2**30)
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy, controllers, relative = line.split(':', 2)
        if hierarchy == '0' and not controllers:
            folder = Path('/sys/fs/cgroup') / relative.lstrip('/')
            for name in ['memory.current', 'memory.max', 'memory.peak']:
                p = folder / name
                if p.is_file():
                    value = p.read_text().strip()
                    if value.isdigit():
                        result['cgroup_' + name.split('.')[-1] + '_gib'] = int(value) / 2**30
    return result


def release_host_cache():
    torch.cuda.synchronize()
    before = host_snapshot()
    gc.collect()
    torch.accelerator.memory.empty_host_cache()
    after = host_snapshot()
    return dict(before=before, after=after)


def main():
    import train
    from common import append, emit, now

    class HostAudit(train.Audit):
        def on_train_begin(self, args, state, control, **kwargs):
            result = super().on_train_begin(args, state, control, **kwargs)
            emit('host_memory_policy', rank=args.process_index,
                 policy='synchronize_gc_release_unused_pinned_after_each_update',
                 initial=release_host_cache())
            return result

        def on_step_end(self, args, state, control, **kwargs):
            # No model/gradient/optimizer mutation; live pinned tensors stay live.
            start = time.perf_counter()
            rng_before = train.rng_digest()
            memory = release_host_cache()
            assert train.rng_digest() == rng_before, 'Allocator cleanup changed RNG'
            append(self.output / f'host-memory-rank-{args.process_index}.jsonl',
                   dict(step=state.global_step, rank=args.process_index, utc=now(),
                        cleanup_seconds=time.perf_counter()-start, rng_unchanged=True, **memory))
            result = super().on_step_end(args, state, control, **kwargs)
            if not self.gate and state.global_step in [1, 10, 20]:
                result.should_save = True
            return result

    train.Audit = HostAudit
    train.main()


if __name__ == '__main__':
    main()
