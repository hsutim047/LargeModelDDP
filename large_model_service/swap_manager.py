import torch

import pickle
import time
import os
import time

from .spectator import spectator
from .storage_manager import storage_manager
from .profiler import parse_event
from .context import context

from .packed_tensor import pack_hook, unpack_hook 
from .multi_gpu import multigpu_iteration_guard


def export_chrome_trace():
    os.system(f"mkdir -p {context.path}")
    if context.enable_multi_gpu:
        path = os.path.join(context.path, f"{context.filename}.rank{context.ipc_object.rank()}.json")
    else:
        path = os.path.join(context.path, f"{context.filename}.json")
    context.torch_profiler.export_chrome_trace(path)
    print(f"Export trace to {path}")


def export_profiler_info(info):
    path = os.path.join(context.path, f'{context.filename}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(info, f)
    print(f"Export profiler info to {path}")


class SwapManager(object):
    def __init__(self):
        self.start_time = 0
        self.elapsed_time = 0

    def __enter__(self):
        torch._C._autograd._push_saved_tensors_default_hooks(pack_hook, unpack_hook)

        # enter the profiler at the first of iterations
        if context.step == 0 and context.profile:
            context.torch_profiler = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=1, warmup=context.total_steps-2, active=1))
            context.torch_profiler.__enter__()
        self.start_time = time.time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.elapsed_time = time.time() - self.start_time
        torch._C._autograd._pop_saved_tensors_default_hooks()

        torch.cuda.synchronize(device=context.device)

        context.step = context.step + 1
        # exit the profiler at the end of iterations
        if context.profile:
            context.torch_profiler.step()
            if context.step == context.total_steps:
                print("Exiting Torch profiler")
                context.torch_profiler.__exit__(type, value, traceback)

                events = list(context.torch_profiler.events())
                info = parse_event(events)

                export_chrome_trace()
                export_profiler_info(info)

        if context.enable_swap:
            context.policy.schedule(spectator.get_tid_size())
        
        multigpu_iteration_guard.step()
        spectator.step()
        storage_manager.step() 

    def iteration_elapsed_time(self):
        return self.elapsed_time
