import torch

from collections import namedtuple
from .context import context
from .utils import human_readable_size, generate_report
from .storage_manager import storage_manager


class CudaStreamTimer():
    def __init__(self, stream):
        self.stream = stream
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stream.record_event(self.start_event)

    def stop(self):
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.stream.record_event(self.end_event)

    def time(self):
        """
        Please call torch.cuda.synchronize(device=context.device) before calling this function!!
        """
        return self.start_event.elapsed_time(self.end_event)

# the function of spectator is highly overlapped with storage manager. 
# The main function of spectator is timing & report
 
IterationStats = namedtuple('IterationStats', ['num_tensors', 'manageable_size', 'memory_footprint', 'swapout_throughput', 'swapin_throughput'])


class Spectator():
    def __init__(self):
        self.reset()

    def reset(self):
        self.tid_size = dict()

        self.record_reuse_distance = False
        self.record_swapping_time = False
        self.timers = {
            'swap_in': {},
            'swap_out': {},
            'reuse': {},
        }

        self.total_swapout_size = 0
        self.cuda_memory_pool_usage_trace = []

    def set_tid_size(self, tid, size):
        self.tid_size[tid] = size
    
    def get_tid_size(self):
        return self.tid_size
    
    def get_memory_footprint(self):
        return max(self.cuda_memory_pool_usage_trace)

    def report(self):
        num_tensors = storage_manager.get_num_tensors()
        # storage_size = storage_manager.get_storage_size()

        manageable_size = sum(self.tid_size.values())
        memory_footprint = self.get_memory_footprint()
        # model_memory_consumption = memory_footprint + self.total_swapout_size

        stat = IterationStats(num_tensors, manageable_size, memory_footprint, 0, 0)
        context.iteration_stats.append(stat)

        generate_report("Spectator Iteration", {
            "Step": context.step,
            "Number of Tensors": num_tensors,
            "Swap-out Size": human_readable_size(self.total_swapout_size),
            "Manageable Size": human_readable_size(stat.manageable_size),
            "Memory Footprint": human_readable_size(stat.memory_footprint)
        })

    def step(self):
        self.report()
        self.reset()
       
    def increase_swap_out_size(self, size):
        self.total_swapout_size += size

    def timer_begin(self, event, tid, stream):
        if self.record_swapping_time:
            self.timers[event][tid] = CudaStreamTimer(stream)            

    def timer_end(self, event, tid):
        if self.record_swapping_time:
            self.timers[event][tid].stop()

    def record_memory_footprint(self):
        usage = torch.cuda.memory_allocated()
        self.cuda_memory_pool_usage_trace.append(usage)

spectator = Spectator()
