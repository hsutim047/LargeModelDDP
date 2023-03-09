from collections import defaultdict

from ..utils import human_readable_size, generate_report
from ..context import context
from ..spectator import spectator


class DynamicEarlyFirstPolicy():
    def __init__(self, tensor_size_threshold):
        self.swap_tid = set()
        self.tensor_size_threshold = tensor_size_threshold
        self.swap_size_ = 0
        self.save_size = 0

    def schedule(self, tid_size):
        memory_footprint = spectator.get_memory_footprint()
        saved_size = max(20 * 1e9 - memory_footprint, 0)

        curr = list(sorted(self.swap_tid))
        current_save_size = 0
        for tid in reversed(curr):
            if (current_save_size + tid_size[tid] < saved_size):
                self.swap_tid.remove(tid)

                size = tid_size[tid]
                current_save_size += size
                self.swap_size_ -= size

        self.save_size = saved_size
        self.report()

    def report(self):
        selected_tensors = ','.join([str(tid) for tid in sorted(self.swap_tid)])
        generate_report("Swapout Policy", {
            "Name": "Dynamic-Early-First",
            "Number of Swapped Out Tensors": self.num_selected_tensors(),
            "Selected Tensors": selected_tensors,
            "Tensor Size Threshold": human_readable_size(self.tensor_size_threshold),
            "Swappable Space": human_readable_size(self.save_size),
            "Total Size": human_readable_size(self.swap_size())
        })

    def swap_size(self):
        return self.swap_size_
    
    def num_selected_tensors(self):
        return len(self.swap_tid)
    
    def is_last_tensor(self, tid): 
        max_id = max(self.swap_tid)     
        return tid == max_id
        
    def is_swap(self, tid, size):
        # At step 0 we swap out all tensor
        if (context.step == 0) and (size > self.tensor_size_threshold):
            self.swap_tid.add(tid)
            self.swap_size_ += size
            return True
        else:
            return (tid in self.swap_tid)
