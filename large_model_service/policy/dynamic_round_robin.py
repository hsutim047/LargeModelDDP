from collections import defaultdict

from ..utils import human_readable_size, generate_report
from ..context import context
from ..spectator import spectator


class DynamicRoundRobinPolicy():
    def __init__(self, tensor_size_threshold):
        self.all_swap_tid = set()
        self.swap_tid = set()

        self.tensor_size_threshold = tensor_size_threshold
        self.swap_size_ = 0
        self.rank = context.ipc_object.rank()

    def schedule(self, tid_size):
        memory_footprint = spectator.get_memory_footprint()
        saved_size = max(19 * 1e9 - memory_footprint, 0)

        curr = list(sorted(self.all_swap_tid))

        current_save_size = 0
        save_idx = 0

        new_swap_tid = set()
        for tid in reversed(curr):
            size = tid_size[tid]
            if (current_save_size + size < saved_size):
                if (save_idx % 2) == self.rank:
                    new_swap_tid.add(tid)
                    current_save_size += size
                save_idx += 1
            else:
                new_swap_tid.add(tid)
        
        new_size = sum(map(lambda x: tid_size[x], new_swap_tid))
        if self.swap_size_ > new_size:
            self.swap_tid = new_swap_tid
            self.swap_size_ = new_size

        self.save_size = saved_size
        self.report()

    def report(self):
        selected_tensors = ','.join([str(tid) for tid in sorted(self.swap_tid)])
        generate_report("Swapout Policy", {
            "Name": "Dynamic-Round-Robin",
            "Number of Swapped Out Tensors": len(self.swap_tid),
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
        if (size < self.tensor_size_threshold):
            return False

        # At step 0 we swap out all tensor
        if context.step == 0:
            self.all_swap_tid.add(tid)
            self.swap_tid.add(tid)
            self.swap_size_ += size
            return True
        else:
            return (tid in self.swap_tid)
