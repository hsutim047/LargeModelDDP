from collections import defaultdict

from ..utils import human_readable_size, generate_report
from ..context import context


class RoundRobinPolicy():
    def __init__(self, swap_amount_threshold, tensor_size_threshold):
        self.swap_tid = set()
        self.tensor_size_threshold = tensor_size_threshold
        self.swap_amount_threshold = swap_amount_threshold
        self.swap_size_ = 0
        self.rank = context.ipc_object.rank()

    def schedule(self, tid_size):
        self.swap_tid = set()
        current_swap_size = 0

        width = 4

        rr_idx = 0
        for tid, size in tid_size.items():
            if (size > self.tensor_size_threshold) and (current_swap_size + size < self.swap_amount_threshold):
                if ( ((rr_idx % (width * 2))//width) == self.rank):
                    self.swap_tid.add(tid)
                    current_swap_size += size
                rr_idx += 1

        self.swap_size_ = current_swap_size
        self.report()

    def report(self):
        print(list(sorted(self.swap_tid)))
        generate_report("Swapout Policy", {
            "Name": "Round-Robin",
            "Number of Swapped Out Tensors": len(self.swap_tid),
            "Total Size": human_readable_size(self.swap_size())
        })

    def swap_size(self):
        return self.swap_size_
    
    def is_last_tensor(self, tid): 
        max_id = max(self.swap_tid)     
        return tid == max_id
    
    def is_swap(self, tid, size):
        # At step 0 we swap out all tensor
        if (context.step == 0) and (size > self.tensor_size_threshold):
            return True
        else:
            return (tid in self.swap_tid)
