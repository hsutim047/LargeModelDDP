import os 
import pickle

from ..utils import human_readable_size, generate_report
from ..context import context


def load_pkl(import_path, filename):
    with open(os.path.join(import_path, f'{filename}.pkl'), 'rb') as f:
        ret = pickle.load(f)

    return ret

    
class FilePolicy():
    def __init__(self, import_path, scheduler_input_filename, scheduler_output_filename):
        self.swap_tid = set()
        self.swap_size_ = 0
        self.scheduler_input_filename = scheduler_input_filename
        self.scheduler_output_filename = scheduler_output_filename
        self.scheduler_input = load_pkl(import_path, scheduler_input_filename)
        self.scheduler_output = load_pkl(import_path, scheduler_output_filename)
        self.swap_tid = {tid for tid, d in self.scheduler_output.items() if d == 1}

    def schedule(self, tid_size):
        current_swap_size = 0
        
        for tid, size in tid_size.items():
            if self.scheduler_output.get(tid) == 1:
                if self.scheduler_input.tensor_size[tid] == size:
                    self.swap_tid.add(tid)
                    current_swap_size += size
                else:
                    print(f"Warning! Runtime Tensor size does not match scheduler tensor size at tid:{tid}")

        self.swap_size_ = current_swap_size
        self.report()

    def report(self):
        selected_tensors = ','.join([str(tid) for tid in sorted(self.swap_tid)])
        generate_report("Swapout Policy", {
            "Name": "File",
            "Scheduler Input Filename": self.scheduler_input_filename,
            "Scheduler Output Filename": self.scheduler_output_filename,
            "Selected Tensors": selected_tensors,
            "Number of Swapped Out Tensors": len(self.swap_tid),
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
        return (tid in self.swap_tid)
