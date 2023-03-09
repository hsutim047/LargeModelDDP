import torch

from .spectator import spectator
from .utils import report_to_torch_profiler
from .context import context
from .storage_manager import storage_manager
from .multi_gpu import multigpu_iteration_guard


class PackedTensor:
    def __init__(self, tensor, tid):
        self.tensor = tensor
        self.size_ = tensor.numel() * tensor.element_size()
        self.ptr_ = tensor.storage().data_ptr()
        self.tid_ = tid
        self.original_device = self.tensor.device

        self.ref_cnt_ = 1

        spectator.set_tid_size(tid, self.size_)
        tensor_cache.add(self)

    def swap_out(self):
        tensor = self.tensor
        context.offload_stream.wait_stream(context.default_stream)

        spectator.increase_swap_out_size(self.size())
        spectator.timer_begin('swap_out', self.tid(), context.offload_stream)
        multigpu_iteration_guard.swap_out()
        
        with torch.cuda.stream(context.offload_stream):
            tensor.record_stream(context.offload_stream)
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(not tensor.is_sparse))
                
            packed.copy_(tensor, non_blocking=True)

        spectator.timer_end('swap_out', self.tid())
        
        # context.offload_stream.synchronize()
        
        storage_manager.alloc(packed.storage().data_ptr(), -1, self.size(), self.tid)
        
        self.tensor = packed

    def swap_in(self):
        packed = self.tensor

        spectator.timer_begin('swap_in', self.tid(), context.prefetch_stream)
        
        multigpu_iteration_guard.swap_in()

        cpu_ptr = packed.storage().data_ptr()
        with torch.cuda.stream(context.prefetch_stream):
            tensor = packed.to(self.original_device, non_blocking=True)
            context.default_stream.wait_stream(context.prefetch_stream)
        tensor.record_stream(context.default_stream)

        spectator.timer_end('swap_in', self.tid())
        storage_manager.delete(cpu_ptr, -1, self.size())

        self.tensor = tensor

    def inc(self):
        self.ref_cnt_ += 1

    def dec(self):
        self.ref_cnt_ -= 1
        if self.ref_cnt_ == 0:
            tensor_cache.remove(self)

    def is_swapped_out(self):
        return self.original_device != self.tensor.device

    def get(self):
        return self.tensor

    def size(self):
        return self.size_

    def tid(self):
        return self.tid_

    def ptr(self):
        return self.ptr_

    def ref_cnt(self):
        return self.ref_cnt_


# prevent multiple packed call on same tensor
class TensorCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cache = dict() # tid -> packed tensor

    def pack(self, tensor): # tensor -> packed tensor
        tid = storage_manager.register(tensor.storage().data_ptr())
        if tid < 0:
            return tensor
            
        report_to_torch_profiler(f"_swap_out|{tid}")
        packed = self.cache.get(tid)
        if packed is not None:
            packed.inc()
            return packed
        else:
            packed = PackedTensor(tensor, tid)
            if context.enable_swap:
                if context.policy.is_swap(packed.tid(), packed.size()):
                    packed.swap_out()
            return packed
        
    def unpack(self, packed): # packed tensor -> tensor
        report_to_torch_profiler(f"_swap_in|{packed.tid()}")
        # first time swap in
        if packed.is_swapped_out():
            packed.swap_in()
            if context.policy.is_last_tensor(packed.tid()):
                print("last swap in", packed.tid())
                event = torch.cuda.Event(enable_timing=False)
                event.record(context.prefetch_stream)
                context.last_swap_event = event

        packed.dec()

        return packed.get()

    def add(self, packed):
        self.cache[packed.tid()] = packed

    def remove(self, packed):
        self.cache.pop(packed.tid(), None)

tensor_cache = TensorCache()


def pack_hook(tensor):
    if (tensor.numel() * tensor.element_size()) < context.basic_size_threshold:
        return tensor
    spectator.record_memory_footprint()
    
    return tensor_cache.pack(tensor)

def unpack_hook(tensor):
    spectator.record_memory_footprint()

    if isinstance(tensor, PackedTensor):
        return tensor_cache.unpack(tensor)
    return tensor
