import torch
from ..context import context


class IpcObject():
    def __init__(self):
        self.world_size = torch.cuda.device_count()
        self.barrier = torch.multiprocessing.Barrier(self.world_size)
        self.queue_ = torch.multiprocessing.SimpleQueue()
    
    def set_rank(self, rank):
        self.rank_ = rank
        self.device = f'cuda:{rank}'

    def rank(self):
        return self.rank_

    def queue(self):
        return self.queue_


class LockContext():
    switch_ratio = 1
    enable_multigpu_lock = False
    skip_sync_iterations = -1
    def init(self, switch_ratio, enable_multigpu_lock, skip_sync_iterations):
        self.switch_ratio = switch_ratio
        self.enable_multigpu_lock = enable_multigpu_lock
        self.skip_sync_iterations = skip_sync_iterations

lock_context = LockContext()


class IpcEvent():
    def __init__(self, ipc_object, stream):
        self.rank = ipc_object.rank()
        self.queue = ipc_object.queue()
        self.stream = stream
        self.device = stream.device
        if self.rank == 0:
            self.event = torch.cuda.Event(enable_timing=False, interprocess=True)

    def enter(self):
        if self.rank == 1:
            handle = self.queue.get()
            self.event = torch.cuda.Event.from_ipc_handle(self.device, handle)
            self.event.wait(self.stream)
    
    def exit(self):
        if self.rank == 0:
            self.event.record(self.stream)
            self.queue.put(self.event.ipc_handle())


class IterationGuard():
    def __init__(self):
        self.iterations = 0
        self.num_selected_tensors = 0
        self.swap_out_number = 0
        self.swap_in_number = 0

    def step(self):
        self.iterations += 1
        if context.enable_swap:
            self.num_selected_tensors = context.policy.num_selected_tensors()
            self.switch_number = int(self.num_selected_tensors * lock_context.switch_ratio)

        self.swap_out_number = 0
        self.swap_in_number = 0
    
    def swap_out(self):
        if not lock_context.enable_multigpu_lock: return

        self.swap_out_number += 1

        if self.iterations > 0:
            if self.swap_out_number == 1:
                self.event = IpcEvent(context.ipc_object, context.offload_stream)
                self.event.enter()
            elif self.swap_out_number == self.switch_number:
                self.event.exit()

    def swap_in(self):
        if not lock_context.enable_multigpu_lock: return

        self.swap_in_number += 1


multigpu_iteration_guard = IterationGuard() 
