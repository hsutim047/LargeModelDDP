import torch
from collections import namedtuple

from .context import context

StorageInfo = namedtuple('StorageInfo', ['tid', 'device', 'size'])
# size here is storage size but not tensor size


class StorageManager:
    def __init__(self, basic_size_threshold):
        self.storage_manager_basic_size_threshold = basic_size_threshold
        self.ptr_info = dict()
        self.reset()

    def reset(self):
        self.current_tid = 0
        for ptr, info in self.ptr_info.items():
            self.ptr_info[ptr] = StorageInfo(-1, info.device, info.size)

    def step(self):
        self.reset()

    def register(self, ptr):
        info = self.ptr_info.get(ptr)

        if info:
            if info.tid != -1:
                # print("register", info.tid)
                return info.tid

            new_info = StorageInfo(self.current_tid, info.device, info.size) 
            self.ptr_info[ptr] = new_info
            self.current_tid += 1

            # print("register", new_info.tid)
            return new_info.tid
        else:
            print('Warning! cannot found pointer')
            return -1

    def get_num_tensors(self):
        return self.current_tid

    def get_storage_size(self):
        return {storage.tid:storage.size for storage in self.ptr_info.values()}

    def get_total_size(self):
        return sum([storage.size for storage in self.ptr_info.values()])

    # the below function is hooked to memory pool, so it is performance critical
    def alloc(self, ptr, device, size, tid=-1):
        if size > self.storage_manager_basic_size_threshold:
            # print("malloc", ptr, size)
            self.ptr_info[ptr] = StorageInfo(tid, device, size) 

    def delete(self, ptr, device, size):
        if size > self.storage_manager_basic_size_threshold:
            # print("delete", ptr, size)
            self.ptr_info.pop(ptr, None)

storage_manager = StorageManager(context.basic_size_threshold)


def alloc_hook(ptr, device, size):
    storage_manager.alloc(ptr, device, size)


def delete_hook(ptr, device, size):
    storage_manager.delete(ptr, device, size)
    
torch._C._autograd._register_alloc_delete_hook(alloc_hook, delete_hook)
