from collections import defaultdict, namedtuple

from .spectator import spectator
from .utils import generate_report

Kernel = namedtuple('Kernel', ['name', 'duration', 'dependency'])
TimeSegment = namedtuple('TimeSegment', ['begin', 'end'])


class ProfilerInformation:
    tensor_size = dict()
    kernels = dict()

    def __init__(self, tensor_size):
        self.tensor_size = tensor_size
        self.kernels = dict()

    def push_kernel(self, kid, name, duration, dependency):
        self.kernels[kid] = Kernel(name, duration, dependency)

    def report(self):
        generate_report("Kernel Dependencies", self.kernels)

    def num_kernels(self):
        return len(self.kernels)
     
    def num_tensors(self):
        return len(self.tensor_size)


def select_event(events):
    fw, bw = [], []
    for node in events:
        if node.name.startswith('ProfilerStep'):
            fw = [e for e in node.cpu_children if not e.name.startswith('cuda')]
        elif node.name.startswith('autograd::engine::evaluate_function'):
            bw.append(node)

    return fw, bw


def dfs_duration_and_swap(evt):
    duration, swap = 0, set()
    
    if evt.name.startswith('cuda'):
        return duration, swap
    
    if len(evt.kernels) > 0:
        for ker in evt.kernels:
            if not (ker.name.startswith('Memcpy HtoD') or ker.name.startswith('Memcpy DtoH')):
                duration += ker.duration
                
    if evt.name.startswith('_swap_'):
        swap.add(int(evt.name.split('|')[-1]))
    
    for children in evt.cpu_children:
        c_d, c_s = dfs_duration_and_swap(children)
        duration += c_d
        swap.update(c_s)
        
    return duration, swap


def extract_information(events, tid_size):
    info = ProfilerInformation(tid_size)

    kid = 0
    fw, bw = select_event(events)
    for evt in fw:
        duration, dep = dfs_duration_and_swap(evt)
        if duration > 0:
            info.push_kernel(kid, evt.name, duration, dep)
            kid += 1

    for evt in bw:
        duration, dep = dfs_duration_and_swap(evt)
        name = evt.name.split()[-1]
        if duration > 0:
            info.push_kernel(kid, name, duration, dep)
            kid += 1

    return info


def parse_event(events):
    info = extract_information(events, spectator.get_tid_size())

    generate_report("Torch Profiler Information", {
        "Number of Tensors": info.num_tensors(),
        "Number of Kernels": info.num_kernels(),
    })

    return info
