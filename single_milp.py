from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import argparse
from large_model_service import generate_report, human_readable_size
import os 
import gurobipy as gp

ManageableKidxRange = namedtuple('ManageableKidxRange', ['begin', 'end'])
TimeSegment = namedtuple('TimeSegment', ['begin', 'end'])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--scheduler_input", type=str, required=True)
    parser.add_argument("--host_to_device_bandwidth", type=float, default=12.3)
    parser.add_argument("--device_to_host_bandwidth", type=float, default=13.2)
    parser.add_argument("--scheduler_output", type=str, required=True)
    
    parser.add_argument("--tensor_size_threshold", type=int, default=1, help='Minimum Tensor Size Threshold in Formulation (MiB)')
    parser.add_argument("--kernel_time_threshold", type=int, default=100, help='Minimum Kernel Time Threshold in Formulation (ms)')
    parser.add_argument("--memory_footprint_reduction", type=int, default=5000, help='Memory Footprint Reduction Constraint (GiB)')
    parser.add_argument("--time_limit", type=int, default=120)

    args = parser.parse_args()
    return args


class MILPFormulation:
    def __init__(self, info):
        self.info = info
        self.kidx_range = self.generate_kidx_range()
        
        self.peak_mem_usage = 0
        self.peak_kidx = 0
        self.total_duration = 0      
        self.original_memory_trace = self.simulate_memory_trace()
        
    def generate_kidx_range(self):
        ret = {tid: ManageableKidxRange(len(self.info.tensor_size), -1) for tid in self.info.tensor_size.keys()}

        for kidx, kernel in self.info.kernels.items():
            for tidx in kernel.dependency:
                beg_kid, end_kid = ret[tidx]
                ret[tidx] = ManageableKidxRange(min(beg_kid, kidx), max(end_kid, kidx))

        return ret
        
    def simulate_memory_trace(self):
        time, mem_usage = 0, 0
        mem_usage_trace = []

        for kidx, kernel in self.info.kernels.items():
            for tidx in kernel.dependency:
                if kidx == self.kidx_range[tidx].end:
                    mem_usage -= self.info.tensor_size[tidx]

            mem_usage_trace.append((time, mem_usage))
            if mem_usage > self.peak_mem_usage:
                self.peak_mem_usage = mem_usage
                self.peak_kidx = kidx

            time += kernel.duration
            for tidx in kernel.dependency:
                if kidx == self.kidx_range[tidx].begin:
                    mem_usage += self.info.tensor_size[tidx]

        self.total_duration = time

        return np.array(mem_usage_trace).transpose()
    
    def get_original_memory_trace(self):
        return self.original_memory_trace
        
    def prepare(self, memory_footprint_reduction, tensor_size_threshold=1e5):
        self.p = gp.Model()

        num_tensors = self.info.num_tensors()
        num_kernels = self.info.num_kernels()

        #variable definition
        self.I = self.p.addMVar((num_kernels + 1), vtype=gp.GRB.SEMICONT, name="I") # swap_in_start_time
        self.K = self.p.addMVar((num_kernels + 1), vtype=gp.GRB.SEMICONT, name="I") # kernel_start_time
        self.O = self.p.addMVar((num_kernels + 1), vtype=gp.GRB.SEMICONT, name="I") # swap_out_start_time

        # 0 : we do not swap the block
        # 1 : we swap the block
        self.d = self.p.addMVar((num_tensors), vtype=gp.GRB.BINARY, name="d")
        
        self.tensor_size_threshold = tensor_size_threshold
        self.peak_memory_constraint = self.peak_mem_usage - memory_footprint_reduction
        generate_report("MILP Formulation Info", {
            "Number of Tensors": self.info.num_tensors(),
            "Number of Kernels": self.info.num_kernels(),
            "Peak Memory Usage": human_readable_size(self.peak_mem_usage),
            "Peak KIdx": self.peak_kidx,
            "Peak Memory Constraint": human_readable_size(self.peak_memory_constraint)
        })
        
    def formulate(self):
        p, I, K, O, d = self.p, self.I, self.K, self.O, self.d
        swap_out_end_time = dict()

        time_inf = 10 * self.total_duration
        mem_usage = 0

        for kidx, kernel in self.info.kernels.items():

            # before kernel, we swap in dependencies
            dI = 0 # total swap in time
            for tidx in kernel.dependency:
                size = self.info.tensor_size[tidx]
                if kidx == self.kidx_range[tidx].end and size > self.tensor_size_threshold:
                    memory_movement_time = size/h2d_bw 
                    dI += d[tidx] * memory_movement_time
                    # if not swapped out, memory usage is decreased
                    mem_usage -= (1 - d[tidx]) * self.info.tensor_size[tidx]
                    swap_out_kidx = self.kidx_range[tidx].begin
                    # swap-in begin later than swap-out end time 
                    p.addConstr(I[kidx] >= swap_out_end_time[swap_out_kidx] - (1 - d[tidx]) * time_inf)

            p.addConstr(K[kidx] >= I[kidx] + dI)

            # we only add memory constraint at peak memory usage
            if kidx == self.peak_kidx:
                p.addConstr(mem_usage <= self.peak_memory_constraint)

            dK = kernel.duration

            # after kernel, we swap out dependencies
            p.addConstr(O[kidx] >= K[kidx] + dK)

            dO = 0  # total swap out time
            for tidx in kernel.dependency:
                size = self.info.tensor_size[tidx]
                if kidx == self.kidx_range[tidx].begin and size > self.tensor_size_threshold:
                    memory_movement_time = size/h2d_bw 
                    dO += d[tidx] * memory_movement_time
                    # if not swapped out, memory usage is increased
                    mem_usage += (1 - d[tidx]) * info.tensor_size[tidx] 

            swap_out_end_time[kidx] = O[kidx] + dO

            p.addConstr(I[kidx + 1] >= I[kidx] + dI)
            p.addConstr(K[kidx + 1] >= K[kidx] + dK)
            p.addConstr(O[kidx + 1] >= O[kidx] + dO)

        
        p.setObjective(K[self.info.num_kernels()])
        return p
    
    def solve(self, timelimit=120):
        self.p.setParam('TimeLimit', timelimit)
        solve_time_start = time.time()

        try:
            self.p.optimize()
        except:
            pass
        self.solution = milp.p.getObjective().getValue()
        generate_report("MILP Scheduler Result", {
            "Solver Elapsed Time": time.time() - solve_time_start,
            "Optimal Execution Time": self.total_duration,
            "Theretical Execution Time": self.solution,
            "Theretical Performance Loss": 1 - ((self.solution) / self.total_duration)
        })

    def summary(self):
        schedule = milp.get_schedule()
        selected = [tid for tid, s in schedule.items() if s == 1]
        selected_size = sum([self.info.tensor_size[tid] for tid in selected])
        generate_report("Summary", {
            "Number of Seleted Tensors": len(selected),
            "Selected Tensors": str(selected),
            "Size of Selected Tensors": human_readable_size(selected_size)
        })

    def get_schedule(self):
        to_dict = lambda d: {i: j for i, j in enumerate(d)}
        return to_dict(self.d.X)      
    
    def simulate_swapped_trace(self):
        p, I, K, O, d = self.p, self.I, self.K, self.O, self.d

        mem_usage = 0
        mem_usage_trace = []
        timelines = defaultdict(list)

        for kidx, kernel in info.kernels.items():
            # before kernel, we swap in dependencies
            dI = 0 # total swap in time
            for tidx in kernel.dependency:
                size = self.info.tensor_size[tidx]
                if kidx == self.kidx_range[tidx].end and size > self.tensor_size_threshold:
                    decision = p.get_values(d[tidx])
                    memory_movement_time = size/h2d_bw 
                    dI += decision * memory_movement_time
                    # if not swapped out, memory usage is decreased
                    mem_usage -= (1 - decision) * info.tensor_size[tidx]

            mem_usage_trace.append((p.get_values(K[kidx]), mem_usage))

            dO = 0  # total swap out time
            for tidx in kernel.dependency:
                size = self.info.tensor_size[tidx]
                if kidx == self.kidx_range[tidx].begin and size > self.tensor_size_threshold:
                    decision = p.get_values(d[tidx])
                    memory_movement_time = size/d2h_bw 
                    dO += decision * memory_movement_time
                    # if not swapped out, memory usage is increased
                    mem_usage += (1 - decision) * self.info.tensor_size[tidx] 

            timelines['swap-in'].append(TimeSegment(p.get_values(I[kidx]), dI))
            timelines['swap-out'].append(TimeSegment(p.get_values(O[kidx]), dO))                      
            timelines['kernel'].append(TimeSegment(p.get_values(K[kidx]), kernel.duration))
            
        return np.array(mem_usage_trace).transpose(), timelines


def kernel_filter(kernels, time_threshold=100):
    ret = dict()
    new_kid = 0
    for kid, ker in kernels.items():
        if len(ker.dependency) == 0 and ker.duration < time_threshold:
            continue 
        ret[new_kid] = ker
        new_kid += 1
    return ret


if __name__ == '__main__':
    args = get_args()

    # GB/s to bytes/ms
    h2d_bw = ((args.host_to_device_bandwidth) * 1e9) / 1e6
    d2h_bw = ((args.device_to_host_bandwidth) * 1e9) / 1e6
    tensor_size_threshold = args.tensor_size_threshold * (1 << 20) 
    kernel_time_threshold = args.kernel_time_threshold 
    memory_footprint_reduction = args.memory_footprint_reduction * (1 << 30)

    generate_report("MILP Parameters", {
        "H2D Bandwidth (bytes/ms)": h2d_bw,
        "D2H Bandwidth (bytes/ms)": d2h_bw,
        "Tensor Size Threshold": human_readable_size(tensor_size_threshold),
        "Kernel Time Threshold (ms)": kernel_time_threshold,
        "Memory Footprint Reduction": human_readable_size(memory_footprint_reduction)
    })

    with open(os.path.join(args.path, f'{args.scheduler_input}.pkl'), 'rb') as f:
        info = pickle.load(f)

    info.kernels = kernel_filter(info.kernels, kernel_time_threshold)

    milp = MILPFormulation(info)
    milp.prepare(memory_footprint_reduction , tensor_size_threshold=tensor_size_threshold)
    milp.formulate()
    milp.solve(args.time_limit)

    with open(os.path.join(args.path, f'{args.scheduler_output}.pkl'), 'wb') as f:
        schedule = milp.get_schedule()
        pickle.dump(schedule, f)

    milp.summary()
