import numpy as np
import large_model_service as lms
import experiment as exp
import sys
from termcolor import colored
import os

lms.redirect_print()

exp.init('log/multi-online')

iteration = 10
contention_workers = [0, 1]
contention_free_workers = [0, 2]


def offline_path_multi(name,batch_size,threshold, policy='multi-milp', defer=False, reprofile=False, size_threshold=0,  kernel_threshold=100, iterations=iteration, save_log=True):
    if reprofile:
        exp.run_single_profile(name, batch_size,'dynamic-early', -1, iterations=1)
    return (exp.multi_milp(name, batch_size, threshold, size_threshold=size_threshold, kernel_threshold=kernel_threshold) and \
            exp.run_multi(name, batch_size, policy, threshold, contention_workers, defer=defer, iterations=iterations, save_log=save_log))


def online_path(name, batch_size, size_threshold=0):
    exp.run_multi(name, batch_size, 'dynamic-early', -1, contention_workers, size_threshold=size_threshold, iterations=iteration)
    exp.run_multi(name, batch_size, 'dynamic-round-robin', -1, contention_workers, size_threshold=size_threshold, iterations=iteration, bucket_cap_mb=25)
    exp.run_multi(name, batch_size, 'dynamic-round-robin-defer', -1, contention_workers, size_threshold=size_threshold, defer=True, iterations=iteration, bucket_cap_mb=25)
    exp.run_multi(name, batch_size, 'dynamic-early-contention-free', -1, contention_free_workers, size_threshold=size_threshold, defer=True, iterations=iteration, bucket_cap_mb=25)


def offline_path(name, batch_size, threshold, size_threshold=0,  kernel_threshold=100):
    args = dict(size_threshold=size_threshold, kernel_threshold=kernel_threshold, iterations=iteration, save_log=True)
    offline_path_multi(name, batch_size, threshold, defer=False, **args)


def profile(name, batch_size, threshold_offline, size_threshold=0,  kernel_threshold=100):
    exp.run_multi(name, batch_size, 'dynamic-early', -1, contention_workers, size_threshold=size_threshold, iterations=iteration, profile=True)
    exp.run_multi(name, batch_size, 'multi-milp', threshold_offline, contention_workers, size_threshold=size_threshold, iterations=1, profile=True)


if __name__ == '__main__':
    batch_sizes = [52, 65, 78]
    models = ['bert-base-uncased', 'roberta-base']
    memory_thresholds = {
        'bert-base-uncased': [15, 25, 35],
        'roberta-base': [15, 25, 35]   
    }

    for model in models:
        for batch_size in batch_sizes:
            profile(model, batch_size, memory_thresholds[model], size_threshold=10)
            offline_path(model, batch_size, memory_thresholds[model], size_threshold=10)
            online_path(model, batch_size, size_threshold=10)
