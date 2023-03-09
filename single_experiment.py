import sys
import large_model_service as lms
from termcolor import colored
import experiment as exp
import os 

lms.redirect_print()
exp.init('log/single_transformer')


def online_path(name, batch_size):
    exp.run_single(name, batch_size, 'dynamic-early', -1, iterations=10)


def offline_path(name,batch_size, threshold, size_threshold=0,  kernel_threshold=100):
    if not os.path.exists(f'./experiments/{name}_{batch_size}/{name}_{batch_size}.pkl'):
        exp.run_single_profile(name, batch_size,'dynamic-early', -1, iterations = 3)
    curr = threshold
    while not (exp.single_milp(name, batch_size, curr, size_threshold=size_threshold, kernel_threshold=kernel_threshold) and \
               exp.run_single(name, batch_size, 'file', curr, iterations=10)):
        curr += 1


if __name__ == '__main__':
    batch_sizes = [52, 65, 78]

    memory_thresholds = {
        'bert-base-uncased': [15, 25, 35],
        'roberta-base':[15, 25, 35],
    }
    
    models = ['bert-base-uncased', 'roberta-base']

    # offline
    for model in models:
        for batch_size, memory_threshold in \
            zip(batch_sizes, memory_thresholds[model]):
            offline_path(model, batch_size, memory_threshold)
    
    # online
    for model in models:
        for batch_size in batch_sizes:
            online_path(model, batch_size)
