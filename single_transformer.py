import torch
import os
import numpy as np
import time
import sys
import large_model_service as lms
import argparse
from contextlib import nullcontext
from transformers import GPT2Config, BertConfig, RobertaConfig, AlbertConfig
from torchinfo import summary


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", choices=[
        'bert-base-uncased','bert-large-uncased',
        'roberta-base', 'roberta-large',
        'albert-base-v2','albert-large-v2',
        'gpt2','gpt2-medium','gpt2-large','gpt2-xl',
    ], required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument('--swap', action='store_true', help='Enable Large Model Swapping Mechanism')
    parser.add_argument('--device', type=str, default='cuda:0')
    lms.add_arguments(parser)    
    
    args = parser.parse_args()
    return args


def get_vocab_size(model_name):
    if 'bert' in model_name.lower():
        return BertConfig().vocab_size
    elif 'gpt2' in model_name:
        return GPT2Config().vocab_size
    elif 'robeta' in model_name:
        return RobertaConfig().vocab_size
    elif 'albert' in model_name:
        return AlbertConfig().vocab_size


def train(device, iterations, batch_size, model, optimizer, num_labels, vocab_size):
    itertime = []
    for _ in range(iterations):
        labels = torch.randint(num_labels, (1, batch_size), device=device)
        input_ids = torch.randint(vocab_size, (batch_size, 512), device=device)
        events = [torch.cuda.Event(enable_timing=True) for i in range(5)]

        with lms.SwapManager() as sm:
            events[0].record()
            output = model(labels=labels, input_ids=input_ids)
            events[1].record()
            output.loss.backward()
            events[2].record()
            optimizer.step()
            optimizer.zero_grad()
            events[3].record()

        torch.cuda.synchronize(device=device)
        t = sm.iteration_elapsed_time()
        itertime.append(t)

        lms.generate_report("Time", {
            "Forward Time": events[0].elapsed_time(events[1])/1e3,
            "Backward Time": events[1].elapsed_time(events[2])/1e3,
            "Optimize Time": events[2].elapsed_time(events[3])/1e3,
            "Elapsed Time": t,            
        })
        
        del output, labels, input_ids

    return itertime


def prepare(args):
    model_name = args.model_name
    iterations = args.iterations
    batch_size = args.batch_size
    num_labels = args.num_labels
    device = args.device

    lms.init(device, args)

    vocab_size = get_vocab_size(args.model_name)
    model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 
                            model_name,
                            num_labels=num_labels)
    summary(model)

    if 'gpt2' in model_name:
        model.config.pad_token_id = model.config.eos_token_id
        
    model = model.train().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    lms.generate_report("Model Basic Information", {
        "Parameter Size": lms.count_parameters(model),
    })

    itertime = train(device, iterations, batch_size, model, optimizer, num_labels, vocab_size)
    itertime = itertime[1:]

    print(f"Train Finished!")
    print(f"quartiles (Q1, Q2, Q3) = ({np.percentile(itertime, 25)}, {np.percentile(itertime, 50)}, {np.percentile(itertime, 75)})")


if __name__ == '__main__':
    args = get_args()
    prepare(args)
