import torch
import numpy as np
import large_model_service as lms
import argparse
from transformers import GPT2Config, BertConfig, RobertaConfig, AlbertConfig
from torchinfo import summary
import torch.distributed as dist
import numpy as np
import argparse
from torch.distributed.optim import ZeroRedundancyOptimizer


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
    parser.add_argument("--bucket_cap_mb", type=int, default=25)
    parser.add_argument('--defer_collective', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--skip_sync_iterations", type=int, default=-1)
    parser.add_argument('--swap', default=True, action=argparse.BooleanOptionalAction, help='Enable Large Model Swapping Mechanism')
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


def train(device, iterations, batch_size, model, optimizer, num_labels, vocab_size, skip_sync_iterations):
    itertime = []
    for i in range(iterations):
        if skip_sync_iterations > 0:
            mod_iter = (i % skip_sync_iterations)
            if mod_iter == 0:
                model.require_backward_grad_sync = False
            elif mod_iter == skip_sync_iterations - 1:
                model.require_backward_grad_sync = True

        labels = torch.randint(num_labels, (1, batch_size), device=device)
        input_ids = torch.randint(vocab_size, (batch_size, 512), device=device)
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

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


def get_quartiles(x):
    x = np.array(x)
    q1, q2, q3 = np.percentile(x, 25), np.percentile(x, 50), np.percentile(x, 75)
    return f"{q1:.8f}, {q2:.8f}, {q3:.8f}"


def comm_hook(ncclSream, bucket):
    size = lambda tensor: tensor.numel() * tensor.element_size()

    if lms.context.last_swap_event:
        ncclSream.wait_event(lms.context.last_swap_event)
    fut = torch.futures.Future()
    buffer = bucket.buffer()
    print("comm_hook: ", lms.human_readable_size(size(buffer)))
    fut.set_result(buffer)    
    return fut


def prepare(rank, ipc_object, args):
    model_name = args.model_name
    iterations = args.iterations
    batch_size = args.batch_size
    num_labels = args.num_labels
    skip_sync_iterations = args.skip_sync_iterations

    ipc_object.set_rank(rank)
    device = ipc_object.device
    torch.cuda.set_device(rank)

    lms.init(device, args, enable_multi_gpu=True, ipc_object=ipc_object, skip_sync_iterations=skip_sync_iterations)

    dist.init_process_group(backend='nccl', world_size=ipc_object.world_size, rank=rank)

    vocab_size = get_vocab_size(args.model_name)
    model = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification', 
                            model_name,
                            num_labels=num_labels)
    summary(model)    
    if 'gpt2' in model_name:
        model.config.pad_token_id = model.config.eos_token_id

    model = model.train().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], bucket_cap_mb=args.bucket_cap_mb) 
    # Tune bucket_cap_mb to a large number to disable the effect of gradient bucketing
    # 03-07 experiment shows the effect is small (bucket_cap_mb=10, bucket_cap_mb=1000)
    packed_stream = torch._C._distributed_c10d.ProcessGroupNCCL.get_nccl_stream(model.process_group, rank)
    ncclStream = torch.cuda.Stream(_cdata=packed_stream)
    if args.defer_collective:
        model.register_comm_hook(state=ncclStream, hook=comm_hook)

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = ZeroRedundancyOptimizer(
    #     model.parameters(),
    #     optimizer_class=torch.optim.Adam,
    # )

    itertime = train(device, iterations, batch_size, model, optimizer, num_labels, vocab_size, skip_sync_iterations)
    itertime = itertime[1:]

    dist.barrier()

    lms.generate_report("Quartiles (Q1, Q2, Q3)", {
        "Iteration Time (s)": get_quartiles(itertime),
    })

    dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args()

    torch.multiprocessing.set_start_method('spawn') 
    ipc_object = lms.IpcObject()
    torch.multiprocessing.spawn(prepare, nprocs=ipc_object.world_size, args=(ipc_object, args))
