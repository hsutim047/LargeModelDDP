from experiment import lms

from .context import experiment_context as ec
from .parser import parse, parse_rank
import subprocess

python_path = '/root/miniconda3/envs/largemodelddp/bin/python'
script_root = '/root/LargeModelDDP/'
transformers = ['bert-base-uncased', 'roberta-base']


def run_single(model_name, batch_size, policy, threshold, iterations=10, device=0, size_threshold=10, profile=False):
    lms.generate_report("Experiment Arguments", {
        "model_name":model_name,
        "batch_size":batch_size,
        "iterations":iterations,
        "policy": policy,
        "threshold":threshold,
        "device":device
    })

    if policy == 'early':
        arg = [
            "--policy", 'early',
            "--tensor_size_threshold", str(size_threshold),
            "--swap_amount_threshold", str(threshold)]
    elif policy == 'dynamic-early':
        arg = [
            "--policy", 'dynamic-early',
            "--tensor_size_threshold", str(size_threshold)]
    elif policy == 'file':
        arg =  [
            "--policy", "file",
            "--import_path",  f"./experiments/{model_name}_{batch_size}",
            "--scheduler_input",  f"{model_name}_{batch_size}",
            "--scheduler_output", f"{model_name}_{batch_size}.schedule"
        ]

    if profile:
        arg.extend([
            "--profile",
            "--path", f"/root/experiments/{model_name}_{batch_size}" ,
            "--filename", f"{model_name}_{batch_size}_{threshold}_{policy}.execution.profile"
        ])

    script = "single_transformer.py"

    script = f"{script_root}/{script}"
    process = subprocess.run(' '.join([
        python_path, 
        script,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--iterations", str(iterations),
        *arg
    ]), env={"CUDA_VISIBLE_DEVICES": str(device)}, capture_output=True, shell=True)

    if int(process.returncode) == 0:
        parse(process.stderr)
        with open(f'{ec.log_output_dir}/{model_name}_{batch_size}_{policy}_{threshold}.log', 'wb') as f:
            f.write(process.stderr)
        return True
    else:
        if process.stderr.find(b'CUDA out of memory') != -1:
            print("failed!", f"CUDA out of memory.")
        else:
            print("failed!, Unknown error.")
            print(process.stderr.decode())
        return False


def run_single_profile(model_name, batch_size, policy, threshold, iterations=1, device=0, size_threshold=10):
    lms.generate_report("Experiment Arguments", {
        "model_name":model_name,
        "batch_size":batch_size,
        "iterations":iterations,
        "policy": policy,
        "threshold":threshold,
    })

    script = "single_transformer.py"

    script = f"{script_root}/{script}"
    process = subprocess.run(' '.join([
        python_path, 
        script,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--iterations", str(iterations),
        "--policy", policy,
        "--tensor_size_threshold", str(size_threshold),
        "--swap_amount_threshold", str(threshold),
        "--profile",
        "--path", f"/root/experiments/{model_name}_{batch_size}" ,
        "--filename", f"{model_name}_{batch_size}",
    ]), env={"CUDA_VISIBLE_DEVICES": str(device)}, capture_output=True, shell=True)

    if int(process.returncode) == 0:
        return True
    else:
        if process.stderr.find(b'CUDA out of memory') != -1:
            print("failed!", f"CUDA out of memory")
        else:
            print("failed!, Unknown error.")
            print(process.stderr)
        return False


def run_multi(
    model_name,
    batch_size,
    policy,
    threshold,
    device,
    iterations=10,
    size_threshold=10,
    defer=False,
    save_log=True,
    profile=False,
    multigpu_lock=False,
    skip_sync_iterations=-1,
    bucket_cap_mb=25
):
    lms.generate_report("Experiment Arguments", {
        "model_name":model_name,
        "batch_size":batch_size,
        "iterations":iterations,
        "policy": policy,
        "threshold":threshold,
        "device":str(device),
        "skip_sync_iterations":skip_sync_iterations
    })

    if policy.startswith('early'):
        arg = [
            "--policy", "early,early",
            "--tensor_size_threshold", str(size_threshold),
            "--swap_amount_threshold", str(threshold)]
    elif policy.startswith('dynamic-early'):
        arg = [
            "--policy", "dynamic-early,dynamic-early",
            "--tensor_size_threshold", str(size_threshold)]
    elif policy.startswith('round-robin'):
        arg = [
            "--policy", "round-robin",
            "--tensor_size_threshold", str(size_threshold),
            "--swap_amount_threshold", str(threshold)]
    elif policy.startswith('dynamic-round-robin'):
        arg = [
            "--policy", "dynamic-round-robin",
            "--tensor_size_threshold", str(size_threshold)]
    elif policy.startswith('single-milp'):
        arg =  [
            "--policy", "file,file",
            "--import_path",  f"/root/experiments/{model_name}_{batch_size}",
            "--scheduler_input",  f"{model_name}_{batch_size}",
            "--scheduler_output", f"{model_name}_{batch_size}.schedule"
        ]
    elif policy.startswith('multi-milp'):
        arg =  [
            "--policy", "file,file",
            "--import_path",  f"/root/experiments/{model_name}_{batch_size}",
            "--scheduler_input",  f"{model_name}_{batch_size}",
            "--scheduler_output", f"{model_name}_{batch_size}.schedule.rank0,{model_name}_{batch_size}.schedule.rank1"
        ]
    else:
        raise Exception("Unknown Policy")

    if profile:
        arg.extend([
            "--profile",
            "--path", f"./experiments/{model_name}_{batch_size}" ,
            "--filename", f"{model_name}_{batch_size}_{threshold}_{policy}.execution.profile"
        ])

    if skip_sync_iterations > 0:
        arg.extend([
            "--skip_sync_iterations", str(skip_sync_iterations),
        ])
    
    if multigpu_lock:
        arg.extend([
            "--multigpu_lock",
        ])
    
    if defer:
        arg.extend([
            "--defer_collective"
        ])

    script = "multi_transformer.py"

    script = f"{script_root}/{script}"
    command = ' '.join([
        python_path, 
        "-m", "torch.distributed.launch",
        script,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--iterations", str(iterations),
        "--bucket_cap_mb", str(bucket_cap_mb),
        *arg
    ])

    process = subprocess.run(command, env={"CUDA_VISIBLE_DEVICES": f"{device[0]},{device[1]}"}, capture_output=True, shell=True)

    if int(process.returncode) == 0:
        for rank in range(2):
            parse_rank(rank, process.stderr)
        
        if save_log:
            path = f'{ec.log_output_dir}/{model_name}_{batch_size}_{threshold}_{policy}.multi.log'
            if skip_sync_iterations > 0:
                path = f'{ec.log_output_dir}/{model_name}_{batch_size}_{threshold}_{policy}_skip{skip_sync_iterations}.multi.log'

            with open(path, 'wb') as f:
                f.write(process.stderr)
        return True
    else:
        if process.stderr.find(b'CUDA out of memory') != -1:
            print("failed!", f"CUDA out of memory.")
        else:
            print("failed!, Unknown error.")
            print(process.stderr.decode())
        return False
