from experiment import lms
import subprocess

python_path = '/root/miniconda3/envs/largmodelddp/bin/python'
script_root = '/root/LargeModelDDP/'


def single_milp(model_name, batch_size, threshold, size_threshold=0, kernel_threshold=100):
        lms.generate_report("Single MILP", {
            "model_name": model_name,
            "batch_size": batch_size,
            "memory footprint threshold": threshold,
            "tensor size threshold": size_threshold,
            "kernel time threshold": kernel_threshold
        })

        command = ' '.join([
            python_path, 
            f"{script_root}/single_milp.py",
            "--path", f"/root/experiments/{model_name}_{batch_size}",
            "--scheduler_input", f"{model_name}_{batch_size}",
            "--scheduler_output", f"{model_name}_{batch_size}.schedule",
            "--tensor_size_threshold", f"{size_threshold}",
            "--memory_footprint_reduction", f"{threshold}",
            "--kernel_time_threshold", f"{kernel_threshold}",
            "--time_limit", "120",
            "--host_to_device_bandwidth", "11.3",
            "--device_to_host_bandwidth", "12.2",
        ])
        process = subprocess.run(command, capture_output=True, shell=True)

        if int(process.returncode) == 0:
            return True
        else:
            print("failed!, Unknown error.")
            print(process.stderr.decode())
            return False


def multi_milp(model_name, batch_size, threshold, size_threshold=0, kernel_threshold=100, time_limit=120):
        lms.generate_report("Multi MILP", {
            "model_name": model_name,
            "batch_size": batch_size,
            "memory footprint threshold": threshold,
            "tensor size threshold": size_threshold,
            "kernel time threshold": kernel_threshold
        })

        command = ' '.join([
            python_path, 
            f"{script_root}/multi_milp.py",
            "--path", f"/root/experiments/{model_name}_{batch_size}",
            "--scheduler_input", f"{model_name}_{batch_size}",
            "--scheduler_output", f"{model_name}_{batch_size}.schedule",
            "--tensor_size_threshold", f"{size_threshold}",
            "--memory_footprint_reduction", f"{threshold}",
            "--kernel_time_threshold", f"{kernel_threshold}",
            "--time_limit", f"{time_limit}",
            "--host_to_device_bandwidth", "11.3",
            "--device_to_host_bandwidth", "12.2",
        ])
        process = subprocess.run(command, capture_output=True, shell=True)

        if int(process.returncode) == 0:
            return True
        else:
            print("failed!, Unknown error.")
            print(process.stderr.decode())
            return False
