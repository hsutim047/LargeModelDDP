import torch
import sys
from .context import context


def redirect_print():
    _print = print  

    def ranked_stderr_print(*args, **options):
        _print(f"[{context.ipc_object.rank()}]", *args, **options, file=sys.stderr)

    def stderr_print(*args, **options):
        _print(*args, **options, file=sys.stderr)

    import builtins

    if context.enable_multi_gpu:
        builtins.print = ranked_stderr_print
    else:
        builtins.print = stderr_print


def generate_report(name, log):
    key_length = max(map(len, log.keys())) + 1
    lines = [f"{k.ljust(key_length, ' ')} : {str(v)}" for k, v in log.items()]
    # log_length = max(max(map(len, lines)), len(name))
    log_length = 70
    print(name.center(log_length, "="))
    for line in lines:
        print(line)
    print("=" * log_length )


def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report_to_torch_profiler(name):
    with torch.autograd.profiler.record_function(name):
        pass
