from .early_first import EarlyFirstPolicy
from .dynamic_early import DynamicEarlyFirstPolicy
from .file import FilePolicy
from .round_robin import RoundRobinPolicy
from .dynamic_round_robin import DynamicRoundRobinPolicy
from ..context import context


def init_policy(args):
    if context.enable_multi_gpu:
        rank = context.ipc_object.rank()

    swap_amount_threshold = args.swap_amount_threshold
    if isinstance(swap_amount_threshold, str) and swap_amount_threshold.find(',') != -1:
        swap_amount_threshold = int(swap_amount_threshold.split(',')[rank])

    policy = args.policy
    if policy.find(',') != -1:
        policy = args.policy.split(',')[rank]

    if policy == 'early':
        swap_amount_threshold = int(swap_amount_threshold) * (1 << 30)
        tensor_size_threshold = int(args.tensor_size_threshold) * (1 << 20)
        context.policy = EarlyFirstPolicy(swap_amount_threshold, tensor_size_threshold)
    elif policy == 'dynamic-early':
        tensor_size_threshold = int(args.tensor_size_threshold) * (1 << 20)
        context.policy = DynamicEarlyFirstPolicy(tensor_size_threshold)
    elif policy == 'file':
        import_path, scheduler_input, scheduler_output  = args.import_path, args.scheduler_input, args.scheduler_output
        if scheduler_output.find(',') != -1:
            scheduler_output = scheduler_output.split(',')[rank]
        context.policy = FilePolicy(import_path, scheduler_input, scheduler_output)
    elif policy == 'round-robin':
        swap_amount_threshold = int(swap_amount_threshold) * (1 << 30)
        tensor_size_threshold = int(args.tensor_size_threshold) * (1 << 20)
        context.policy = RoundRobinPolicy(swap_amount_threshold, tensor_size_threshold)
    elif policy == 'dynamic-round-robin':
        tensor_size_threshold = int(args.tensor_size_threshold) * (1 << 20)
        context.policy = DynamicRoundRobinPolicy(tensor_size_threshold)
    else:
        raise Exception("Unrecognized policy!")


def add_arguments(parser):
    parser.add_argument("--policy", type=str)
    parser.add_argument("--swap_amount_threshold", type=str, default=0)
    parser.add_argument("--tensor_size_threshold", type=int, default=1, help='Minimum Tensor Size Threshold in Formulation (MiB)')

    parser.add_argument("--import_path", type=str)
    parser.add_argument("--scheduler_input", type=str)
    parser.add_argument("--scheduler_output", type=str)
