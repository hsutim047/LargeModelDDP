import argparse

from .main import IpcObject, lock_context, multigpu_iteration_guard
from ..context import context

def init_multi(args, ipc_object, skip_sync_iterations):
    if not ipc_object:
        raise ValueError('IPC Object is required if multi-gpu is enabled.')
    context.ipc_object = ipc_object

    lock_context.init(args.switch_ratio, args.multigpu_lock, skip_sync_iterations)

def add_arguments(parser):
    parser.add_argument("--switch_ratio", type=float, default=1)
    parser.add_argument('--multigpu_lock', default=False, action=argparse.BooleanOptionalAction)