

class LargeModelSupportContext():
    # stream
    default_stream = None
    offload_stream = None
    prefetch_stream = None

    # single-gpu large model swapping mechanism
    enable_swap = True
    policy = None

    # basic tensor size threshold
    basic_size_threshold = 1e5 
        
    # iteration profile information
    step = 0
    total_steps = 0
    iteration_stats = list()

    # profile save path
    torch_profiler = None
    path = ""
    filename = ""

    # multi-gpu
    enable_multi_gpu = False
    ipc_object = None

    # collective defer
    last_swap_event = None

context = LargeModelSupportContext()