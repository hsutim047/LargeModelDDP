import re
import numpy as np
from experiment import lms

int_re = "\d+"
fp_re = "[-+]?(?:\d*\.\d+|\d+)"


def extract_from_log(regex, log):
    ret = np.array([float(i) for i in regex.findall(log)])
    return ret[1:]


def parse(log):
    log = log.decode()
    regex = {
        'memory_consumption': re.compile(f"Manageable Size\s*: ({fp_re})"),
        'memory_footprint': re.compile(f"Memory Footprint\s*: ({fp_re})"),
        'swapout_size': re.compile(f"Total Size\s*: ({fp_re})"),
        'fw_time': re.compile(f"Forward Time\s*: ({fp_re})"),
        'bw_time': re.compile(f"Backward Time\s*: ({fp_re})"),
        'elp_time': re.compile(f"Elapsed Time\s*: ({fp_re})"),
    }

    results = dict()

    for name, regc in regex.items():
        arr = extract_from_log(regc, log)
        if len(arr) > 1:
            if name == 'memory_consumption':
                results[name] = arr.max()
            else:
                results[name] = arr.mean()
        
    lms.generate_report("Exp Result", results)


def parse_rank(rank, log):
    log = log.decode()

    # parameter_size = re.compile(f"\[{rank}\]Parameter Size\s*: ({int_re})").findall(log)[0]
    regex = {
        'memory_consumption': re.compile(f"\[{rank}\] Manageable Size\s*: ({fp_re})"),
        'memory_footprint'  : re.compile(f"\[{rank}\] Memory Footprint\s*: ({fp_re})"),
        'swapout_size'      : re.compile(f"\[{rank}\] Total Size\s*: ({fp_re})"),
        'fw_time'           : re.compile(f"\[{rank}\] Forward Time\s*: ({fp_re})"),
        'bw_time'           : re.compile(f"\[{rank}\] Backward Time\s*: ({fp_re})"),
        'elp_time'          : re.compile(f"\[{rank}\] Elapsed Time\s*: ({fp_re})"),
    }

    results = dict()

    for name, regc in regex.items():
        arr = extract_from_log(regc, log)
        if name == 'memory_consumption':
            results[name] = arr.max()
        else:
            results[name] = arr.mean()

    lms.generate_report(f"Exp Result - rank{rank}", results)
