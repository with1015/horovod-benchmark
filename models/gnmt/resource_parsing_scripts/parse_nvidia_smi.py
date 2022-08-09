import os
import socket
import argparse

import torch


parser = argparse.ArgumentParser(description='Average GPU Utilization')
parser.add_argument('--log-dir', type=str, default=None, required=True,
                    help='A dir path for nvidia-smi log')


def get_avg(input):
    if not isinstance(input, (tuple, list)):
        raise ValueError(
            f"Type of argument input must be list or tuple: {type(input)}")
    if len(input) == 0:
        return 0
    digit = 2
    return round(sum(input) / len(input), digit)


def get_avg_gpu_util(filename, threshold=10, bound=2):
    active_gpu_util = []
    start_step = 0
    if filename == '':
        raise FileExistsError(f"No such file exists: {filename}")
    else:
        line_data = []
        with open(filename, 'r') as f:
            for line in f:
                line_data.append(line)

        for line_num, line_str in enumerate(line_data):
            if "timestamp" in line_str:  # The first line demonstrating row data
                continue
            g_util = int(line_str.split(',')[2].strip()[:-1])
            # Since we want to skip initial warm-up time, check if util is over threshold.
            if g_util > threshold:
                start_step += 1
                if start_step <= bound:
                    continue
                active_gpu_util.append(g_util)
        # At the end of program, GPU utilization decreases so we skip it
        active_gpu_util = active_gpu_util[:len(active_gpu_util)-bound]
        print(f'Active GPU utilization: {active_gpu_util}')
        avg_gpu_util = get_avg(active_gpu_util)
        return avg_gpu_util


def main():
    args = parser.parse_args()
    if not os.path.isdir(args.log_dir):
        print(f'No such directory - args.log_dir: {args.log_dir}')
        exit(1)
    filelist = os.listdir(args.log_dir)
    filelist.sort()
    filelist = [os.path.join(args.log_dir, file) for file in filelist if 'csv' in file]
    if not filelist:
        print(f'No csv file in log dir: {filelist}')
        exit(1)
    host = socket.gethostname()[:-1]
    ngpus_per_node = torch.cuda.device_count()
    gpu_filelist = ['' for _ in range(ngpus_per_node)]
    gpu_avg = ['' for _ in range(ngpus_per_node)]
    for f in filelist:
        if "gpu_stat" in f:
            print(f"log file: {f}")
            device_num = int(f.strip().split("device")[1][0])
            gpu_filelist[device_num] = f
            gpu_avg[device_num] = get_avg_gpu_util(f)
    """
    print format
    [device0, device1, device2, device3]
    e.g, [90.5, 95, 92, 93]
    """
    print(gpu_avg)


if __name__ == "__main__":
    main()
