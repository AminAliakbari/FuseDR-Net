import random
import torch
import shutil
import numpy as np
import time
import copy
import os
import argparse
import subprocess
import torch.backends.cudnn as cudnn
import glob
import logging
import sys
import yaml
import json


# Find the largest existing graphics card number
def get_gpus_memory_info():
    """Get the maximum free usage memory of gpu"""
    rst = subprocess.run('nvidia-smi -q -d Memory', stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8') #shows all data associated with GUP usage
    rst = rst.strip().split('\n')
    #memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2], it rases an error because it has N/A value in it.
    memory_available = []
    for line in rst:
        if 'Free' in line:
            try:
                # Split at ':', get the right-hand side, strip spaces
                value_str = line.split(':')[1].strip()
                if value_str != 'N/A':
                    value = int(value_str.split()[0])  # Get the number, skip 'MiB' if present
                    memory_available.append(value)
            except (IndexError, ValueError):
                continue  # Skip malformed or non-numeric lines

    id = int(np.argmax(memory_available))

    return id, memory_available

# Set seeds for random, numpy, and torch (CPU and CUDA)
def set_seed(seed):
    """
    set seed of numpy and torch
    :param seed:
    :return:
    """
    if seed is None:
        seed = np.random.randint(1e6)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # To prevent hash randomization and make the experiment reproducible。
    torch.manual_seed(seed) # Set a random seed for the CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Set a random seed for the current GPU
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，Set a random seed for all GPUs
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed

def print_log(print_string, log_file, visible=True):
    if visible:
        print("{}".format(print_string))
    # Write to log file
    log_file.write('{}\n'.format(print_string))
    # Flush the buffer to write the data
    log_file.flush()