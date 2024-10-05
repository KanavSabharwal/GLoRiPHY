"""Helpful functions for project."""
import os
import sys
import torch

import numpy as np
import random
import math
from scipy.ndimage import uniform_filter1d 

def add_noise(chirp_raw,opts,snr):
    phase = random.uniform(-np.pi, np.pi) 
    chirp_raw *= (np.cos(phase)+1j*np.sin(phase)) 
    
    mwin = opts.demod.d_num_samples //2;  
    datain = to_data(chirp_raw)
    A = uniform_filter1d(abs(datain),size=mwin) 
    datain = datain[A >= max(A)/2] 
    amp_sig = torch.mean(torch.abs(torch.tensor(datain))) 
    chirp_raw /= amp_sig #normalization
    
    amp = math.pow(0.1, snr/20)
    nsamp = opts.demod.d_num_samples
    noise =  torch.tensor(amp / math.sqrt(2) * np.random.randn(nsamp) + 1j * amp / math.sqrt(2) * np.random.randn(nsamp), dtype = torch.cfloat)

    data_per = (chirp_raw + noise)
    return data_per

def spec_to_network_input(x, opts):
    """Converts numpy to variable."""
    freq_size = opts.freq_size
    trim_size = freq_size // 2
    y = torch.cat((x[:, -trim_size:, :], x[:, 0:trim_size, :]), 1)

    if opts.normalization:
        y_abs_max = torch.max(torch.abs(y).view(x.shape[0], -1), dim=1)[0].view(-1, 1, 1)
        y = y / y_abs_max


    if opts.x_image_channel == 2:
        y = torch.view_as_real(y)  # [B,H,W,2]
        y = torch.transpose(y, 2, 3)
        y = torch.transpose(y, 1, 2)
    else:
        y = torch.angle(y)  # [B,H,W]
        y = torch.unsqueeze(y, 1)  # [B,H,W]
    return y  # [B,2,H,W]

def set_gpu(free_gpu_id):
    """Converts numpy to variable."""
    torch.cuda.set_device(free_gpu_id)

def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if key in ['preamble_stft','hamming_window','device','mod','demod','preamble_conj']:
            continue
        try:
            if opts.__dict__[key]:           
                print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
        except:
            print('{:>30}: {:<30}'.format(key, 'Non-Printable').center(80))
    print('=' * 80)

class FilteredLogger(object):
    def __init__(self, logfile, stream=sys.stdout, filters=None):
        self.logfile = logfile
        self.stream = stream
        self.original_stream = stream 
        self.filters = filters if filters is not None else []

    def write(self, message):
        # Log to file unconditionally
        self.stream.write(message)
        
        if not any(f in message for f in self.filters):
            with open(self.logfile, "a") as log_file:
                log_file.write(message)

    def flush(self):
        pass

    def __del__(self):
        # Reset the stream when the logger is deleted
        if self.stream == sys.stdout:
            sys.stdout = self.original_stream
        elif self.stream == sys.stderr:
            sys.stderr = self.original_stream
