"""Helpful functions for project."""
import os
import sys
import torch
from torch.autograd import Variable

from random import shuffle
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

def increase_dropout(model, increase_amount=0.05):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            new_dropout_rate = module.p + increase_amount
            module.p = min(new_dropout_rate, 1.0) 
            print(f"Updated Dropout p to {module.p}")

def spec_to_network_input(x, opts):
    """Converts numpy to variable."""
    freq_size = opts.freq_size
    trim_size = freq_size if opts.dechirp else freq_size // 2                                                                                                      
    
    y = torch.cat((x[:, -trim_size:, :], x[:, :trim_size, :]), dim=1)

    if opts.normalization:
        y_abs_max = torch.max(torch.abs(y).view(x.shape[0], -1), dim=1)[0].view(-1, 1, 1)
        y = y / y_abs_max  # Element-wise division for normalization

    if opts.x_image_channel == 2:
        y = torch.view_as_real(y)  # Convert complex to real representation
        y = y.permute(0, 3, 1, 2) 
    else:
        y = torch.angle(y)  # [B,H,W]
        y = y.unsqueeze(1)  # [B,H,W]
    return y  # [B,2,H,W]

def spec_to_network_input_complex(x, opts):
    """Converts numpy to variable."""
    freq_size = opts.freq_size
    trim_size = freq_size if opts.dechirp else freq_size // 2
    
    y = torch.cat((x[:, -trim_size:, :], x[:, :trim_size, :]), dim = 1)

    if opts.normalization:
        y_abs_max = torch.max(torch.abs(y).view(x.shape[0], -1), dim=1)[0].view(-1, 1, 1)
        y = y / y_abs_max
    return y  # [B,2,H,W]

def torch_fpmod(x, n):
    return (x % n + n) % n

def torch_demod_symbol_batched(symbols, opts):
    # Assume symbols is a batch of signals [batch_size, d_num_samples]
    symbols *= opts.downchirp 
    fft_input = torch.zeros((symbols.shape[0], opts.demod.d_fft_size), dtype=torch.complex64, device=symbols.device)
    fft_input[:, :opts.demod.d_num_samples] = symbols
    fft_res = torch.fft.fft(fft_input)
    buffer1 = torch.abs(fft_res)
    buffer2 = buffer1[:, :opts.demod.d_bin_size] + buffer1[:, -opts.demod.d_bin_size:]
    max_indices = torch.argmax(buffer2, dim=1)
    return max_indices

def soft_argmax(x, beta=10):
    weights = torch.nn.functional.softmax(beta * x, dim=1)
    indices = torch.arange(x.size(1), device=x.device).float()
    soft_max_index = torch.sum(weights * indices, dim=1)
    return soft_max_index

def reconstruct_from_stft_batched(processed_symbol, opts):
    # Convert real and imaginary parts to a complex tensor
    processed_symbol = torch.view_as_complex(processed_symbol.permute(0, 2, 3, 1).contiguous())

    # Prepare the complex tensor for ISTFT
    shape = (opts.batch_size, opts.demod.d_num_samples, 33)
    trim_size = opts.freq_size // 2
    x = torch.zeros(shape, dtype=torch.complex64, device=processed_symbol.device)
    x[:, -trim_size:, :] = processed_symbol[:, :trim_size, :]
    x[:, :trim_size, :] = processed_symbol[:, trim_size:, :]

    # Apply batched ISTFT
    reconstructed_signals = torch.istft(x,
                                        n_fft=opts.stft_nfft,
                                        hop_length=opts.hop_length,
                                        win_length=opts.stft_window,
                                        window=opts.hamming_window,  # Ensure window is on the correct device
                                        return_complex=True)

    # Batch process demodulation and bin index calculation
    max_indices = torch_demod_symbol_batched(reconstructed_signals, opts)
    bin_indices = torch_fpmod(max_indices / opts.demod.d_fft_size_factor, opts.demod.d_num_symbols)
    return torch_fpmod(torch.round(bin_indices), opts.demod.d_num_symbols)

def get_channel_estimate(received_preamble, opts):    
    channel_estimate = received_preamble * opts.preamble_conj.unsqueeze(0)

    power = torch.mean(torch.abs(channel_estimate) ** 2, dim=(1, 2), keepdim=True)
    normalized_estimate = channel_estimate / torch.sqrt(power)
        
    # Convert complex to real and rearrange dimensions
    normalized_estimate = torch.view_as_real(normalized_estimate)
    normalized_estimate = normalized_estimate.permute(0, 3, 1, 2)     
    return normalized_estimate


def set_gpu(free_gpu_id):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        torch.cuda.set_device(free_gpu_id)

def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


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
