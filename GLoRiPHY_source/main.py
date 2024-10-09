from __future__ import print_function
from utils import create_dir, set_gpu, print_opts, FilteredLogger
import config
from data_loader import *
import train as jgj_train
import os
import sys
import torch
import time
from demod import DemodImpl
from mod import ModImpl


def main(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop.
    """
    set_gpu(opts.free_gpu_id)
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opts.hamming_window = torch.hamming_window(opts.stft_window).to(opts.device)
    opts.downchirp = torch.tensor(opts.demod.d_downchirp[:opts.demod.d_num_samples]).unsqueeze(0).to(opts.device)

    if opts.real_data or opts.sim_data:
        opts.gt_dir = os.path.join(os.path.dirname(opts.data_dir),f'SF_{opts.sf}_GT')
    
    opts.checkpoint_dir = opts.checkpoint_dir + '_' + str(opts.sf)
    if opts.test_mode:
        opts.checkpoint_dir = opts.checkpoint_dir + '_test'
    create_dir(opts.checkpoint_dir)

    if opts.train_denoiseGenCore:
        train_dloader,test_dloader = lora_loader(opts)
    elif opts.train_denoiseGen:
        print("Data dir: ",opts.data_dir)
        time.sleep(2)

        if opts.real_data:
            if opts.test_mode:
                test_dloader = lora_real_dataset_loader_test(opts)
            else:
                train_dloader,test_dloader = lora_real_dataset_loader(opts)
        elif opts.sim_data:
            if opts.test_mode:
                test_dloader = lora_denoise_raw_loader_test_adjusted(opts)
            else:
                train_dloader,test_dloader = lora_denoise_raw_loader(opts)
    
    log_file = os.path.join(opts.checkpoint_dir,"filtered_output.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    sys.stdout = FilteredLogger(log_file, sys.stdout, filters=["Progress"]) 
    sys.stderr = FilteredLogger(log_file, sys.stderr, filters=["Progress"])
    if not opts.test_mode:
        print_opts(opts)

    if opts.test_mode:
        if opts.train_denoiseGen:
            jgj_train.testing_loop_denoise(test_dloader, opts)
        elif opts.train_denoiseGenCore:
            jgj_train.testing_loop_denoiseCore(test_dloader, opts)
    else:
        if opts.train_denoiseGen:
            jgj_train.training_loop_denoise(train_dloader, test_dloader, opts)
        elif opts.train_denoiseGenCore:
            jgj_train.training_loop_denoiseCore(train_dloader, test_dloader, opts)


if __name__ == "__main__":
    parser = config.create_parser()
    opts = parser.parse_args()

    assert (opts.train_denoiseGen and opts.train_denoiseGenCore) == False, "Choose either train_denoiseGen or train_denoiseGenCore"

    opts.demod = DemodImpl(opts.sf,fs_bw_ratio=opts.fs_bw_ratio,fft_factor=16)
    opts.mod = ModImpl(opts.sf)

    # STFT parameters
    opts.stft_nfft = opts.demod.d_num_samples
    opts.stft_window = opts.demod.d_num_symbols//2
    opts.hop_length = opts.stft_window // 2

    opts.freq_size = opts.demod.d_num_symbols  
    opts.n_classes = opts.demod.d_num_symbols   

    opts.checkpoint_dir = os.path.join(opts.root_path,opts.checkpoint_dir)
    opts.num_perturbations_train = int(opts.num_perturbations*0.8)
    opts.num_packets_train = int(opts.num_packets*0.85) 

    main(opts)
