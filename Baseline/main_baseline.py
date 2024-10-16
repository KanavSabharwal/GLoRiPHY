from __future__ import print_function
from utils_baseline import create_dir, set_gpu, print_opts,FilteredLogger
import config_baseline as config
from data_loader import *
import train_baseline as jgj_train
import os
import sys
import torch
import time

from demod import DemodImpl
from mod import ModImpl


def main(opts):
    set_gpu(opts.free_gpu_id)
    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opts.sim_data:
        if opts.test_mode:
            test_dloader = lora_denoise_raw_loader_test_adjusted(opts)
        else:
            train_dloader,test_dloader = lora_denoise_raw_loader(opts)
    elif opts.real_data:
        opts.gt_dir = os.path.join(os.path.dirname(opts.data_dir),f'SF_{opts.sf}_GT')
        print("Data dir: ",opts.data_dir)
        time.sleep(2)
        if opts.test_mode:
            test_dloader = lora_real_dataset_loader_test(opts)
        else:
            train_dloader,test_dloader = lora_real_dataset_loader(opts)
    else:
        train_dloader,test_dloader = lora_loader(opts)
    
    if opts.test_mode:
        # print_opts(opts)
        jgj_train.testing_loop(test_dloader, opts)
    else:
        log_file = os.path.join(opts.checkpoint_dir,"filtered_output.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        sys.stdout = FilteredLogger(log_file, sys.stdout, filters=["Progress"]) 
        sys.stderr = FilteredLogger(log_file, sys.stderr, filters=["Progress"])
        print_opts(opts)
        jgj_train.training_loop(train_dloader,test_dloader, opts)
        
if __name__ == "__main__":
    parser = config.create_parser()
    opts = parser.parse_args()

    opts.demod = DemodImpl(opts.sf,fs_bw_ratio=opts.fs_bw_ratio,fft_factor=16)
    opts.mod = ModImpl(opts.sf)

    # STFT parameters
    opts.n_classes = 2 ** opts.sf
    opts.stft_nfft = opts.n_classes * opts.fs // opts.bw

    opts.stft_window = opts.n_classes // 2
    opts.hop_length = opts.stft_window//2 
    opts.stft_overlap = opts.stft_window // 2
    opts.conv_dim_lstm = opts.n_classes * opts.fs // opts.bw
    opts.freq_size = opts.n_classes  

    if opts.test_mode:
        checkpoint_tar = os.path.join(os.path.dirname(opts.root_path),'GLoRiPHY_source')
        opts.checkpoint_dir = os.path.join(checkpoint_tar,opts.checkpoint_dir)
    else:
        opts.checkpoint_dir = os.path.join(opts.root_path,opts.checkpoint_dir)
    # create_dir(opts.checkpoint_dir)
    opts.num_perturbations_train = int(opts.num_perturbations*0.8)
    opts.num_packets_train = int(opts.num_packets*0.85) 
    main(opts)
