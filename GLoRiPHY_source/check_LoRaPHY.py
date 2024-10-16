import os
import config
from data_loader import lora_loader
from demod import DemodImpl,fpmod
from mod import ModImpl
from tqdm import tqdm
import numpy as np
import csv

def get_LoRaPHY_BER(opts,test_dloader):
    count = 0
    progress_bar = tqdm(test_dloader, desc=f"LoRaPHY Test Progress", leave=False)
    for (symbol,_), labels in progress_bar:
        max_idx = opts.demod.demod_symbol(symbol)
        bin_idx = fpmod((max_idx) / opts.demod.d_fft_size_factor, opts.demod.d_num_symbols)
        bin_idx = np.round(bin_idx).astype(int)
        bin_idx = fpmod(bin_idx, opts.demod.d_num_symbols)

        if bin_idx != labels:
            count+=1

    ber = count*100/opts.test_data_size
    return ber
 
if __name__ == "__main__":
    parser = config.create_parser()
    opts = parser.parse_args()

    opts.demod = DemodImpl(opts.sf,fs_bw_ratio=opts.fs_bw_ratio,fft_factor=16)
    opts.mod = ModImpl(opts.sf)
    opts.batch_size = 1

    opts.stft_nfft = opts.demod.d_num_samples
    opts.stft_window = opts.demod.d_num_symbols//2
    opts.hop_length = opts.stft_window // 2
    opts.freq_size = opts.demod.d_num_symbols  
    opts.n_classes = opts.demod.d_num_symbols   

    opts.checkpoint_dir = os.path.join(opts.root_path,opts.checkpoint_dir)
    log_file_path = os.path.join(opts.checkpoint_dir,'testing_log.csv')

    _,test_dloader = lora_loader(opts)
    ber = get_LoRaPHY_BER(opts,test_dloader)
    
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['LoRaPHY',opts.snr_list[0],ber])