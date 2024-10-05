import os 
import csv
import random 
import numpy as np
from scipy.io import loadmat
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch 
from torch.utils.data import DataLoader 
from torch.utils import data 
from utils import *

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def get_sim_files(opts):    
    gt_codewords = []
    gt_spec = []
    perturbed_packets_train = []
    perturbed_packets_test = []

    obj = os.scandir(opts.data_dir)

    for entry in obj:
        if entry.is_file() and entry.name != 'params.mat' and 'preamble' not in entry.name:
            parts = entry.name.split('_')
            packet_number = int(parts[1])
            entry_type = parts[2]

            if packet_number <= opts.num_packets:
                if entry_type == "codewords":
                    gt_codewords.append(entry.name)

                elif entry_type == 'GT':
                    gt_spec.append(entry.name)
                
                elif entry_type == 'RayleighAWGN':
                    perturbation_number = int(parts[3])
                    if perturbation_number <= opts.num_perturbations_train:
                        perturbed_packets_train.append(entry.name)
                    elif perturbation_number <= opts.num_perturbations:
                        perturbed_packets_test.append(entry.name)

    return gt_codewords, gt_spec, perturbed_packets_train, perturbed_packets_test 

def get_real_files(opts):
    gt_codewords = []
    perturbed_packets_train = []
    perturbed_packets_test = []

    obj = os.scandir(opts.data_dir)

    for entry in obj:
        if entry.is_file():
            parts = entry.name.split('_')
            packet_number = int(parts[1])
            entry_type = parts[2]

            if entry_type == "codewords":
                gt_codewords.append(entry.name)
                
            elif entry_type != 'GT':
                node_id = int(parts[2])
                if packet_number <= opts.num_packets_train and node_id not in opts.test_nodes:
                    perturbed_packets_train.append(entry.name)
                else:
                    perturbed_packets_test.append(entry.name)
    
    obj = os.scandir(opts.gt_dir)
    for entry in obj:
        if entry.is_file():
            parts = entry.name.split('_')
            packet_number = int(parts[1])
            entry_type = parts[2]

            if entry_type == "codewords":
                gt_codewords.append(entry.name)
    
    return gt_codewords, perturbed_packets_train, perturbed_packets_test

#  Pureley AWGN Data Loader
class lora_dataset(data.Dataset): 
    'Characterizes a dataset for PyTorch' 
 
    def __init__(self, opts, mode = 'train'): 
        # Simulating ideal symbols
        self.opts = opts 
        self.chirp_ideal_dict = dict()
        self.chirp_ideal_stft_dict = dict()
        self.dataset_size = opts.train_data_size if mode == 'train' else opts.test_data_size
        for ind in range(1,opts.demod.d_num_symbols+1):
            chirp = torch.tensor(opts.mod.gen_symbol(ind), dtype=torch.cfloat)
            self.chirp_ideal_dict[opts.demod.d_num_symbols-ind] = chirp
            
            ideal_stft = torch.stft(chirp, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, pad_mode='constant',window=torch.hamming_window(opts.stft_window))
            self.chirp_ideal_stft_dict[opts.demod.d_num_symbols-ind] = torch.squeeze(spec_to_network_input(ideal_stft.unsqueeze(0), opts))

        if mode == 'train':
            set_seed(77)
        else:
            set_seed(7)

    def __len__(self): 
        return self.dataset_size
  
    def __getitem__(self,index): 
        try: 
            symbol_index = random.randint(1,self.opts.n_classes)

            actual_index = self.opts.n_classes-symbol_index
            chirp_ideal = self.chirp_ideal_dict[actual_index]
            chirp_ideal_stft = self.chirp_ideal_stft_dict[actual_index]

            chirp_raw = add_noise(chirp_ideal,self.opts,random.choice(self.opts.snr_list))
            if self.opts.dechirp:
                chirp_raw = self.opts.demod.dechirp(chirp_raw).to(torch.cfloat)

            label_per = torch.tensor(actual_index, dtype=int)
            return (chirp_raw,chirp_ideal_stft),label_per

        except ValueError as e: 
            print(e) 
        except OSError as e: 
            print(e) 
 
def lora_loader(opts):  
    train_dataset = lora_dataset(opts, mode = 'train') 
    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,  drop_last=True, pin_memory=True) 

    test_dataset = lora_dataset(opts, mode = 'test')
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers, drop_last=True, pin_memory=True)

    return train_dloader,test_dloader
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulated Data Loaders

# Denoise Loader
class lora_denoise_dataset_raw(data.Dataset):
    def __init__(self, opts, f_gt_codewords, f_iq_pairs_gt, f_perturbed_packets):
        self.opts = opts 
        self.gt_codewords = dict()
        for file in f_gt_codewords:
            symbol_coded = loadmat(os.path.join(opts.data_dir,file))
            symbol_coded = symbol_coded['codeArray']
            self.gt_codewords[int(file.split('_')[1])] = symbol_coded.squeeze()

        packet = np.memmap(os.path.join(opts.data_dir,'packet_7_GT_7_.bin'),np.float32)
        packet = packet[::2] + packet[1::2]*1j
        preamble = packet[:int(opts.demod.d_num_samples*8)]
        sfd = packet[int(opts.demod.d_num_samples*10):int(opts.demod.d_num_samples*12.25)]
        
        preamble_stft = torch.stft(torch.tensor(preamble), n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=torch.hamming_window(opts.stft_window))
        preamble_stft = torch.squeeze(spec_to_network_input_complex(preamble_stft.unsqueeze(0), opts))
        sfd_stft = torch.stft(torch.tensor(sfd), n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=torch.hamming_window(opts.stft_window))
        sfd_stft = torch.squeeze(spec_to_network_input_complex(sfd_stft.unsqueeze(0), opts))

        preamble_stft = torch.cat((preamble_stft,sfd_stft),dim=-1)
        self.opts.preamble_stft = preamble_stft.to(self.opts.device)
        self.opts.preamble_conj = torch.conj(preamble_stft).to(self.opts.device)

        self.ideal_stft_dict = dict()
        for ind in range(1,opts.demod.d_num_symbols+1):
            symbol = torch.tensor(opts.mod.gen_symbol(ind), dtype=torch.cfloat)
            symbol = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=torch.hamming_window(opts.stft_window))
            symbol = torch.squeeze(spec_to_network_input(symbol.unsqueeze(0), opts))
            self.ideal_stft_dict[opts.demod.d_num_symbols-ind] = symbol

        self.perturbed_packets_files = f_perturbed_packets
        
    def __len__(self):
        return len(self.perturbed_packets_files)*self.opts.demod.d_approx_packet_symbol_len

    def __getitem__(self, idx):
        file_ind = idx // self.opts.demod.d_approx_packet_symbol_len
        symbol_ind = idx % self.opts.demod.d_approx_packet_symbol_len

        file = self.perturbed_packets_files[file_ind]
        packet_number = int(file.split('_')[1])

        packet = np.memmap(os.path.join(self.opts.data_dir,file),np.float32)
        packet = packet[::2] + packet[1::2]*1j

        preamble_recd = torch.tensor(packet[:int(self.opts.demod.d_num_samples*8)])
        sfd_recd = torch.tensor(packet[int(self.opts.demod.d_num_samples*10):int(self.opts.demod.d_num_samples*12.25)])
        
        symbol = torch.tensor(packet[int(self.opts.demod.d_num_samples*(12.25+symbol_ind)):int(self.opts.demod.d_num_samples*(13.25+symbol_ind)):])
        
        # symbol_gt_stft = self.gt_symbols[packet_number][symbol_ind]
        Y = self.gt_codewords[packet_number][symbol_ind]
        symbol_gt_stft = self.ideal_stft_dict[Y]
        Y = torch.tensor(Y, dtype=torch.long)
        symbol_ind = torch.tensor([symbol_ind], dtype=torch.float32)
        return (preamble_recd,sfd_recd,symbol,symbol_gt_stft,symbol_ind), Y
    
def lora_denoise_raw_loader(opts):
    gt_codewords, iq_pairs_gt, perturbed_packets_train, perturbed_packets_test = get_sim_files(opts)
    train_dataset = lora_denoise_dataset_raw(opts, gt_codewords, iq_pairs_gt, perturbed_packets_train)
    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)

    test_dataset = lora_denoise_dataset_raw(opts, gt_codewords, iq_pairs_gt, perturbed_packets_test)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    return train_dloader, test_dloader

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Real world Data Loaders

# Real Denoise Loader
class lora_real_dataset(data.Dataset):
    def __init__(self, opts, f_gt_codewords, f_perturbed_packets):
        self.opts = opts 
        self.gt_codewords = dict()

        for file in f_gt_codewords:
            symbol_coded = np.load(os.path.join(opts.gt_dir,file))
            self.gt_codewords[int(file.split('_')[1])] = symbol_coded
        
        self.ideal_stft_dict = dict()
        for ind in range(1,opts.demod.d_num_symbols+1):
            symbol = torch.tensor(opts.mod.gen_symbol(ind), dtype=torch.cfloat)
            symbol = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=torch.hamming_window(opts.stft_window))
            symbol = torch.squeeze(spec_to_network_input(symbol.unsqueeze(0), opts))
            self.ideal_stft_dict[opts.demod.d_num_symbols-ind] = symbol

        packet = np.memmap(os.path.join(opts.root_path,'checkpoints/packet_7_GT_7_.bin'),np.float32)
        packet = packet[::2] + packet[1::2]*1j
        preamble = packet[:int(opts.demod.d_num_samples*8)]
        sfd = packet[int(opts.demod.d_num_samples*10):int(opts.demod.d_num_samples*12.25)]
        
        preamble_stft = torch.stft(torch.tensor(preamble), n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=torch.hamming_window(opts.stft_window))
        preamble_stft = torch.squeeze(spec_to_network_input_complex(preamble_stft.unsqueeze(0), opts))
        sfd_stft = torch.stft(torch.tensor(sfd), n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant',window=torch.hamming_window(opts.stft_window))
        sfd_stft = torch.squeeze(spec_to_network_input_complex(sfd_stft.unsqueeze(0), opts))

        preamble_stft = torch.cat((preamble_stft,sfd_stft),dim=-1)
        self.opts.preamble_stft = preamble_stft.to(self.opts.device)
        self.opts.preamble_conj = torch.conj(preamble_stft).to(self.opts.device)

        self.perturbed_packets_files = f_perturbed_packets

        self.exclude_indices =  set()
        self.valid_indices = [i for i in range(opts.demod.d_approx_packet_symbol_len) if i not in self.exclude_indices]

    def __len__(self):
        return len(self.perturbed_packets_files) * len(self.valid_indices)

    def __getitem__(self, idx):        
        file_ind = idx // len(self.valid_indices)
        valid_idx = idx % len(self.valid_indices)
        symbol_ind = self.valid_indices[valid_idx]

        file = self.perturbed_packets_files[file_ind]
        packet_number = int(file.split('_')[1])

        packet = np.load(os.path.join(self.opts.data_dir,file))
        preamble_recd = torch.tensor(packet[:int(self.opts.demod.d_num_samples*8)])
        sfd_recd = torch.tensor(packet[int(self.opts.demod.d_num_samples*10):int(self.opts.demod.d_num_samples*12.25)])
        
        symbol = torch.tensor(packet[int(self.opts.demod.d_num_samples*(12.25+symbol_ind)):int(self.opts.demod.d_num_samples*(13.25+symbol_ind)):])
        
        Y = self.gt_codewords[packet_number][symbol_ind]
        symbol_gt_stft = self.ideal_stft_dict[Y]
        symbol_ind = torch.tensor([symbol_ind], dtype=torch.float32)

        return (preamble_recd,sfd_recd,symbol,symbol_gt_stft,symbol_ind), Y
    
def lora_real_dataset_loader(opts):
    gt_codewords, perturbed_packets_train, perturbed_packets_test = get_real_files(opts)
    train_dataset = lora_real_dataset(opts, gt_codewords, perturbed_packets_train)
    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    
    test_dataset = lora_real_dataset(opts, gt_codewords, perturbed_packets_test)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)

    return train_dloader, test_dloader


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Testing Conditional Data Loaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Simulated Data Loaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_perturbed_file(file, opts, gt_codewords_dict):
    packet_number = int(file.split('_')[1])
    packet = np.memmap(os.path.join(opts.data_dir, file), np.float32)
    packet = packet[::2] + packet[1::2] * 1j
    lora_demod = opts.demod.general_work_2(packet)
    gt_demod = gt_codewords_dict[packet_number]
    errors = np.sum(lora_demod != gt_demod)
    return file, errors

def load_sim_files_test_results(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results['gt_codewords'], results['gt_spec'], results['perturbed_packets_test']

def get_sim_files_test_adjusted(opts):  
    save_files_dest = os.path.join(opts.checkpoint_dir,'files.pkl')
    if os.path.exists(save_files_dest):
        return load_sim_files_test_results(save_files_dest)
    
    gt_codewords = []
    gt_spec = []
    perturbed_packets_test_all = []
    perturbed_packets_test = []
    total_errors = 0

    obj = os.scandir(opts.data_dir)

    params_data = loadmat(os.path.join(opts.data_dir,'params.mat'))
    valid_perturbations = list()
    for i,arr in enumerate(params_data['codeArray'].squeeze()):
        if i > opts.num_perturbations_train:
            valid_perturbations.append(i)

    for entry in obj:
        if entry.is_file() and entry.name != 'params.mat' and 'preamble' not in entry.name:
            parts = entry.name.split('_')
            packet_number = int(parts[1])
            entry_type = parts[2]

            if packet_number <= opts.num_packets:
                if entry_type == "codewords":
                    gt_codewords.append(entry.name)

                elif entry_type == 'GT':
                    gt_spec.append(entry.name)
                
                elif entry_type == 'RayleighAWGN':
                    perturbation_number = int(parts[3])
                    if perturbation_number in valid_perturbations:
                        perturbed_packets_test_all.append(entry.name)

    gt_codewords_dict = {}
    for file in gt_codewords:
        symbol_coded = loadmat(os.path.join(opts.data_dir, file))
        symbol_coded = symbol_coded['codeArray']
        gt_codewords_dict[int(file.split('_')[1])] = symbol_coded.squeeze()

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_perturbed_file, file, opts, gt_codewords_dict): file for file in perturbed_packets_test_all}
        for future in as_completed(future_to_file):
            result, errors = future.result()
            if errors > 0:
                perturbed_packets_test.append(result)
                total_errors += errors
    
    lora_phy_ber = total_errors * 100 / (len(perturbed_packets_test) * opts.demod.d_approx_packet_symbol_len)
    print("LoRa PHY BER:", lora_phy_ber)
    time.sleep(2)

    results = {
        'gt_codewords': gt_codewords,
        'gt_spec': gt_spec,
        'perturbed_packets_test': perturbed_packets_test,
    }

    with open(save_files_dest, 'wb') as f:
        pickle.dump(results, f)

    log_file_path = os.path.join(opts.checkpoint_dir, 'testing_log.csv')
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Testing', 'Accuracy'])
    
    with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['LoRaPHY', lora_phy_ber])

    return gt_codewords, gt_spec, perturbed_packets_test

def lora_denoise_raw_loader_test_adjusted(opts):
    gt_codewords, iq_pairs_gt, perturbed_packets_test = get_sim_files_test_adjusted(opts)

    test_dataset = lora_denoise_dataset_raw(opts, gt_codewords, iq_pairs_gt, perturbed_packets_test)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    return test_dloader

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Real Data Loaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def process_perturbed_file_real(file, opts, gt_codewords_dict):
    packet_number = int(file.split('_')[1])
    packet = np.load(os.path.join(opts.data_dir,file))
    lora_demod = opts.demod.general_work_2(packet)
    gt_demod = gt_codewords_dict[packet_number]
    errors = np.sum(lora_demod != gt_demod)
    return file, errors

def get_real_files_test_adjusted(opts):
    save_files_dest = os.path.join(opts.checkpoint_dir,'files.pkl')
    if os.path.exists(save_files_dest):
        gt_codewords,_,perturbed_packets_test = load_sim_files_test_results(save_files_dest)
        return gt_codewords, perturbed_packets_test
    
    gt_codewords = []
    perturbed_packets_test_all = []
    perturbed_packets_test = []
    total_errors = 0

    obj = os.scandir(opts.data_dir)

    for entry in obj:
        if entry.is_file():
            parts = entry.name.split('_')
            node_id = int(parts[2])
            if node_id in opts.test_nodes:
                perturbed_packets_test_all.append(entry.name)
    
    obj = os.scandir(opts.gt_dir)
    for entry in obj:
        if entry.is_file():
            parts = entry.name.split('_')
            entry_type = parts[2]

            if entry_type == "codewords":
                gt_codewords.append(entry.name)

    gt_codewords_dict = {}
    for file in gt_codewords:
        symbol_coded = np.load(os.path.join(opts.gt_dir,file))
        gt_codewords_dict[int(file.split('_')[1])] = symbol_coded

    with ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_perturbed_file_real, file, opts, gt_codewords_dict): file for file in perturbed_packets_test_all}
        for future in as_completed(future_to_file):
            result, errors = future.result()
            if errors > 0:
                perturbed_packets_test.append(result)
                total_errors += errors
    
    lora_phy_ber = total_errors * 100 / (len(perturbed_packets_test) * opts.demod.d_approx_packet_symbol_len)
    print("LoRa PHY BER:", lora_phy_ber)

    results = {
        'gt_codewords': gt_codewords,
        'gt_spec': None,
        'perturbed_packets_test': perturbed_packets_test,
    }

    with open(save_files_dest, 'wb') as f:
        pickle.dump(results, f)

    log_file_path = os.path.join(opts.checkpoint_dir, 'testing_log.csv')
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Testing', 'Accuracy'])
    
    with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['LoRaPHY', lora_phy_ber])
                
    return gt_codewords, perturbed_packets_test

def lora_real_dataset_loader_test(opts):
    gt_codewords, perturbed_packets_test = get_real_files_test_adjusted(opts)
    test_dataset = lora_real_dataset(opts, gt_codewords, perturbed_packets_test)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    return  test_dloader

