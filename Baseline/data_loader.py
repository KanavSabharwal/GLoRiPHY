import os 
import random 
import numpy as np
from scipy.io import loadmat
import pickle
import torch 
from torch.utils.data import DataLoader 
from torch.utils import data 
from utils_baseline import spec_to_network_input, add_noise

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
            
            ideal_stft = torch.stft(chirp, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, pad_mode='constant')
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

        self.ideal_stft_dict = dict()
        for ind in range(1,opts.demod.d_num_symbols+1):
            symbol = torch.tensor(opts.mod.gen_symbol(ind), dtype=torch.cfloat)
            symbol = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant')
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

        symbol = torch.tensor(packet[int(self.opts.demod.d_num_samples*(12.25+symbol_ind)):int(self.opts.demod.d_num_samples*(13.25+symbol_ind)):])
        
        Y = self.gt_codewords[packet_number][symbol_ind]
        symbol_gt_stft = self.ideal_stft_dict[Y]
        Y = torch.tensor(Y, dtype=torch.long)
        return (symbol,symbol_gt_stft), Y
    
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
            symbol = torch.stft(symbol, n_fft=opts.stft_nfft, hop_length=opts.hop_length, win_length=opts.stft_window, return_complex=True, pad_mode='constant')
            symbol = torch.squeeze(spec_to_network_input(symbol.unsqueeze(0), opts))
            self.ideal_stft_dict[opts.demod.d_num_symbols-ind] = symbol

        self.perturbed_packets_files = f_perturbed_packets

    def __len__(self):
        return len(self.perturbed_packets_files) * self.opts.demod.d_approx_packet_symbol_len

    def __getitem__(self, idx):        
        file_ind = idx // self.opts.demod.d_approx_packet_symbol_len
        symbol_ind = idx % self.opts.demod.d_approx_packet_symbol_len

        file = self.perturbed_packets_files[file_ind]
        packet_number = int(file.split('_')[1])

        packet = np.load(os.path.join(self.opts.data_dir,file))        
        symbol = torch.tensor(packet[int(self.opts.demod.d_num_samples*(12.25+symbol_ind)):int(self.opts.demod.d_num_samples*(13.25+symbol_ind)):])
        
        Y = self.gt_codewords[packet_number][symbol_ind]
        symbol_gt_stft = self.ideal_stft_dict[Y]
        return (symbol,symbol_gt_stft), Y
    
def lora_real_dataset_loader(opts):
    gt_codewords, perturbed_packets_train, perturbed_packets_test = get_real_files(opts)
    train_dataset = lora_real_dataset(opts, gt_codewords, perturbed_packets_train)
    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)

    test_dataset = lora_real_dataset(opts, gt_codewords, perturbed_packets_test)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)

    return train_dloader, test_dloader

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Testing Conditional Data Loaders
def process_perturbed_file(file, opts, gt_codewords_dict):
    packet_number = int(file.split('_')[1])
    packet = np.memmap(os.path.join(opts.data_dir, file), np.float32)
    packet = packet[::2] + packet[1::2] * 1j
    lora_demod = opts.demod.general_work_2(packet)
    gt_demod = gt_codewords_dict[packet_number]
    errors = np.sum(lora_demod != gt_demod)
    return file if errors > 0 else None, errors

def load_sim_files_test_results(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results['gt_codewords'], results['gt_spec'], results['perturbed_packets_test']

def get_sim_files_test_adjusted(opts):  
    save_files_dest = os.path.join(opts.checkpoint_dir,'files.pkl')
    if os.path.exists(save_files_dest):
        return load_sim_files_test_results(save_files_dest)
    else:
        raise Exception("Please test GLoRiPHY first")

def get_real_files_test_adjusted(opts):  
    save_files_dest = os.path.join(opts.checkpoint_dir,'files.pkl')
    if os.path.exists(save_files_dest):
        gt_codewords,_,perturbed_packets_test = load_sim_files_test_results(save_files_dest)
        return gt_codewords, perturbed_packets_test
    else:
        raise Exception("Please test GLoRiPHY first")

def lora_denoise_raw_loader_test_adjusted(opts):
    gt_codewords, iq_pairs_gt, perturbed_packets_test = get_sim_files_test_adjusted(opts)
    test_dataset = lora_denoise_dataset_raw(opts, gt_codewords, iq_pairs_gt, perturbed_packets_test)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    return test_dloader

def lora_real_dataset_loader_test(opts):
    gt_codewords, perturbed_packets_test = get_real_files_test_adjusted(opts)
    test_dataset = lora_real_dataset(opts, gt_codewords, perturbed_packets_test)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers, drop_last=True, pin_memory=True)
    return  test_dloader

