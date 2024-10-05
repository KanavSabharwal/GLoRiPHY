import os
import shutil
import multiprocessing
import numpy as np
from demod import DemodImpl
import config


def correct_offset_sim(opts,filename):
    packet = np.memmap(os.path.join(opts.data_dir,filename),np.float32)
    packet = packet[::2] + packet[1::2]*1j
    corrected_packet = opts.demod.correct_packet(packet)

    processed_data_real = np.real(corrected_packet).astype(np.float32)
    processed_data_imag = np.imag(corrected_packet).astype(np.float32)
    processed_data = np.stack((processed_data_real, processed_data_imag), axis=-1).flatten()
    
    # # Save the processed data to a new binary file
    save_path = os.path.join(opts.corrected_dir, os.path.basename(filename))
    processed_data.tofile(save_path)

def correct_offset_real(opts,filename):
    packet  = np.load(os.path.join(opts.data_dir,filename))
    corrected_packet = opts.demod.correct_packet(packet)
    save_path = os.path.join(opts.corrected_dir, os.path.basename(filename))
    np.save(save_path, corrected_packet)

def worker_sim(args):
    opts, filename = args
    correct_offset_sim(opts, filename)

def worker_real(args):
    opts, filename = args
    correct_offset_real(opts, filename)

def parallel_process(opts):
    num_cores = multiprocessing.cpu_count()
    workers = max(1, int(num_cores * 0.9))

    tasks = None
    if opts.real_data:
        tasks = [(opts, filename.path) for filename in os.scandir(opts.data_dir) if filename.is_file()]

    elif opts.sim_data:
        tasks = [(opts, filename.path) for filename in  os.scandir(opts.data_dir) if 'RayleighAWGN' in filename.name]
    
    print('Offset correction started to test NeLoRa. This may take some time...')
    print(f'Processing {len(tasks)} tasks with {workers} workers')
    with multiprocessing.Pool(workers) as pool:
        if opts.real_data:
            pool.map(worker_real, tasks)
        elif opts.sim_data:
            pool.map(worker_sim, tasks)

def copy_data(opts):
    obj = os.scandir(opts.data_dir)

    for entry in obj:
        if entry.is_file():
            if entry.name != 'params.mat':
                parts = entry.name.split('_')
                if len(parts) > 2:  # Ensure there are enough parts in the filename
                    packet_number = int(parts[1])
                    entry_type = parts[2]

                    if packet_number <= opts.num_packets and entry_type in ["codewords", "GT"]:
                        shutil.copy(os.path.join(opts.data_dir, entry.name), opts.corrected_dir)

    # Copy params.mat specifically if needed
    params_path = os.path.join(opts.data_dir, 'params.mat')
    if os.path.exists(params_path):
        shutil.copy(params_path, opts.corrected_dir)

if __name__ == "__main__":
    parser = config.create_parser()
    opts = parser.parse_args()

    if not (opts.real_data or opts.sim_data):
        raise ValueError('Please specify the type of data to process')

    opts.demod = DemodImpl(opts.sf,fs_bw_ratio=opts.fs_bw_ratio,fft_factor=16)
    opts.corrected_dir = opts.data_dir + '_corrected'
    
    if not os.path.exists(opts.corrected_dir):
        os.makedirs(opts.corrected_dir)
        if opts.sim_data:
            copy_data(opts)
    else:
        raise ValueError('Directory already exists')

    parallel_process(opts)
    
    