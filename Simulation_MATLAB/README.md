# Generating Simulated Data

This section provides instructions on how to generate the simulated dataset used by GLoRiPHY. The provided MATLAB scripts generate the dataset simulated by Rayleigh/Rician + AWGN channels. The steps to generate the dataset are as follows:

## Requirements
- MATLAB R2020a or later
- Communications Toolbox
- Parallel Computing Toolbox (optional)

## Generating the Simulated Dataset
### 1.Specify LoRa Parameters in ```param_configs.m```

You can specify the Spreading Factor (SF), and the Bandwidth (BW) in the ```param_configs.m``` file in the ```LORA_SF``` and  ```LORA_BW``` variables, respectively. The default values are set to SF = 8 and BW = 125 kHz.

### 2. Specify the Dataset Path in ```main.m```
You are required to specify the path where the dataset will be saved in the ```main.m``` file, by modifying the ```base_dir``` variable.

### 3. Specify the Pertubation Parameters in ```main.m```
You can modify the perturbation configurations in the ```main.m``` file. The perturbation configurations include:
- ```numPackets```: Define the number of distinct LoRa packets, each with a random payload to generate.
- ```numInstances```: Define the number of unique channel instances that perturb each packet.
- ```maxDopplerShiftRange```: Define the range of maximum Doppler shifts (in Hz) for the Rayleigh and Rician channels.
- ```numPathsRange```: Define the range of the number of multi-path components for the Rayleigh and Rician channels.
- ```minDelay```: Define the minimum delay (in nanoseconds) for the reception of the first multi-path component.
- ```maxDelay```: Define the maximum delay (in nanoseconds) for the reception of the last multi-path component.
- ```K_factors```: Define the range of K-factors for the Rician noise model.
- ```snrRange```: Define the range of SNR values (in dB) for the AWGN channel.
- ```noiseModels```: Define the noise models for the convolved noise. The available options are 'Rayleigh' and 'Rician'.

### 4 Run the MATLAB Script ```main.m``` to Generate the Dataset

## Dataset Structure
The generated dataset is saved in the specified path in the following structure:
- ```params.mat```: The parameters for each perturbation channel are stored in this file.
- ```packet_i_codewords_.mat```: The ground truth codewords, i.e. symbol indices, for the i-th packet.
- ```packet_i_GT_7_.bin```: The ground truth transmitted signal for the i-th packet.
- ```packet_i_RayleighAWGN_j_.bin```: The received signal for the i-th packet after passing through the j-th perturbation channel.