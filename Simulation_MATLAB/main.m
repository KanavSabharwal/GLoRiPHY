clc;
clear;
close all;

base_dir = '/path/to/save/directory'; % Define the directory to save the files

SF = param_configs(1);
rng(7);

start_index = 1; 
numPackets = 100; % Define the number of packets to generate
num_instances = 1000; % Define the number of instances to generate
maxDopplerShiftRange = [2 10];
numPathsRange = [1 10];
minDelay = 7;
maxDelay = 2500;
snrRange = [-25 0];
noiseModels = {'Rayleigh', 'Rician'}; % Noise model types
K_factors = [1 7]; % Range for K-factor in Rician model

for i = start_index:num_instances
    numPaths = randi(numPathsRange); % Generate a random number of paths
    params(i).numPaths = numPaths;
    delay = minDelay + (maxDelay - minDelay) * rand(1, numPaths - 1);
    delay = sort(delay * 1e-9); % Convert delays to seconds
    params(i).delay = [0, delay];
    params(i).averagePathGains = sort(-rand(1, numPaths) * 15, 'descend'); % Path gains
    params(i).dopplerShift = randi(maxDopplerShiftRange); % Doppler shift
    % params(i).SNR = fixedSNRs(randi(length(fixedSNRs))); % Randomly select SNR from fixed set
    params(i).SNR = randi(snrRange);
    % Assign noise model type and additional parameters for Rician
    params(i).noiseModel = noiseModels{randi(length(noiseModels))};
    if strcmp(params(i).noiseModel, 'Rician')
        params(i).KFactor = K_factors(randi(length(K_factors))); % Random K-factor for Rician model
        params(i).LOS_DopplerShift = randi([2, params(i).dopplerShift]); % LOS Path Doppler Shift for Rician
    end
end

% Save the parameters
filename = sprintf('%s/SF%d/params.mat', base_dir, SF);
Utils.parsave(filename, params);
disp(['File written: ', filename]);

codeArrays = cell(1, numPackets);
% Generate the packets
for i = start_index:numPackets
    codeArray = randi([0, 2^SF-1], 1, 18); % Generate a random set of 23 symbols, i.e. 8 bytes in SF8
    codeArrays{i} = codeArray;
end

parfor i = start_index:numPackets
    codeArray = codeArrays{i};
    
    % Save the codeArray
    codeArray_filename = sprintf('%s/SF%d/packet_%d_codewords_.mat',base_dir, SF, i);
    Utils.parsave(codeArray_filename, codeArray);
    
    % Generate the packet
    packet = Utils.gen_packet(codeArray); 
    filename = sprintf('%s/SF%d/packet_%d_GT_7',base_dir, SF, i);

    filename = [filename, '_.bin'];
    io_write_iq(filename, packet);

    disp(['File written: ', filename]);

    packet = packet.'; 

    for j = 1:num_instances
        temp = perturb_jgj(packet,params(j)); 
        filename = sprintf('%s/SF%d/packet_%d_RayleighAWGN_%d',base_dir, SF, i, j);
        temp = temp.';

        filename = [filename, '_.bin'];
        io_write_iq(filename, temp);

        disp(['File written: ', filename]);
    end
end