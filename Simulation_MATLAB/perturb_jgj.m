function packet = perturb_jgj(packet,param)
    SNR = param.SNR;
    if strcmp(param.noiseModel, 'Rayleigh')
       packet = Utils.add_rayleigh_noise(packet,param);
    elseif strcmp(param.noiseModel, 'Rician')
       packet = Utils.add_rician_noise(packet,param);
   else
       error('Unknown noise type');
   end
    packet = Utils.add_awgn_noise(packet, SNR);
end
