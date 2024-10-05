function par = param_configs(p_id)

    % LoRa PHY transmitting parameters
    LORA_SF = 8;            % LoRa spreading factor
    LORA_BW = 125e3;        % LoRa bandwidth
    
    % Receiving device parameters
    RX_Sampl_Rate = LORA_BW*8;  % recerver's sampling rate
    
    switch(p_id)
        case 1
            par = LORA_SF;
        case 2
            par = LORA_BW;
        case 3
            par = RX_Sampl_Rate;
        otherwise
    end
end