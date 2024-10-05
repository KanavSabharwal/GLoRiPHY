classdef Utils < handle 
    properties (Constant) % Constant properties are inherently static
        SF = param_configs(1)
        BW = param_configs(2)
        Fs = param_configs(3)
        
        num_samp = Utils.Fs * 2^Utils.SF / Utils.BW;
        num_symb = 2^Utils.SF;
        T = (0:Utils.num_samp-1) / Utils.Fs;   

        stft_nfft = Utils.num_samp;
        stft_window = Utils.num_symb/2;
        hop_length = Utils.stft_window/2;
        noverlap = Utils.stft_window - Utils.hop_length;
        window = hamming(Utils.stft_window);
        trim_size = floor(Utils.num_symb / 2);
    end

    methods (Static = true)
        % Noise Generation Functions:  
        function rx_signal = add_awgn_noise(tx_signal, snr)
            phase = 2*pi*rand() - pi;
            tx_signal = tx_signal * (cos(phase) + 1i*sin(phase));
            
            AWGNChannel = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)',...
                                            'SNR', snr);
            rx_signal = AWGNChannel(tx_signal);

            release(AWGNChannel);
        end

        function rxSignal = add_rayleigh_noise(txSignal,params)
            pathDelays = params.delay;
            averagePathGains = params.averagePathGains;
            maxDopplerShift = params.dopplerShift;

            rayleighChannel = comm.RayleighChannel(...
                'SampleRate', Utils.Fs, ...
                'PathDelays', pathDelays, ...
                'AveragePathGains', averagePathGains, ...
                'NormalizePathGains', true, ...
                'MaximumDopplerShift', maxDopplerShift);
            
            rxSignal = rayleighChannel(txSignal);
            reset(rayleighChannel);
        end

        function rxSignal = add_rician_noise(txSignal,params)
            pathDelays = params.delay;
            averagePathGains = params.averagePathGains;
            maxDopplerShift = params.dopplerShift;
            k_factor = params.KFactor;
            directDopplerShift = params.LOS_DopplerShift;

            ricianchan = comm.RicianChannel( ...
                            'SampleRate', Utils.Fs, ...
                            'PathDelays', pathDelays, ...
                            'AveragePathGains',averagePathGains, ...
                            'KFactor',k_factor, ...
                            'DirectPathDopplerShift',directDopplerShift, ...
                            'MaximumDopplerShift',maxDopplerShift);
            
            rxSignal = ricianchan(txSignal);
            reset(ricianchan);
        end

        % Packet Generation Functions:
        function symb = gen_symbol(code_word,down)
               % time vector a chirp
            
            % I/Q traces
            f0 = -Utils.BW/2; % start freq
            f1 = Utils.BW/2;  % end freq
            chirpI = chirp(Utils.T, f0, Utils.T(end), f1, 'linear', 90);
            chirpQ = chirp(Utils.T, f0, Utils.T(end), f1, 'linear', 0);
            baseline = complex(chirpI, chirpQ);
            
            if nargin >= 2 && down
                baseline = conj(baseline);
            end
            
            baseline = repmat(baseline,1,2);
            clear chirpI chirpQ
            
            % Shift for encoding
            offset = round((2^Utils.SF - code_word) / 2^Utils.SF * Utils.num_samp);
            symb = baseline(offset+(1:Utils.num_samp));
        end
        
        function real_sig = gen_packet(codeArray)
            invert = 0;
            codeChirp = Utils.gen_symbol(0,invert);
            syncChirp = Utils.gen_symbol(0,~invert);
            
            L = length(codeChirp); 
            sig_length = L * (12.25+length(codeArray));
            real_sig = zeros(1, sig_length);

            pos = 1; % Start position for real_sig
            real_sig(pos:pos+L*8-1) = repmat(codeChirp, 1, 8);
            pos = pos + L * 8;

            real_sig(pos:pos+L-1) = Utils.gen_symbol(2^Utils.SF-24, invert);
            pos = pos + L;
            real_sig(pos:pos+L-1) = Utils.gen_symbol(2^Utils.SF-32, invert);
            pos = pos + L;

            real_sig(pos:pos+L*2-1) = repmat(syncChirp, 1, 2);
            pos = pos + L * 2;
            real_sig(pos:pos+L/4-1) = syncChirp(1:end/4);
            pos = pos + L/4;

            for i = codeArray(1:end)
                tmp_symb = Utils.gen_symbol(2^Utils.SF-i, invert);
                real_sig(pos:pos+L-1) = tmp_symb;
                pos = pos + L;
            end
            
        end

        function parsave(fname, codeArray)
            save(fname, 'codeArray');    
        end
    end
end