import numpy as np
from scipy.fftpack import fft
from scipy.signal import chirp
from mod import ModImpl

def fpmod(x, n):
    return (x % n + n) % n

class DemodImpl:
    def __init__(self, spreading_factor, header = True, bw =125e3, payload_len = 8, cr = 1, crc = True , fft_factor = 16, fs_bw_ratio = 8):
        # Assertions
        assert 5 < spreading_factor < 13
        if spreading_factor == 6:
            assert not header
        assert fft_factor > 0
        assert int(fs_bw_ratio) == fs_bw_ratio

        # Initializations from the provided code
        self.d_sf = spreading_factor
        self.d_bw = bw
        self.d_Fs = bw * fs_bw_ratio
        self.d_header = header
        self.d_payload_len = payload_len
        self.d_cr = cr
        self.d_crc = crc
        if spreading_factor == 12:
            self.d_ldr = True
        else:
            self.d_ldr = False
        self.d_fft_size_factor = fft_factor
        self.d_p = int(fs_bw_ratio)

        self.d_approx_packet_symbol_len = 8 + max((4 + self.d_cr) * int(np.ceil((2.0 * self.d_payload_len - self.d_sf + 7 + 4 * self.d_crc - 5 * (not self.d_header)) / (self.d_sf - 2 * self.d_ldr))), 0)

        if not header:
            self.d_packet_symbol_len = 8 + max((4 + self.d_cr) * int(np.ceil((2.0 * self.d_payload_len - self.d_sf + 7 + 4 * self.d_crc - 5 * (not self.d_header)) / (self.d_sf - 2 * self.d_ldr))), 0)

        # Calculations
        self.d_num_symbols = (1 << self.d_sf)
        self.d_num_samples = self.d_p * self.d_num_symbols
        self.d_bin_size = self.d_fft_size_factor * self.d_num_symbols
        self.d_fft_size = self.d_fft_size_factor * self.d_num_samples
        # FFT initialization can be done using numpy when needed
        
        mod = ModImpl(self.d_sf, Fs=self.d_Fs, bw=self.d_bw)
        packet = mod.modulate([0])
        preamble = packet[:int(self.d_num_samples*8)]
        sfd = packet[int(self.d_num_samples*10):int(self.d_num_samples*12.25)]
        self.ref_signal = np.concatenate((preamble,sfd))

        self.d_downchirp = self.gen_symbol(down=True)
        self.d_upchirp = self.gen_symbol()

    def gen_symbol(self, code_word=0, down=False):
        T = np.arange(self.d_num_samples)/self.d_Fs
        
        f0 = -self.d_bw/2  # start frequency
        f1 = self.d_bw/2   # end frequency
        chirpI = chirp(T, f0, T[-1], f1, method='linear', phi=90)
        chirpQ = chirp(T, f0, T[-1], f1, method='linear', phi=0)
        baseline = chirpI + 1j * chirpQ
        
        if down:
            baseline = np.conjugate(baseline)
            
        baseline = np.tile(baseline, 2)
        
        # Shift for encoding
        offset = round((2**self.d_sf - code_word) / 2**self.d_sf * self.d_num_samples)
        symb = baseline[offset:offset + self.d_num_samples]
        
        return symb
        
    def dechirp(self, in_data, is_sfd = False):
        if is_sfd:
            return in_data * self.d_upchirp[:self.d_num_samples]
        else:
            return in_data * self.d_downchirp[:self.d_num_samples]
        
    def get_fft(self, in_data):
        fft_input = np.zeros(self.d_fft_size, dtype=np.complex64)
        fft_input[:self.d_num_samples] = in_data
        return fft(fft_input)
    
    def search_fft_peak(self, fft_result, phase_search = True, k = 16, get_fft_res = False):
        max_idx = 0

        if not phase_search:
            buffer1 = np.abs(fft_result)
            buffer2 = buffer1[:self.d_bin_size] + buffer1[-self.d_bin_size:]
            # Take argmax of returned FFT (similar to MFSK demod)
            max_idx = np.argmax(buffer2)
            
        else:
            mx_pk = -1
            cut1 = fft_result[:self.d_bin_size]
            cut2 = fft_result[-self.d_bin_size:]
            for i in np.linspace(0, 1, num=k, endpoint=False):
                tmp = cut1 + cut2 * np.exp(1j * 2 * np.pi * i)
                current_max = np.max(np.abs(tmp))
                if current_max > mx_pk:
                    mx_pk = current_max
                    out_rst = tmp
            max_idx = np.argmax(np.abs(out_rst))

        if get_fft_res:
            return max_idx, out_rst
        return max_idx

    def demod_symbol(self, symbol, is_sfd = False, phase_search = False):
        dechirped = self.dechirp(symbol, is_sfd)
        fft_result = self.get_fft(dechirped)
        max_idx = self.search_fft_peak(fft_result,phase_search)
        return max_idx
    
    def dynamic_compensation(self,d_symbols):
        compensated_symbols = []
        
        modulus = 4.0
        bin_drift = 0
        bin_comp = 0
        v = 0
        v_last = 1
                
        for v in d_symbols:
            bin_drift = fpmod(v - v_last, modulus)

            # compensate bin drift
            if bin_drift < modulus / 2:
                bin_comp -= bin_drift
            else:
                bin_comp -= (bin_drift - modulus)
            
            bin_comp = 0 if not self.d_ldr else bin_comp
            
            v_last = v
            compensated_value = np.round(fpmod(v + bin_comp, self.d_num_symbols)).astype(int)
            compensated_symbols.append(fpmod(compensated_value, self.d_num_symbols))

        return compensated_symbols
    
    def process_symbols(self, symbols_v):
        symbols_in = []

        for i in range(len(symbols_v)):
            v = symbols_v[i]
            
            if i < 8 or self.d_ldr:
                v //= 4
            else:
                v = fpmod(v - 1, 1 << self.d_sf)
            
            symbols_in.append(v)

        return symbols_in
    
    def find_offsets(self, packet):
        freq_res = self.d_Fs / self.d_fft_size

        symb_id = 5
        start_ind = int(symb_id*self.d_num_samples)
        end_ind = int((symb_id+1)*self.d_num_samples)
        up_chirp = packet[start_ind:end_ind]
        up_freq = self.demod_symbol(up_chirp,phase_search= False) * freq_res
                       
        symb_id = 11
        start_ind = int(symb_id*self.d_num_samples)
        end_ind = int((symb_id+1)*self.d_num_samples)
        downchirp = packet[start_ind:end_ind]
        down_freq = self.demod_symbol(downchirp,phase_search= False,is_sfd=True) * freq_res

        if up_freq > self.d_bw/2:
            up_freq -= self.d_bw  

        if down_freq > self.d_bw/2:
            down_freq -= self.d_bw  

        frequency_offset = (up_freq + down_freq) / 2
        time_offset = round((frequency_offset-up_freq)/ self.d_bw * self.d_num_samples)
        if abs(time_offset)>100 or abs(frequency_offset)>10000:
            time_offset = 0
            frequency_offset = 0
        return time_offset, frequency_offset
    
    def correct_frequency_offset(self, packet, frequency_offset):
        t = np.arange(len(packet)) / self.d_Fs
        correction = np.exp(-1j * 2 * np.pi * frequency_offset * t)
        corrected_signal = packet * correction
        return corrected_signal.astype(np.complex64)
    
    def correct_packet(self, packet):
        time_offset, frequency_offset = self.find_offsets(packet)
        if time_offset < 0:
            packet = np.pad(packet, (abs(time_offset), 0), mode='constant')
        else:
            packet = packet[time_offset:]
            packet = np.pad(packet, (0, time_offset), mode='constant')

        corrected_signal = self.correct_frequency_offset(packet, frequency_offset)
        return corrected_signal
    
    def general_work_2(self, full):        
        full = self.correct_packet(full)
        payload = full[int(12.25*self.d_num_samples):]

        symbols = list()
        for ind in range(0,self.d_approx_packet_symbol_len):
            symbol = payload[ind*self.d_num_samples : (ind+1)*self.d_num_samples]
            max_idx = self.demod_symbol(symbol)
            bin_idx = fpmod((max_idx) / self.d_fft_size_factor, self.d_num_symbols)
            symbols.append(bin_idx)

        symbols = np.round(symbols).astype(int)
        symbols = self.dynamic_compensation(symbols)
        # symbols = self.process_symbols(symbols.copy())  
        return symbols