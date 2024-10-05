import numpy as np
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