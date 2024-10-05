import numpy as np
from scipy.signal import chirp

NUM_PREAMBLE_CHIRPS = 8
class ModImpl:
    def __init__(self, spreading_factor, sync_word = 0x34, Fs = 1e6, bw = 125e3):
        # Assertions
        # Sync Word 12 or 34, seems 34
        assert 5 < spreading_factor < 13

        self.d_sf = spreading_factor
        self.d_sync_word = sync_word
        self.d_bw = bw
        self.d_Fs = Fs
        self.d_p = int(Fs/bw)
        self.d_num_symbols = (1 << self.d_sf)
        self.d_num_samples = self.d_p * self.d_num_symbols
 
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

    def modulate(self, in_data):
        symbols_in = in_data.copy()
        pkt_len = len(symbols_in)
        iq_out = []

        # Preamble
        iq_out = np.tile(self.d_upchirp, NUM_PREAMBLE_CHIRPS)
         # Sync Word 0
        iq_out = np.append(iq_out, self.gen_symbol(self.d_num_symbols - (8 * ((self.d_sync_word & 0xF0) >> 4))))
        # Sync Word 1
        iq_out = np.append(iq_out, self.gen_symbol(self.d_num_symbols - (8 * (self.d_sync_word & 0x0F))))
        # SFD Downchirps
        iq_out = np.append(iq_out, np.tile(self.d_downchirp, 2))
        iq_out = np.append(iq_out, self.d_downchirp[0:self.d_num_samples//4])
        # Payload
        for i in range(pkt_len):
            iq_out = np.append(iq_out, self.gen_symbol(self.d_num_symbols - symbols_in[i]))
        
        return iq_out