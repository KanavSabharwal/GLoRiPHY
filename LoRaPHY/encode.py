import numpy as np
from decode import rotl, parity, header_checksum, data_checksum

HAMMING_P1_BITMASK = 0x0D  
HAMMING_P2_BITMASK = 0x0B
HAMMING_P3_BITMASK = 0x07
HAMMING_P4_BITMASK = 0x0F 
HAMMING_P5_BITMASK = 0x0E

class EncodeImpl:
    def __init__(self, spreading_factor, header = True, cr = 1, crc = True):
        # Assertions
        assert 5 < spreading_factor < 13
        assert 0 < cr < 5
        if spreading_factor == 6:
            assert not header

        # Initializations from the provided code
        self.d_sf = spreading_factor
        self.d_header = header
        self.d_cr = cr
        self.d_crc = crc
        if spreading_factor == 12:
            self.d_ldr = True
        else:
            self.d_ldr = False

        self.d_whitening_sequence = [0xff, 0xfe, 0xfc, 0xf8, 0xf0, 0xe1, 0xc2, 0x85, 0x0b, 0x17, 0x2f, 0x5e, 0xbc, 0x78, 0xf1, 0xe3, 0xc6, 0x8d, 0x1a, 0x34, 0x68, 0xd0, 0xa0, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x11, 0x23, 0x47, 0x8e, 0x1c, 0x38, 0x71, 0xe2, 0xc4, 0x89, 0x12, 0x25, 0x4b, 0x97, 0x2e, 0x5c, 0xb8, 0x70, 0xe0, 0xc0, 0x81, 0x03, 0x06, 0x0c, 0x19, 0x32, 0x64, 0xc9, 0x92, 0x24, 0x49, 0x93, 0x26, 0x4d, 0x9b, 0x37, 0x6e, 0xdc, 0xb9, 0x72, 0xe4, 0xc8, 0x90, 0x20, 0x41, 0x82, 0x05, 0x0a, 0x15, 0x2b, 0x56, 0xad, 0x5b, 0xb6, 0x6d, 0xda, 0xb5, 0x6b, 0xd6, 0xac, 0x59, 0xb2, 0x65, 0xcb, 0x96, 0x2c, 0x58, 0xb0, 0x61, 0xc3, 0x87, 0x0f, 0x1f, 0x3e, 0x7d, 0xfb, 0xf6, 0xed, 0xdb, 0xb7, 0x6f, 0xde, 0xbd, 0x7a, 0xf5, 0xeb, 0xd7, 0xae, 0x5d, 0xba, 0x74, 0xe8, 0xd1, 0xa2, 0x44, 0x88, 0x10, 0x21, 0x43, 0x86, 0x0d, 0x1b, 0x36, 0x6c, 0xd8, 0xb1, 0x63, 0xc7, 0x8f, 0x1e, 0x3c, 0x79, 0xf3, 0xe7, 0xce, 0x9c, 0x39, 0x73, 0xe6, 0xcc, 0x98, 0x31, 0x62, 0xc5, 0x8b, 0x16, 0x2d, 0x5a, 0xb4, 0x69, 0xd2, 0xa4, 0x48, 0x91, 0x22, 0x45, 0x8a, 0x14, 0x29, 0x52, 0xa5, 0x4a, 0x95, 0x2a, 0x54, 0xa9, 0x53, 0xa7, 0x4e, 0x9d, 0x3b, 0x77, 0xee, 0xdd, 0xbb, 0x76, 0xec, 0xd9, 0xb3, 0x67, 0xcf, 0x9e, 0x3d, 0x7b, 0xf7, 0xef, 0xdf, 0xbf, 0x7e, 0xfd, 0xfa, 0xf4, 0xe9, 0xd3, 0xa6, 0x4c, 0x99, 0x33, 0x66, 0xcd, 0x9a, 0x35, 0x6a, 0xd4, 0xa8, 0x51, 0xa3, 0x46, 0x8c, 0x18, 0x30, 0x60, 0xc1, 0x83, 0x07, 0x0e, 0x1d, 0x3a, 0x75, 0xea, 0xd5, 0xaa, 0x55, 0xab, 0x57, 0xaf, 0x5f, 0xbe, 0x7c, 0xf9, 0xf2, 0xe5, 0xca, 0x94, 0x28, 0x50, 0xa1, 0x42, 0x84, 0x09, 0x13, 0x27, 0x4f, 0x9f, 0x3f, 0x7f]
        self.d_interleaver_size = spreading_factor
        self.d_fft_size = (1 << spreading_factor)

    def gen_header(self,nibbles, payload_len):
        cr_crc = (self.d_cr << 1) | self.d_crc  
        cks = header_checksum(payload_len, cr_crc)  
        nibbles.append(payload_len >> 4)
        nibbles.append(payload_len & 0xF)
        nibbles.append(cr_crc)
        nibbles.append(cks >> 4)
        nibbles.append(cks & 0xF)
    
    def calc_sym_num(self, payload_len):
        return 8 + max((4 + self.d_cr) * int(np.ceil((2.0 * payload_len - self.d_sf + 7 + 4 * self.d_crc - 5 * (not self.d_header)) / (self.d_sf - 2 * self.d_ldr))), 0)

    def whiten(self, bytes, length):
        for i in range(min(length, 255)): 
            bytes[i] = ((bytes[i] & 0xFF) ^ self.d_whitening_sequence[i]) & 0xFF 

    def hamming_encode(self, nibbles, codewords):
        for i in range(len(nibbles)):
            p1 = parity(nibbles[i], HAMMING_P1_BITMASK)  # Assuming parity function & bitmask constants are defined
            p2 = parity(nibbles[i], HAMMING_P2_BITMASK)
            p3 = parity(nibbles[i], HAMMING_P3_BITMASK)
            p4 = parity(nibbles[i], HAMMING_P4_BITMASK)
            p5 = parity(nibbles[i], HAMMING_P5_BITMASK)

            cr_now = 4 if i < self.d_sf - 2 else self.d_cr  # Assuming d_sf and d_cr are defined elsewhere

            if cr_now == 1:
                codewords.append((p4 << 4) | (nibbles[i] & 0xF))
            elif cr_now == 2:
                codewords.append((p5 << 5) | (p3 << 4) | (nibbles[i] & 0xF))
            elif cr_now == 3:
                codewords.append((p2 << 6) | (p5 << 5) | (p3 << 4) | (nibbles[i] & 0xF))
            elif cr_now == 4:
                codewords.append((p1 << 7) | (p2 << 6) | (p5 << 5) | (p3 << 4) | (nibbles[i] & 0xF))
            else:
                raise ValueError("Invalid Code Rate -- this state should never occur.")
            
    def interleave(self,codewords, symbols):
        bits_per_word = 8
        ppm = self.d_sf - 2 
        start_idx = 0

        while start_idx + ppm - 1 < len(codewords):
            bits_per_word = 8 if start_idx == 0 else (self.d_cr + 4)
            ppm = (self.d_sf - 2) if start_idx == 0 else (self.d_sf - 2 * self.d_ldr)
            block = [0] * bits_per_word

            for i in range(ppm):
                word = codewords[start_idx + i]
                for j, x in zip([1 << (bits_per_word - 1 - n) for n in range(bits_per_word)], range(bits_per_word - 1, -1, -1)):
                    block[x] |= ((word & j) != 0) << i

            for i in range(bits_per_word):
                # Rotate each element to the right by i bits
                block[i] = rotl(block[i], 2 * ppm - i, ppm)  # Assuming rotl function is defined

            symbols.extend(block)
            start_idx += ppm

    def from_gray(self,symbols):
        for i in range(len(symbols)):
            symbols[i] = symbols[i] ^ (symbols[i] >> 16)
            symbols[i] = symbols[i] ^ (symbols[i] >> 8)
            symbols[i] = symbols[i] ^ (symbols[i] >> 4)
            symbols[i] = symbols[i] ^ (symbols[i] >> 2)
            symbols[i] = symbols[i] ^ (symbols[i] >> 1)
            if i < 8 or self.d_ldr:  # Assuming d_ldr and d_sf are defined elsewhere
                symbols[i] = (symbols[i] * 4 + 1) % (1 << self.d_sf)
            else:
                symbols[i] = (symbols[i] + 1) % (1 << self.d_sf)

    def encode(self, in_bytes):
        bytes_in = in_bytes.copy()
        pkt_len = len(bytes_in)
        nibbles = []
        codewords = []
        payload_nibbles = []
        symbols = []

        if self.d_crc: 
            checksum = data_checksum(bytes_in,len(bytes_in))  
            bytes_in += [checksum & 0xFF, (checksum >> 8) & 0xFF]

        sym_num = self.calc_sym_num(pkt_len)  
        nibble_num = self.d_sf - 2 + (sym_num - 8) // (self.d_cr + 4) * (self.d_sf - 2 * self.d_ldr)
        redundant_num = int(np.ceil((nibble_num - 2 * pkt_len) / 2))

        for i in range(redundant_num):
            bytes_in.append(0)

        self.whiten(bytes_in, pkt_len)  

        # Split bytes into separate data nibbles
        for i in range(nibble_num):
            if i % 2 == 0:
                payload_nibbles.append(bytes_in[i // 2] & 0xF)
            else:
                payload_nibbles.append(bytes_in[i // 2] >> 4)

        if self.d_header: 
            self.gen_header(nibbles, pkt_len)  


        nibbles.extend(payload_nibbles)
        self.hamming_encode(nibbles, codewords) 

        self.interleave(codewords, symbols) 

        self.from_gray(symbols) 

        return symbols
