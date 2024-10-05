import numpy as np
HAMMING_P1_BITMASK = 0x2E  # 0b00101110
HAMMING_P2_BITMASK = 0x4B  # 0b01001011
HAMMING_P3_BITMASK = 0x17  # 0b00010111
HAMMING_P4_BITMASK = 0xFF  # 0b11111111
HAMMING_D1_BITMASK = 0x08  # 0b00001000
HAMMING_D2_BITMASK = 0x04  # 0b00000100
HAMMING_D3_BITMASK = 0x01  # 0b00000001
HAMMING_D4_BITMASK = 0x02  # 0b00000010

def fpmod(x, n):
    return (x % n + n) % n

def rotl(bits, count=1, size=8):
    len_mask = (1 << size) - 1
    count %= size        # Limit bit rotate count to size
    bits &= len_mask     # Limit given bits to size
    
    return ((bits << count) & len_mask) | (bits >> (size - count))

def parity(c, bitmask):
    parity_val = 0
    shiftme = c & bitmask
    
    for _ in range(8):
        if shiftme & 0x1:
            parity_val += 1
        shiftme >>= 1
    
    return parity_val % 2

def header_checksum(length, cr_crc):
    a0 = (length >> 4) & 0x1
    a1 = (length >> 5) & 0x1
    a2 = (length >> 6) & 0x1
    a3 = (length >> 7) & 0x1

    b0 = (length >> 0) & 0x1
    b1 = (length >> 1) & 0x1
    b2 = (length >> 2) & 0x1
    b3 = (length >> 3) & 0x1

    c0 = (cr_crc >> 0) & 0x1
    c1 = (cr_crc >> 1) & 0x1
    c2 = (cr_crc >> 2) & 0x1
    c3 = (cr_crc >> 3) & 0x1

    res = (a0 ^ a1 ^ a2 ^ a3) << 4
    res |= (a3 ^ b1 ^ b2 ^ b3 ^ c0) << 3
    res |= (a2 ^ b0 ^ b3 ^ c1 ^ c3) << 2
    res |= (a1 ^ b0 ^ b2 ^ c0 ^ c1 ^ c2) << 1
    res |= a0 ^ b1 ^ c0 ^ c1 ^ c2 ^ c3

    return res

def data_checksum(data, length):
    crc = 0
    for j in range(length - 2):
        new_byte = data[j]
        for i in range(8):
            if ((crc & 0x8000) >> 8) ^ (new_byte & 0x80):
                crc = (crc << 1) ^ 0x1021
            else:
                crc = crc << 1
            new_byte <<= 1

    # XOR the obtained CRC with the last 2 data bytes
    x1 = data[length-1] if length >= 1 else 0
    x2 = (data[length-2] << 8) if length >= 2 else 0
    crc = crc ^ x1 ^ x2
    return crc & 0xFFFF 

class DecodeImpl:
    def __init__(self, spreading_factor, header = True, payload_len = 8, cr = 1, crc = True, correct_payload = False):
        # Assertions
        assert 5 < spreading_factor < 13
        assert 0 < cr < 5
        if spreading_factor == 6:
            assert not header

        # Initializations from the provided code
        self.d_sf = spreading_factor
        self.d_header = header
        self.d_payload_len = payload_len
        self.d_cr = cr
        self.d_crc = crc
        if spreading_factor == 12:
            self.d_ldr = True
        else:
            self.d_ldr = False
        self.d_correct_payload = correct_payload

        self.d_whitening_sequence = [0xff, 0xfe, 0xfc, 0xf8, 0xf0, 0xe1, 0xc2, 0x85, 0x0b, 0x17, 0x2f, 0x5e, 0xbc, 0x78, 0xf1, 0xe3, 0xc6, 0x8d, 0x1a, 0x34, 0x68, 0xd0, 0xa0, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x11, 0x23, 0x47, 0x8e, 0x1c, 0x38, 0x71, 0xe2, 0xc4, 0x89, 0x12, 0x25, 0x4b, 0x97, 0x2e, 0x5c, 0xb8, 0x70, 0xe0, 0xc0, 0x81, 0x03, 0x06, 0x0c, 0x19, 0x32, 0x64, 0xc9, 0x92, 0x24, 0x49, 0x93, 0x26, 0x4d, 0x9b, 0x37, 0x6e, 0xdc, 0xb9, 0x72, 0xe4, 0xc8, 0x90, 0x20, 0x41, 0x82, 0x05, 0x0a, 0x15, 0x2b, 0x56, 0xad, 0x5b, 0xb6, 0x6d, 0xda, 0xb5, 0x6b, 0xd6, 0xac, 0x59, 0xb2, 0x65, 0xcb, 0x96, 0x2c, 0x58, 0xb0, 0x61, 0xc3, 0x87, 0x0f, 0x1f, 0x3e, 0x7d, 0xfb, 0xf6, 0xed, 0xdb, 0xb7, 0x6f, 0xde, 0xbd, 0x7a, 0xf5, 0xeb, 0xd7, 0xae, 0x5d, 0xba, 0x74, 0xe8, 0xd1, 0xa2, 0x44, 0x88, 0x10, 0x21, 0x43, 0x86, 0x0d, 0x1b, 0x36, 0x6c, 0xd8, 0xb1, 0x63, 0xc7, 0x8f, 0x1e, 0x3c, 0x79, 0xf3, 0xe7, 0xce, 0x9c, 0x39, 0x73, 0xe6, 0xcc, 0x98, 0x31, 0x62, 0xc5, 0x8b, 0x16, 0x2d, 0x5a, 0xb4, 0x69, 0xd2, 0xa4, 0x48, 0x91, 0x22, 0x45, 0x8a, 0x14, 0x29, 0x52, 0xa5, 0x4a, 0x95, 0x2a, 0x54, 0xa9, 0x53, 0xa7, 0x4e, 0x9d, 0x3b, 0x77, 0xee, 0xdd, 0xbb, 0x76, 0xec, 0xd9, 0xb3, 0x67, 0xcf, 0x9e, 0x3d, 0x7b, 0xf7, 0xef, 0xdf, 0xbf, 0x7e, 0xfd, 0xfa, 0xf4, 0xe9, 0xd3, 0xa6, 0x4c, 0x99, 0x33, 0x66, 0xcd, 0x9a, 0x35, 0x6a, 0xd4, 0xa8, 0x51, 0xa3, 0x46, 0x8c, 0x18, 0x30, 0x60, 0xc1, 0x83, 0x07, 0x0e, 0x1d, 0x3a, 0x75, 0xea, 0xd5, 0xaa, 0x55, 0xab, 0x57, 0xaf, 0x5f, 0xbe, 0x7c, 0xf9, 0xf2, 0xe5, 0xca, 0x94, 0x28, 0x50, 0xa1, 0x42, 0x84, 0x09, 0x13, 0x27, 0x4f, 0x9f, 0x3f, 0x7f]
        self.d_interleaver_size = spreading_factor
        self.d_fft_size = (1 << spreading_factor)
    
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
    
    def to_gray(self,symbols):
        for i in range(len(symbols)):
            symbols[i] = (symbols[i] >> 1) ^ symbols[i]
        return symbols
    
    def deinterleave(self,symbols, ppm, rdd):
        bits_per_word = rdd + 4
        codewords = []

        for start_idx in range(0, len(symbols), bits_per_word):
            block = [0] * ppm
            for i in range(bits_per_word):
                word = rotl(symbols[start_idx + i], i, ppm)
                
                j = 1 << (ppm - 1)
                x = ppm - 1
                
                while j:
                    block[x] |= ((word & j) != 0) << i
                    j >>= 1
                    x -= 1
            codewords.extend(block)
        return codewords

    def hamming_decode(self,codewords, rdd):
        bytes_out = []

        for i in range(len(codewords)):
            if (rdd > 2 or i < self.d_sf - 2) and self.d_correct_payload:  # Hamming(8,4) or Hamming(7,4)
                p1 = parity(codewords[i], HAMMING_P1_BITMASK)
                p2 = parity(codewords[i], HAMMING_P2_BITMASK)
                p3 = parity(codewords[i], HAMMING_P3_BITMASK)

                switch_case = (p3 << 2) | (p2 << 1) | p1
                if switch_case == 3:
                    codewords[i] ^= HAMMING_D1_BITMASK
                elif switch_case == 5:
                    codewords[i] ^= HAMMING_D2_BITMASK
                elif switch_case == 6:
                    codewords[i] ^= HAMMING_D3_BITMASK
                elif switch_case == 7:
                    codewords[i] ^= HAMMING_D4_BITMASK
            
            bytes_out.append(codewords[i] & 0x0F)

        return bytes_out
    
    def whiten(self,codewords):
        offset = 3 if self.d_header else 0
        crc_offset = 2 if self.d_crc else 0
        whitening_sequence_length = len(self.d_whitening_sequence)
        
        for i in range(min(whitening_sequence_length, len(codewords) - offset - crc_offset)):
            codewords[i + offset] ^= self.d_whitening_sequence[i]

        return codewords
        
    def general_work(self, symbols):
        symbols = self.process_symbols(symbols.copy())    
        symbols = self.to_gray(symbols)
        if self.d_header:
            header_codewords = self.deinterleave(symbols[:8], self.d_sf-2, 4)
            header_nibbles = self.hamming_decode(header_codewords, 4)

            payload_len = (header_nibbles[0] << 4) | header_nibbles[1]
            crc = header_nibbles[2] & 1
            cr = header_nibbles[2] >> 1
            checksum = (header_nibbles[3] << 4) | header_nibbles[4]
            is_header_valid = True
            if checksum != header_checksum(self.d_payload_len, header_nibbles[2] & 0xF):
                is_header_valid = False
            else:
                if payload_len!=self.d_payload_len:
                    print("Payload length mismatch: ", payload_len, " vs ", self.d_payload_len)
                if crc!=self.d_crc:
                    print("CRC mismatch")
                if cr!=self.d_cr:
                    print("CR mismatch")
            

        payload_codewords = self.deinterleave(symbols[8:], (self.d_sf-2) if self.d_ldr else self.d_sf, self.d_cr)
        codewords = header_codewords + payload_codewords
        if self.d_header:
            codewords.insert(5, 0)
        
        nibbles = self.hamming_decode(codewords, self.d_cr)
        min_len = self.d_payload_len * 2 + self.d_header * 6 + self.d_crc * 4

        combined_bytes = []
        for i in range(0, min_len, 2):
            if self.d_header and i < 6:
                combined_bytes.append((nibbles[i] << 4) | nibbles[i+1])
            else:
                combined_bytes.append((nibbles[i+1] << 4) | nibbles[i])

        whitened = self.whiten(combined_bytes)
        if self.d_crc:
            offset = 3 if self.d_header else 0
            checksum = whitened[self.d_payload_len + offset] | (whitened[self.d_payload_len + offset + 1] << 8)
            is_payload_valid = True if checksum == data_checksum(whitened[offset:],self.d_payload_len) else False

        return is_header_valid,is_payload_valid,whitened
        
