# Copyright (c) 2021, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .encoder import *
from .convolution import *
from utils import reconstruct_from_stft_batched

class ConformerGen_Core(nn.Module):
    def __init__(
            self,
            opts,
            conformer_for: str = 'symbol',
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            no_encoder = False
    ) -> None:
        super(ConformerGen_Core, self).__init__()
        if no_encoder:
            self.encoder = ConformerNoEncoder(
                conformer_for=conformer_for,
                encoder_dim=encoder_dim,
                num_layers=num_encoder_layers,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                input_dropout_p=input_dropout_p,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual
            )
        else:
            self.encoder = ConformerEncoder(
                conformer_for=conformer_for,
                encoder_dim=encoder_dim,
                num_layers=num_encoder_layers,
                num_attention_heads=num_attention_heads,
                feed_forward_expansion_factor=feed_forward_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                input_dropout_p=input_dropout_p,
                feed_forward_dropout_p=feed_forward_dropout_p,
                attention_dropout_p=attention_dropout_p,
                conv_dropout_p=conv_dropout_p,
                conv_kernel_size=conv_kernel_size,
                half_step_residual=half_step_residual
            )
        
        self.opts = opts
        if conformer_for == 'sf_7':
            self.decoder = DenseDecoder_SF7()
        elif conformer_for == 'sf_8':
            self.decoder = DenseDecoder_SF8()
        elif conformer_for == 'sf_9':
            self.decoder = DenseDecoder_SF9()
        elif conformer_for == 'sf_10':
            self.decoder = DenseDecoder_SF10()
        elif conformer_for == 'sf_11':
            self.decoder = DenseDecoder_SF11()
        elif conformer_for == 'sf_12':
            self.decoder = DenseDecoder_SF12()

        self.process_output = nn.Sequential(
                                            nn.Linear(1, opts.n_classes),
                                            nn.ReLU(),
                                            nn.Dropout(0.1),
                                            nn.Linear(opts.n_classes, opts.n_classes))

    def train(self, mode: bool = True) -> None:
        """ Set the training mode of the model """
        self.training_mode = mode
        super(ConformerGen_Core, self).train(mode)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, symbol) -> Tuple[Tensor]:
        processed_masked_symbol = self.encoder(symbol)
        processed_masked_symbol = self.decoder(processed_masked_symbol)

        pred_ind = reconstruct_from_stft_batched(processed_masked_symbol,self.opts) 
        outputs = self.process_output(pred_ind.unsqueeze(1))      
        return processed_masked_symbol,outputs,pred_ind
    
class ConformerGen2(nn.Module):
    def __init__(self,
                opts,
                conformer_for: str = 'symbol',
                encoder_dim: int = 512,
                num_encoder_layers: int = 17,
                num_attention_heads: int = 8,
                feed_forward_expansion_factor: int = 4,
                conv_expansion_factor: int = 2,
                input_dropout_p: float = 0.1,
                feed_forward_dropout_p: float = 0.1,
                attention_dropout_p: float = 0.1,
                conv_dropout_p: float = 0.1,
                conv_kernel_size: int = 31,
                half_step_residual: bool = True,
                require_symbol_index: bool = False
        ) -> None:
        super(ConformerGen2, self).__init__()
        self.preamble_encode_pre = Preamble_preEncoder(in_channel=2)
        self.preamble_encoder = ConformerEncoder(conformer_for=conformer_for+'_channel',
                                                encoder_dim = encoder_dim, 
                                                num_layers=num_encoder_layers,
                                                num_attention_heads=num_attention_heads,
                                                feed_forward_expansion_factor=feed_forward_expansion_factor,
                                                conv_expansion_factor=conv_expansion_factor,
                                                input_dropout_p=input_dropout_p,
                                                feed_forward_dropout_p=feed_forward_dropout_p,
                                                attention_dropout_p=attention_dropout_p,
                                                conv_dropout_p=conv_dropout_p,
                                                conv_kernel_size=conv_kernel_size,
                                                half_step_residual=half_step_residual,
                                                in_channel = 32)
        if conformer_for == 'sf_8':
            self.channel_decoder = ChannelDecoder_SF8()
        elif conformer_for == 'sf_9':
            self.channel_decoder = ChannelDecoder_SF9()
        elif conformer_for == 'sf_10':
            self.channel_decoder = ChannelDecoder_SF10()
        elif conformer_for == 'sf_11':
            self.channel_decoder = ChannelDecoder_SF11()
        elif conformer_for == 'sf_12':
            self.channel_decoder = ChannelDecoder_SF12()

        self.conformer_mask = ConformerEncoder(conformer_for=conformer_for,
                                                encoder_dim = encoder_dim, 
                                                num_layers = num_encoder_layers,
                                                num_attention_heads=num_attention_heads,
                                                feed_forward_expansion_factor=feed_forward_expansion_factor,
                                                conv_expansion_factor=conv_expansion_factor,
                                                input_dropout_p=input_dropout_p,
                                                feed_forward_dropout_p=feed_forward_dropout_p,
                                                attention_dropout_p=attention_dropout_p,
                                                conv_dropout_p=conv_dropout_p,
                                                conv_kernel_size=conv_kernel_size,
                                                half_step_residual=half_step_residual,
                                                in_channel = 4)

        self.conformer_symbol = ConformerGen_Core(opts,
                                conformer_for=conformer_for,
                                encoder_dim=encoder_dim, 
                                num_attention_heads= num_attention_heads,
                                num_encoder_layers=2*num_encoder_layers,
                                input_dropout_p =input_dropout_p,
                                feed_forward_dropout_p = feed_forward_dropout_p,
                                attention_dropout_p = attention_dropout_p,
                                conv_dropout_p = conv_dropout_p,
                                no_encoder = True)
        
        self.opts = opts

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, *args) -> Tuple[Tensor]:
        preamble = args[0]
        symbol = args[1]
        preamble = self.preamble_encode_pre(preamble)
        channel_est = self.preamble_encoder(preamble)
        channel_est = self.channel_decoder(channel_est)

        channel_symbol = torch.cat([channel_est,symbol],dim=1)
        if len(args) > 2 and self.opts.use_symbol_index:
            symbol_index = args[2]
            processed_symbol = self.conformer_mask(channel_symbol,symbol_index)
        else:
            processed_symbol = self.conformer_mask(channel_symbol)

        generated_symbol,outputs,pred_ind = self.conformer_symbol(processed_symbol)      
        
        return generated_symbol,outputs,pred_ind