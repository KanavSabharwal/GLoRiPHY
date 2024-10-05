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

from .feed_forward import FeedForwardModule
from .attention import MultiHeadedSelfAttentionModule
from .convolution import *
from .modules import ResidualConnectionModule


class ConformerBlock(nn.Module):
    def __init__(
            self,
            encoder_dim: int = 512,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
    ):
        super(ConformerBlock, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.sequential = nn.Sequential(
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            ResidualConnectionModule(
                module=MultiHeadedSelfAttentionModule(
                    d_model=encoder_dim,
                    num_heads=num_attention_heads,
                    dropout_p=attention_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=ConformerConvModule(
                    in_channels=encoder_dim,
                    kernel_size=conv_kernel_size,
                    expansion_factor=conv_expansion_factor,
                    dropout_p=conv_dropout_p,
                ),
            ),
            ResidualConnectionModule(
                module=FeedForwardModule(
                    encoder_dim=encoder_dim,
                    expansion_factor=feed_forward_expansion_factor,
                    dropout_p=feed_forward_dropout_p,
                ),
                module_factor=self.feed_forward_residual_factor,
            ),
            nn.LayerNorm(encoder_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs)


class ConformerEncoder(nn.Module):
    def __init__(
            self,
            conformer_for: str = 'symbol',
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            num_features = 32,
            in_channel = 2
    ):
        super(ConformerEncoder, self).__init__()
        mod_fact = encoder_dim // num_features

        if conformer_for == 'sf_7':
            self.conv_subsample = DenseEncoder_SF7(in_channel=in_channel)
        elif conformer_for == 'sf_8':
            self.conv_subsample = DenseEncoder_SF8(in_channel=in_channel)
        elif conformer_for == 'sf_8_channel':
            self.conv_subsample = ChannelEncoder_SF8(in_channel=32)
        elif conformer_for == 'sf_9':
            self.conv_subsample = DenseEncoder_SF9(in_channel=in_channel)
        elif conformer_for == 'sf_9_channel':
            self.conv_subsample = ChannelEncoder_SF9(in_channel=32)
        elif conformer_for == 'sf_10':
            self.conv_subsample = DenseEncoder_SF10(in_channel=in_channel)
        elif conformer_for == 'sf_10_channel':
            self.conv_subsample = ChannelEncoder_SF10(in_channel=32)
        elif conformer_for == 'sf_11':
            self.conv_subsample = DenseEncoder_SF11(in_channel=in_channel)
        elif conformer_for == 'sf_11_channel':
            self.conv_subsample = ChannelEncoder_SF11(in_channel=32)
        elif conformer_for == 'sf_12':
            self.conv_subsample = DenseEncoder_SF12(in_channel=in_channel)
        elif conformer_for == 'sf_12_channel':
            self.conv_subsample = ChannelEncoder_SF12(in_channel=32)
        
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

        self.input_dropout = nn.Dropout(p=input_dropout_p)

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """ Update dropout probability of encoder """
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor) -> Tuple[Tensor]:
        outputs = self.conv_subsample(inputs)
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs = layer(outputs)
        
        return outputs
    
class ConformerNoEncoder(nn.Module):
    def __init__(
            self,
            conformer_for: str = 'symbol',
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True
    ):
        super(ConformerNoEncoder, self).__init__()        
        self.layers = nn.ModuleList([ConformerBlock(
            encoder_dim=encoder_dim,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_layers)])

        self.input_dropout = nn.Dropout(p=input_dropout_p)

    def forward(self, inputs: Tensor) -> Tuple[Tensor]:
        outputs = self.input_dropout(inputs)

        for layer in self.layers:
            outputs = layer(outputs)
        
        return outputs