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

import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .activation import Swish, GLU
from .modules import Transpose

# ~~~~~~~~~~~~~~~~~~~ UTIL LAYERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout

    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences

    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConformerConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            nn.LayerNorm(in_channels),
            Transpose(shape=(1, 2)),
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.sequential(inputs).transpose(1, 2)
 
# ~~~~~~~~~~~~~~~~~~~ ENCODERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DenseEncoder_SF7(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(DenseEncoder_SF7, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, 3, 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (3,1), (2,1),padding=(1,0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 8

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x

class DenseEncoder_SF8(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(DenseEncoder_SF8, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, 3, 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(channels, channels*2, (3,1), (2,1)),  #For 32
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, (3,1), (2,1),padding=(1,0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 8

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_mid(x)  #Include for 32
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x

class DenseEncoder_SF9(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(DenseEncoder_SF9, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, 3, 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(channels, channels*2, 3, 2, padding=(0, 1)),  #For 32
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, (3,1), (2,1),padding=(1,0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 4

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_mid(x)  #Include for 32
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x
    
class DenseEncoder_SF10(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(DenseEncoder_SF10, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, 3, 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(channels, channels*2, 3, 2, padding=(0, 1)),  #For 32
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, (3,3), (2,2),padding=(1,1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 2

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_mid(x)  #Include for 32
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x

class DenseEncoder_SF11(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(DenseEncoder_SF11, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, 3, 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_Extra = nn.Sequential(
            nn.Conv2d(channels, channels*2, (3,1), (2,1),padding=(1,0)),
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2)
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, 3, 2, padding=(0, 1)),  #For 32
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, (3,3), (2,2),padding=(1,1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 2

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_Extra(x)
        x = self.conv_mid(x)  #Include for 32
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x

class DenseEncoder_SF12(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(DenseEncoder_SF12, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, 3, 2),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_Extra = nn.Sequential(
            nn.Conv2d(channels, channels*2, 3, 2,padding=(1,0)),
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2)
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, 3, 2, padding=(0, 1)),  #For 32
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, (3,3), (2,2),padding=(1,1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_Extra(x)
        x = self.conv_mid(x)  #Include for 32
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(batch_size, channels*subsampled_time, subsampled_freq)
        return x
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PREAMBLE ENCODERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Preamble_preEncoder(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(Preamble_preEncoder, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, kernel_size=(3, 3), stride=(2, 2)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels*2, kernel_size=(3, 4), stride=(2, 3)), 
            nn.InstanceNorm2d(channels*2, affine=True),
            nn.PReLU(channels*2),
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=(3, 4), stride=(2, 3), padding=(1,0)), 
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x
    
class ChannelEncoder_SF8(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(ChannelEncoder_SF8, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, kernel_size=(1, 2), stride=(1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 8

    def forward(self, x):
        x = self.conv_1(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x

class ChannelEncoder_SF9(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(ChannelEncoder_SF9, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, kernel_size=(1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 4

    def forward(self, x):
        x = self.conv_1(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x

class ChannelEncoder_SF10(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(ChannelEncoder_SF10, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 4

    def forward(self, x):
        x = self.conv_1(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x
    
class ChannelEncoder_SF11(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(ChannelEncoder_SF11, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2),padding=(0, 1)), 
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 2

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x
    
class ChannelEncoder_SF12(nn.Module):
    def __init__(self, in_channel, channels=32):
        super(ChannelEncoder_SF12, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(2, 2),padding=(1, 1)), 
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.mod_fact = 2

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, channels*time_extra, subsampled_freq*self.mod_fact)
        return x
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CHANNEL DECODERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ChannelDecoder_SF8(nn.Module):
    def __init__(self):
        super(ChannelDecoder_SF8, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 2)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 1, 3, 2)
        x = self.conv_1(x)
        return x
    
class ChannelDecoder_SF9(nn.Module):
    def __init__(self):
        super(ChannelDecoder_SF9, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=(1, 2), stride=(1, 1), padding=(0, 1)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2),
        )
        self.mod_fact = 2

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 1, 3, 2)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,self.mod_fact,time_extra).contiguous()
        x = x.view(batch_size, channels, subsampled_freq*self.mod_fact,time_extra)
        x = self.conv_1(x)
        return x

class ChannelDecoder_SF10(nn.Module):
    def __init__(self):
        super(ChannelDecoder_SF10, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(1, 2, (3,2), (2,1), padding=(1, 0), output_padding=(1, 0)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2),
        )
        self.mod_fact = 2
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 1, 3, 2)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,self.mod_fact,time_extra).contiguous()
        x = x.view(batch_size, channels, subsampled_freq*self.mod_fact,time_extra)
        x = self.conv_1(x)
        return x
    
class ChannelDecoder_SF11(nn.Module):
    def __init__(self):
        super(ChannelDecoder_SF11, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(1, 2, (3,3), (2,2), padding=(0, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=(2, 1), stride=(1, 1), padding=(2, 0)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2),
        )
        self.mod_fact = 4
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.permute(0, 1, 3, 2)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 2, 4, 3).contiguous()
        x = x.view(batch_size, channels, subsampled_freq*self.mod_fact, time_extra)
        return x
    
class ChannelDecoder_SF12(nn.Module):
    def __init__(self):
        super(ChannelDecoder_SF12, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(1, 2, (4,3), (2,2), padding=(0, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2),
        )
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(2, 2, (2,3), (1,2), padding=(0, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(2, affine=True),
            nn.PReLU(2),
        )
        self.mod_fact = 4
        
    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.permute(0, 1, 3, 2)
        batch_size,channels,subsampled_freq,subsampled_time = x.size()
        time_extra = subsampled_time // self.mod_fact
        x = x.view(batch_size,channels,subsampled_freq,time_extra,self.mod_fact).permute(0, 1, 2, 4, 3).contiguous()
        x = x.view(batch_size, channels, subsampled_freq*self.mod_fact, time_extra)
        return x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DECODERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DenseDecoder_SF7(nn.Module):
    def __init__(self, out_channel = 2, channels=32):
        super(DenseDecoder_SF7, self).__init__()
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, (3,1), (2,1), padding=(1, 0), output_padding=(0, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(channels, 2, (3, 3), (2, 2), padding=(0, 0),output_padding=(1,0)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )
        
        self.time_extra = 2
        self.mod_fact = 8

    def forward(self, input):
        batch_size,channels,encoded_dim = input.size()
        output = input.view(batch_size,channels//self.time_extra,self.time_extra,encoded_dim//self.mod_fact,self.mod_fact)
        output = output.permute(0,1,3,2,4).contiguous()
        batch_size,channels,subsampled_freq,_,_ = output.size()
        output = output.view(batch_size,channels,subsampled_freq,self.mod_fact*self.time_extra)
        output = self.deconv_1(output)
        output = self.deconv_2(output)
        return output

class DenseDecoder_SF8(nn.Module):
    def __init__(self, out_channel = 2, channels=32):
        super(DenseDecoder_SF8, self).__init__()
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 2, (3,1), (2,1), padding=(1, 0)),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.PReLU(channels * 2),
        )
        self.deconv_mid = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, (3,1), (2,1), padding=(0, 0), output_padding=(0, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(channels, 2, (3, 3), (2, 2), padding=(0, 0),output_padding=(1,0)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )
        
        self.time_extra = 2
        self.mod_fact = 8

    def forward(self, input):
        batch_size,channels,encoded_dim = input.size()
        output = input.view(batch_size,channels//self.time_extra,self.time_extra,encoded_dim//self.mod_fact,self.mod_fact)
        output = output.permute(0,1,3,2,4).contiguous()
        batch_size,channels,subsampled_freq,_,_ = output.size()
        output = output.view(batch_size,channels,subsampled_freq,self.mod_fact*self.time_extra)
        output = self.deconv_1(output)
        output = self.deconv_mid(output)
        output = self.deconv_2(output)
        return output
    
class DenseDecoder_SF9(nn.Module):
    def __init__(self, out_channel = 2, channels=32):
        super(DenseDecoder_SF9, self).__init__()
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 2, (3,1), (2,1), padding=(1, 0)),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.PReLU(channels * 2),
        )
        self.deconv_mid = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 3, 2, padding=(0, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(channels, 2, (3, 3), (2, 2), padding=(0, 0),output_padding=(1,0)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )
        self.time_extra = 2
        self.mod_fact = 4

    def forward(self, input):
        batch_size,channels,encoded_dim = input.size()
        output = input.view(batch_size,channels//self.time_extra,self.time_extra,encoded_dim//self.mod_fact,self.mod_fact)
        output = output.permute(0,1,3,2,4).contiguous()
        batch_size,channels,subsampled_freq,_,_ = output.size()
        output = output.view(batch_size,channels,subsampled_freq,self.mod_fact*self.time_extra)
        output = self.deconv_1(output)
        output = self.deconv_mid(output)
        output = self.deconv_2(output)
        return output
    
class DenseDecoder_SF10(nn.Module):
    def __init__(self, out_channel = 2, channels=32):
        super(DenseDecoder_SF10, self).__init__()
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 2, (3,3), (2,2), padding=(1, 1)),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.PReLU(channels * 2),
        )
        self.deconv_mid = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, (3,3), (2,2), padding=(0, 0), output_padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(channels, 2, (3, 3), (2, 2), padding=(0, 0),output_padding=(1,0)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )
        self.time_extra = 2
        self.mod_fact = 2

    def forward(self, input):
        batch_size,channels,encoded_dim = input.size()
        output = input.view(batch_size,channels//self.time_extra,self.time_extra,encoded_dim//self.mod_fact,self.mod_fact)
        output = output.permute(0,1,3,2,4).contiguous()
        batch_size,channels,subsampled_freq,_,_ = output.size()
        output = output.view(batch_size,channels,subsampled_freq,self.mod_fact*self.time_extra)
        output = self.deconv_1(output)
        output = self.deconv_mid(output)
        output = self.deconv_2(output)
        return output
    
class DenseDecoder_SF11(nn.Module):
    def __init__(self, out_channel = 2, channels=32):
        super(DenseDecoder_SF11, self).__init__()
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 2, (3,3), (2,2), padding=(1, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.PReLU(channels * 2),
        )
        self.deconv_mid = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels* 2, (3,3), (2,2), padding=(0, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(channels* 2, affine=True),
            nn.PReLU(channels* 2),
        )
        self.deconv_Extra = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, (3,1), (2,1), padding=(1, 0), output_padding=(0, 0)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(channels, 2, (3, 3), (2, 2), padding=(0, 0),output_padding=(1,0)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )
        self.time_extra = 2
        self.mod_fact = 2

    def forward(self, input):
        batch_size,channels,encoded_dim = input.size()
        output = input.view(batch_size,channels//self.time_extra,self.time_extra,encoded_dim//self.mod_fact,self.mod_fact)
        output = output.permute(0,1,3,2,4).contiguous()
        batch_size,channels,subsampled_freq,_,_ = output.size()
        output = output.view(batch_size,channels,subsampled_freq,self.mod_fact*self.time_extra)
        output = self.deconv_1(output)
        output = self.deconv_mid(output)
        output = self.deconv_Extra(output)
        output = self.deconv_2(output)
        return output

class DenseDecoder_SF12(nn.Module):
    def __init__(self, out_channel = 2, channels=32):
        super(DenseDecoder_SF12, self).__init__()
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels * 2, (3,3), (2,2), padding=(1, 1), output_padding=(0, 1)),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.PReLU(channels * 2),
        )
        self.deconv_mid = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels* 2, (3,3), (2,2), padding=(0, 1), output_padding=(1, 0)),
            nn.InstanceNorm2d(channels* 2, affine=True),
            nn.PReLU(channels* 2),
        )
        self.deconv_Extra = nn.Sequential(
            nn.ConvTranspose2d(channels*2, channels, (3,3), (2,2), padding=(1, 0), output_padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels),
        )
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(channels, 2, (3, 3), (2, 2), padding=(0, 0),output_padding=(1,0)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
        )

    def forward(self, input):
        batch_size,channels,encoded_dim = input.size()
        output = input.view(batch_size,channels//2,2,encoded_dim)
        output = output.permute(0,1,3,2).contiguous()
        output = self.deconv_1(output)
        output = self.deconv_mid(output)
        output = self.deconv_Extra(output)
        output = self.deconv_2(output)
        return output