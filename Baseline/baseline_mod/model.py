import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .modules import maskCNNModel, classificationHybridModel


class NeLoRa(nn.Module):
    def __init__(
            self,
            opts
    ) -> None:
        super(NeLoRa, self).__init__()
        self.mask_model = maskCNNModel(opts)
        self.hybrid_model = classificationHybridModel(conv_dim_in=opts.y_image_channel,
                                                      conv_dim_out=opts.n_classes,
                                                      conv_dim_lstm=opts.conv_dim_lstm)

    def forward(self, symbol) -> Tuple[Tensor]:
        masked_symbol = self.mask_model(symbol)
        outputs = self.hybrid_model(masked_symbol)
    
        return masked_symbol,outputs
    
