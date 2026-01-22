'''
# --------------------------------------------
# Group Inertial Poser Network
# --------------------------------------------
# Group Inertial Poser: Multi-Person Pose and Global Translation from Sparse Inertial Sensors and Ultra-Wideband Ranging (ICCV 2025)
# https://github.com/eth-siplab/GroupInertialPoser
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from mamba_ssm import Mamba
    

class MambaModel(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=256, n_layers=4, dropout=0.2, prenorm=False, lr=0.001, bidirectional=False):
        super().__init__()
        
        self.model = Mamba(
            d_model=d_model, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.encoder = nn.Linear(d_input, d_model)
        self.decoder = nn.Linear(d_model, d_output)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = pad_sequence(x).permute(1, 0, 2) # 12 200, 72
        length = [_.shape[0] for _ in x]
        
        x = self.encoder(x)

        # x = self.dropout(nn.functional.relu(x))

        
        x = self.model(x)
        
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        x = x.permute(1, 0, 2)
        x_list = [x[:l, i].clone() for i, l in enumerate(length)]

        return x_list
        