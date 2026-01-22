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
from models.s4.s4 import S4Block as S4
import torch
from torch.nn.utils.rnn import pad_sequence

ver = tuple(map(int, torch.__version__.split('.')[:2]))
if ver == (1, 11):
    dropout_fn = nn.Dropout
elif ver >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        bidirectional=False,
        lr=0.001,
    ):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """

        x = pad_sequence(x).permute(1, 0, 2)
        length = [_.shape[0] for _ in x]

        input_x = x.clone()

        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        x = x.permute(1, 0, 2)
        x_list = [x[:l, i].clone() for i, l in enumerate(length)]

        return x_list