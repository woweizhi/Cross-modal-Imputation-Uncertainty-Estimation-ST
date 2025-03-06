import collections
from typing import Callable, Iterable, List, Literal, Optional

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList

from scvi.nn._base_components import FCLayers

####### add cross attention module into base MultiEncoder in gimVI #############
class MultiEncoderCrossAttention(nn.Module):
    """MultiEncoder."""

    def __init__(
            self,
            n_heads: int,
            n_input_list: List[int],
            n_output: int,
            n_hidden: int = 128,
            n_layers_individual: int = 1,
            n_layers_shared: int = 2,
            n_cat_list: Iterable[int] = None,
            dropout_rate: float = 0.1,
            return_dist: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.encoders = ModuleList(
            [
                FCLayers(
                    n_in=n_input_list[i],
                    n_out=n_hidden,
                    n_cat_list=n_cat_list,
                    n_layers=n_layers_individual,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    use_batch_norm=True,
                    **kwargs,
                )
                for i in range(n_heads)
            ]
        )

        self.attention = nn.MultiheadAttention(n_hidden, num_heads=4)
        self.norm = nn.BatchNorm1d(n_hidden)
        #self.norm = nn.LayerNorm(n_hidden)

        self.encoder_shared = FCLayers(
            n_in=n_hidden,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers_shared,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

    def forward(self, x: List[torch.Tensor], head_id: int, *cat_list: int):
        """Forward pass."""
        # input x should be a list of tensors, contains both the scRNA and ST

        enc_list = []
        for mode, in_tensor in enumerate(x):
            enc = self.encoders[mode](in_tensor, *cat_list)
            enc_list.append(enc)

        q = enc_list[head_id]

        if head_id == 0:
            q, _ = self.attention(enc_list[0], enc_list[1], enc_list[1])
            q = q + enc_list[0]
            q = self.norm(q)
        if head_id == 1:
            q, _ = self.attention(enc_list[1], enc_list[0], enc_list[0])
            q = q + enc_list[1]
            q = self.norm(q)

        q_attent = self.encoder_shared(q, *cat_list)
        ## add skip connection from q to a_attent
        q_attent = q + q_attent
        q_attent = self.norm(q_attent)
        q_m = self.mean_encoder(q_attent)
        q_v = torch.exp(self.var_encoder(q_attent))
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        ### add the variational mean to the return
        if self.return_dist:
            return dist, q_m, latent, q_attent
        return q_m, q_v, latent, q_attent