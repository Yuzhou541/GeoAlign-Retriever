from __future__ import annotations
import torch
import torch.nn as nn

class KGEModel(nn.Module):
    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_entity_representations_for_alignment(self) -> torch.Tensor:
        raise NotImplementedError
