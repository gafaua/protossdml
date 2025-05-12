import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseEMA(nn.Module):
    def __init__(self, base_encoder, proj_dim=1024, out_dim=384, momentum=0.999, projection_head = False) -> None:
        super(SiameseEMA, self).__init__()
        self.momentum = momentum

        if projection_head:
            print("PROJECTION HEAD ADDED TO BACKBONE")
            self.encoder_q = nn.Sequential(
                base_encoder,
                nn.Linear(out_dim, proj_dim, bias=False),
                nn.BatchNorm1d(proj_dim),
                nn.ReLU(inplace=True),   # hidden layer
                nn.Linear(proj_dim, out_dim), # output layer
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),   # hidden layer
                nn.Linear(out_dim, out_dim) # output layer
            )
        else:
            print("NO PROJECTION HEAD")
            self.encoder_q = base_encoder

        self.encoder_k = copy.deepcopy(self.encoder_q)


    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)


    def forward(self, x1, x2):
        q = self.encoder_q(x1)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(x2)
            k = F.normalize(k, dim=1)

        return q, k


