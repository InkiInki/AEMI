"""
Author: Inki
Email: inki.yinji@qq.com
Create: 2021 0406
Last modify: 2021 0601
"""

import torch.nn as nn
import torch.nn.functional as func_nn
import warnings

warnings.filterwarnings('ignore')


class ANET(nn.Module):

    def __init__(self,
                 num_att: int,
                 num_class: int,
                 mapping_len: int = 100
                 ):
        super(ANET, self).__init__()
        self.L = 512
        self.D = 128
        self.H1 = 128
        self.H2 = mapping_len

        self.feature_extractor = nn.Sequential(
            nn.Linear(num_att, self.L),
            nn.ReLU(),
        )

        self.attention_v = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
        )

        self.attention_u = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.L)

        self.mapping = nn.Sequential(
            nn.Linear(self.L, self.H1),
            nn.Sigmoid(),
            nn.Linear(self.H1, self.H2),
            nn.Sigmoid(),
            nn.Linear(self.H2, num_class),
        )

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x_a_v = self.attention_v(x_f)
        x_a_u = self.attention_u(x_f)

        x_a_w = self.attention_weights(x_a_v * x_a_u)
        x_a_w = func_nn.softmax(x_a_w, dim=1)

        x_m = x_a_w + x_f
        x_o = self.mapping(x_m)
        return x_o


class ENET(nn.Module):

    def __init__(self,
                 num_att: int,
                 num_class: int,
                 mapping_len: int = 100
                 ):
        super(ENET, self).__init__()
        self.H1 = 128
        self.H2 = mapping_len

        self.fc = nn.Sequential(  # The fully connected layer
            nn.Linear(num_att, self.H1),
            nn.Sigmoid(),
            nn.Linear(self.H1, self.H2),
            nn.Sigmoid(),
            nn.Linear(self.H2, num_class)
        )

    def forward(self, x):
        return self.fc(x)
