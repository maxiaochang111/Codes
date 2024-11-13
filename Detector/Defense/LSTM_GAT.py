import math
import torch
from torch import nn
import torch.nn.functional as F
from model.GAT import GAT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GDEncoder(nn.Module):
    def __init__(self):
        super(GDEncoder, self).__init__()
        self.device = device
        self.in_length = 5
        self.out_length = 1
        self.f_length = 8
        self.relu_param = 0.1
        self.use_elu = True
        self.traj_linear_hidden = 32
        self.lstm_encoder_size = 64

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)

        self.linear1 = nn.Linear(self.in_length*self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)
        self.GAT = GAT()


    def forward(self, hist, adj):
        reshaped_hist = hist.permute((0, 2, 3, 1)).flatten(2, 3)
        hist_enc = self.activation(self.linear1(reshaped_hist))
        hist_hidden_enc, (_, _) = self.lstm(hist_enc)
        GAT_out = self.GAT(hist_hidden_enc, adj[:, -1, :, :])
        values = GAT_out

        return values


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.device = device
        self.relu_param = 0.1
        self.use_elu = True
        self.in_length = 5
        self.out_length = 1
        self.f_length = 8
        self.encoder_size = 64

        self.lstm = torch.nn.LSTM(self.encoder_size, self.encoder_size)

        self.linear1 = nn.Linear(self.encoder_size, self.out_length*self.f_length)

    def forward(self, dec):
        h_dec, _ = self.lstm(dec)
        fut_pred = self.linear1(h_dec)

        return fut_pred
