import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
from einops import rearrange, repeat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GDEncoder(nn.Module):
    def __init__(self):
        super(GDEncoder, self).__init__()
        self.device = device
        self.n_head = 4
        self.att_out = 48
        self.in_length = 5
        self.out_length = 1
        self.f_length = 8
        self.relu_param = 0.1
        self.traj_linear_hidden = 32
        self.lstm_encoder_size = 64
        self.GAT_encoder_size = 32
        self.GAT_head = 8
        self.use_maneuvers = False
        self.use_elu = True
        self.use_spatial = False
        self.dropout = 0

        self.linear1 = nn.Linear(self.f_length, self.traj_linear_hidden)
        self.lstm = nn.LSTM(self.traj_linear_hidden, self.lstm_encoder_size)

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)
        
        self.GAT = GATConv(self.lstm_encoder_size, self.GAT_encoder_size, self.GAT_head, dropout=0.6)
        self.GAT_Dense = nn.Linear(self.GAT_encoder_size*self.GAT_head, 64)
        
        self.qt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.kt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)
        self.vt = nn.Linear(self.lstm_encoder_size, self.n_head * self.att_out)

        self.second_glu = GLU(
            input_size=24576,
            hidden_layer_size=self.lstm_encoder_size,
            dropout_rate=self.dropout)

        self.addAndNorm = AddAndNorm(self.lstm_encoder_size)
        self.fc = nn.Linear(self.lstm_encoder_size * 2, self.lstm_encoder_size)

    def forward(self, hist, adj):
        hist_enc = self.activation(self.linear1(hist))
        new_shape = (128 * 5, 40, 32)
        hist_enc_reshaped = hist_enc.reshape(new_shape)

        hist_hidden_enc, (_, _) = self.lstm(hist_enc_reshaped)
        values = hist_hidden_enc

        hist_hidden_enc = hist_hidden_enc.permute(0,2,1)

        embed_size = self.n_head * self.att_out
        qt = torch.cat(torch.split(self.qt(values), int(embed_size / self.n_head), -1), 0)
        kt = torch.cat(torch.split(self.kt(values), int(embed_size / self.n_head), -1), 0).permute(0, 2, 1)
        vt = torch.cat(torch.split(self.vt(values), int(embed_size / self.n_head), -1), 0)
        a = torch.matmul(qt, kt)
        a /= math.sqrt(self.lstm_encoder_size) 
        a = torch.softmax(a, -1)
        values = torch.matmul(a, vt)
        values = torch.cat(torch.split(values, int(hist.shape[1]), 0), -1)

        time_values, _ = self.second_glu(values)

        if self.use_spatial:
            values = self.addAndNorm(hist_hidden_enc, time_values)
        else:
            values = self.addAndNorm(hist_hidden_enc, time_values)

        return values


class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            assert x1.size() == x2.size() == x3.size(), "Input tensor sizes must match."
            x = torch.add(torch.add(x1, x2), x3)
        else:
            assert x1.size() == x2.size(), "Input tensor sizes must match."
            x = torch.add(x1, x2)
        return self.normalize(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.relu_param = 0.1
        self.use_elu = True
        self.in_length = 5
        self.out_length = 1
        self.encoder_size = 64
        self.n_head = 4
        self.att_out = 48
        self.device = device

        if self.use_elu:
            self.activation = nn.ELU()
        else:
            self.activation = nn.LeakyReLU(self.relu_param)

        self.lstm = torch.nn.LSTM(self.encoder_size, self.encoder_size)

        self.linear1 = nn.Linear(self.encoder_size, self.out_length)

    def forward(self, dec):
        h_dec, _ = self.lstm(dec)

        fut_pred = self.linear1(h_dec)

        return fut_pred


class GLU(nn.Module):
    def __init__(self, input_size, hidden_layer_size, dropout_rate=None):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = torch.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = torch.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return torch.mul(activation, gated), gated
