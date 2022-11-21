import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from functools import reduce
from utils import nice_print, mem_report, cpu_stats
import copy
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat


class E3DLSTM(nn.Module):
    def __init__(self, input_shape, hidden_size, num_layers, kernel_size):
        super().__init__()

        # self._tau = tau
        self._cells = []

        input_shape = list(input_shape)
        for i in range(num_layers):
            cell = E3DLSTMCell(input_shape, hidden_size, kernel_size)
            # NOTE hidden state becomes input to the next cell
            input_shape[0] = hidden_size
            self._cells.append(cell)
            # Hook to register submodule
            setattr(self, "cell{}".format(i), cell)

    def forward(self, input):
        # NOTE (seq_len, batch, input_shape)
        batch_size = input.size(1)
        c_history_states = []
        h_states = []
        outputs = []
        attns = []
        seq_len = input.shape[0]

        for step, x in enumerate(input):
            attn_lis = []
            for cell_idx, cell in enumerate(self._cells):
                if step == 0:
                    c_history, m, h = self._cells[cell_idx].init_hidden(
                        batch_size,seq_len-1, input.device
                    )
                    c_history_states.append(c_history)
                    h_states.append(h)

                # NOTE c_history and h are coming from the previous time stamp, but we iterate over cells
                c_history, m, h,attn = cell(
                    x, c_history_states[cell_idx], m, h_states[cell_idx]
                )
                c_history_states[cell_idx] = c_history
                h_states[cell_idx] = h
                attn_lis.append(attn)
                # NOTE hidden state of previous LSTM is passed as input to the next one
                x = h
            outputs.append(h)
            attns.append(attn_lis)


        self.attns = attns
        return outputs[-1]



class E3DLSTMCell(nn.Module):
    def __init__(self, input_shape, hidden_size, kernel_size):
        super().__init__()

        in_channels = input_shape[0]
        self._input_shape = input_shape
        self._hidden_size = hidden_size

        # memory gates: input, cell(input modulation), forget
        self.weight_xi = ConvDeconv3d(in_channels, hidden_size, kernel_size)
        self.weight_hi = ConvDeconv3d(hidden_size, hidden_size, kernel_size, bias=False)

        self.weight_xg = copy.deepcopy(self.weight_xi)
        self.weight_hg = copy.deepcopy(self.weight_hi)

        self.weight_xr = copy.deepcopy(self.weight_xi)
        self.weight_hr = copy.deepcopy(self.weight_hi)

        memory_shape = list(input_shape)
        memory_shape[0] = hidden_size

        self.layer_norm = nn.LayerNorm(memory_shape)

        # for spatiotemporal memory
        self.weight_xi_prime = copy.deepcopy(self.weight_xi)
        self.weight_mi_prime = copy.deepcopy(self.weight_hi)

        self.weight_xg_prime = copy.deepcopy(self.weight_xi)
        self.weight_mg_prime = copy.deepcopy(self.weight_hi)

        self.weight_xf_prime = copy.deepcopy(self.weight_xi)
        self.weight_mf_prime = copy.deepcopy(self.weight_hi)

        self.weight_xo = copy.deepcopy(self.weight_xi)
        self.weight_ho = copy.deepcopy(self.weight_hi)
        self.weight_co = copy.deepcopy(self.weight_hi)
        self.weight_mo = copy.deepcopy(self.weight_hi)

        self.weight_111 = nn.Conv3d(hidden_size + hidden_size, hidden_size, 1)

    def self_attention(self, r, c_history):
        batch_size = r.size(0)
        channels = r.size(1)
        r_flatten = r.view(batch_size, -1, channels)
        # BxtaoTHWxC
        c_history_flatten = c_history.view(batch_size, -1, channels)

        # Attention mechanism
        # BxTHWxC x BxtaoTHWxC' = B x THW x taoTHW
        scores = torch.einsum("bxc,byc->bxy", r_flatten, c_history_flatten)
        attention = F.softmax(scores, dim=0)

        return torch.einsum("bxy,byc->bxc", attention, c_history_flatten).view(*r.shape),

    def self_attention_fast(self, r, c_history):
        # Scaled Dot-Product but for tensors
        # instead of dot-product we do matrix contraction on twh dimensions
        scaling_factor = 1 / (reduce(operator.mul, r.shape[-3:], 1) ** 0.5)
        scores = torch.einsum("bctwh,lbctwh->bl", r, c_history) * scaling_factor
        attention = F.softmax(scores,dim=-1)
        return torch.einsum("bl,lbctwh->bctwh", attention, c_history),attention

    def forward(self, x, c_history, m, h):
        # Normalized shape for LayerNorm is CxT×H×W
        normalized_shape = list(h.shape[-3:])

        def LR(input):
            return F.layer_norm(input, normalized_shape)

        # R is CxT×H×W
        r = torch.sigmoid(LR(self.weight_xr(x) + self.weight_hr(h)))
        i = torch.sigmoid(LR(self.weight_xi(x) + self.weight_hi(h)))
        g = torch.tanh(LR(self.weight_xg(x) + self.weight_hg(h)))

        recall,attention = self.self_attention_fast(r, c_history)
        # nice_print(**locals())
        # mem_report()
        # cpu_stats()

        c = i * g + self.layer_norm(c_history[-1] + recall)

        i_prime = torch.sigmoid(LR(self.weight_xi_prime(x) + self.weight_mi_prime(m)))
        g_prime = torch.tanh(LR(self.weight_xg_prime(x) + self.weight_mg_prime(m)))
        f_prime = torch.sigmoid(LR(self.weight_xf_prime(x) + self.weight_mf_prime(m)))

        m = i_prime * g_prime + f_prime * m
        o = torch.sigmoid(
            LR(
                self.weight_xo(x)
                + self.weight_ho(h)
                + self.weight_co(c)
                + self.weight_mo(m)
            )
        )
        h = o * torch.tanh(self.weight_111(torch.cat([c, m], dim=1)))

        # TODO is it correct FIFO?
        c_history = torch.cat([c_history[1:], c[None, :]], dim=0)
        # nice_print(**locals())

        return (c_history, m, h,attention.cpu().tolist())

    def init_hidden(self, batch_size, tau, device=None):
        memory_shape = list(self._input_shape)
        memory_shape[0] = self._hidden_size
        c_history = torch.zeros(tau, batch_size, *memory_shape, device=device)
        m = torch.zeros(batch_size, *memory_shape, device=device)
        h = torch.zeros(batch_size, *memory_shape, device=device)

        return (c_history, m, h)


class ConvDeconv3d(nn.Module):
    def __init__(self, in_channels, out_channels, *vargs, **kwargs):
        super().__init__()

        self.conv3d = nn.Conv3d(in_channels, out_channels, *vargs, **kwargs)
        # self.conv_transpose3d = nn.ConvTranspose3d(out_channels, out_channels, *vargs, **kwargs)

    def forward(self, input):
        # print(self.conv3d(input).shape, input.shape)
        # return self.conv_transpose3d(self.conv3d(input))
        return F.interpolate(self.conv3d(input), size=input.shape[-3:], mode="nearest")


class MSTFNet(nn.Module):
    def __init__(self,frame,window,units,layer):
        super().__init__()
        self.frame=frame
        self.window=window
        self.units=units
        self.layer=layer
        self.convlstm = nn.Sequential(
            E3DLSTM(input_shape=(units, frame, 12, 16), hidden_size=units, num_layers=layer, kernel_size=(3, 3, 3)),
            nn.ReLU(),
        )

        self.conv3d_encoder = nn.Sequential(
            nn.Conv3d(2, units, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # b c d(frame) h w
            nn.ReLU(),
            nn.Conv3d(units, units, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 2
            nn.ReLU()
        )

        self.conv3d_decoder = nn.Sequential(
            nn.Conv3d(units, units, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 5
            nn.ReLU(),
            nn.Conv3d(units, 2, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 2
        )

        self.vector = nn.Sequential(
            nn.Linear(56, 10),
            nn.ReLU(),
            nn.Linear(10, 2 * 12 * 16),
        )

        self.map= nn.Sequential(
            nn.Conv2d(2*frame,2,1)
        )


    def forward(self, x1, x2,y=None): #x1:videos (b,f,c,h,w), f=n*t, n=fragments, t=frames in each fragment
        # encoder
        lis = []
        x1 = rearrange(x1, 'b (n t) c h w -> n b c t h w ', n=self.window)
        for t in range(self.window):
            x1_n = self.conv3d_encoder(x1[t])  # 维度不要弄错了
            lis.append(x1_n)
        x1 = torch.stack(lis)
        x1 = self.convlstm(x1)
        self.attns = self.convlstm[0].attns
        x1=self.conv3d_decoder(x1)
        x1 = rearrange(x1, 'b c t h w -> b (c t) h w')
        x1 = self.map(x1)
        x2 = self.vector(x2)
        x2=torch.reshape(x2,(-1,2,12,16))
        x = x1+x2
        if y is not None:
            loss=F.mse_loss(x, y)
            return loss
        else:
            return x

if __name__ == '__main__':
    x1 = np.ones((2,12,2,12, 16))  # videos (b,f,c,h,w), b=batch size, f=n*t, n=fragments, t=frames in each fragment, c=channel, h=height, w=width
    x1 = torch.tensor(x1, dtype=torch.float).cuda()
    x2 = np.ones((2, 56))  # external factors (b, feathers)
    x2 = torch.tensor(x2, dtype=torch.float).cuda()
    t_model = MSTFNet(4,3,32,2).cuda()
    y = t_model(x1, x2)
    print(y.shape)
