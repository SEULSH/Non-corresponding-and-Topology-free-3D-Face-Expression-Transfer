import torch
import torch.nn as nn
import math
import pdb
import copy
from numpy import inf
# from sparsemax import Sparsemax
import torch.nn.functional as F
import math


# whole version, res
class PaiConv(nn.Module):
    def __init__(self, num_pts, in_c, num_neighbor, out_c, activation='elu', bias=True):  # ,device=None):
        super(PaiConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv = nn.Linear(in_c * num_neighbor, out_c, bias=bias)
        self.adjweight = nn.Parameter(torch.randn(num_pts, num_neighbor, num_neighbor), requires_grad=True)
        self.adjweight.data = torch.eye(num_neighbor).unsqueeze(0).expand_as(self.adjweight)
        self.zero_padding = torch.ones((1, num_pts, 1))
        self.zero_padding[0, -1, 0] = 0.0
        self.mlp_out = nn.Linear(in_c, out_c)  # for res connection
        # self.sparsemax = Sparsemax(dim=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, t_vertex, neighbor_index):
        bsize, num_pts, feats = x.size()
        _, _, num_neighbor = neighbor_index.size()

        x = x * self.zero_padding.to(x.device)
        neighbor_index = neighbor_index.view(bsize * num_pts * num_neighbor)  # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=x.device).view(-1, 1).repeat([1, num_pts * num_neighbor]).view(
            -1).long()
        x_neighbors = x[batch_index, neighbor_index, :].view(bsize, num_pts, num_neighbor, feats)
        # x_neighbors = x_neighbors.view(num_pts, bsize*feats, num_neighbor)
        # weight = self.softmax(torch.bmm(torch.transpose(x_neighbors, 1, 2), x_neighbors))
        # x_neighbors = torch.bmm(x_neighbors, weight) #.view(num_pts, feats, num_neighbor)
        x_neighbors = torch.einsum('bnkf, bnkt->bntf', x_neighbors,
                                   self.adjweight[None].repeat(bsize, 1, 1, 1))  # self.sparsemax(self.adjweight))
        x_neighbors = self.activation(x_neighbors.contiguous().view(bsize * num_pts, num_neighbor * feats))
        out_feat = self.activation(self.conv(x_neighbors)).view(bsize, num_pts, self.out_c)
        out_feat = out_feat * self.zero_padding.to(out_feat.device)
        x_res = self.mlp_out(x.view(-1, self.in_c)).view(bsize, -1, self.out_c)
        return out_feat + x_res


class PaiAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size,
                 t_vertices, sizes, num_neighbors, x_neighbors, D, U, flag, activation='elu'):
        super(PaiAutoencoder, self).__init__()
        self.flag = flag
        self.latent_size = latent_size
        self.sizes = sizes  # real vertex number
        # add center node index, the first index value of each row
        self.x_neighbors = [
            torch.cat([torch.cat([torch.arange(x.shape[0] - 1), torch.tensor([-1])]).unsqueeze(1), x], 1) for x in
            x_neighbors]
        # self.x_neighbors = [x.float().cuda() for x in x_neighbors]
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.num_neighbors = num_neighbors
        self.D = [nn.Parameter(x, False) for x in D]
        self.D = nn.ParameterList(self.D)
        self.U = [nn.Parameter(x, False) for x in U]
        self.U = nn.ParameterList(self.U)

        mappingsize = 64  #
        #
        self.B = nn.Parameter(torch.randn(6, mappingsize), requires_grad=False)
        # cat difference between neighbors' mean value and each template vertex?
        self.t_vertices = [torch.cat([x[self.x_neighbors[i]][:, 1:].mean(dim=1) - x, x], dim=-1) for i, x in
                           enumerate(t_vertices)]
        #
        self.t_vertices = [2. * math.pi * x @ (self.B.data).to(x) for x in self.t_vertices]
        #
        self.t_vertices = [((x - x.min(dim=0, keepdim=True)[0]) / (x.max(dim=0, keepdim=True)[0] \
                                                                   - x.min(dim=0, keepdim=True)[0]) - 0.5) * 100 for x
                           in self.t_vertices]
        #
        self.t_vertices = [torch.cat([torch.sin(x), torch.cos(x)], dim=-1) for x in self.t_vertices]

        self.eps = 1e-7
        # self.reset_parameters()
        # self.device = device
        self.activation = activation
        self.conv = []
        input_size = filters_enc[0]
        for i in range(len(num_neighbors) - 1):
            self.conv.append(PaiConv(self.x_neighbors[i].shape[0], input_size, num_neighbors[i], filters_enc[i + 1],
                                     activation=self.activation))

            input_size = filters_enc[i + 1]

        self.conv = nn.ModuleList(self.conv)

        self.fc_latent_enc = nn.Linear((sizes[-1] + 1) * input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1] + 1) * filters_dec[0])

        self.dconv = []
        input_size = filters_dec[0]
        for i in range(len(num_neighbors) - 1):
            self.dconv.append(
                PaiConv(self.x_neighbors[-2 - i].shape[0], input_size, num_neighbors[-2 - i], filters_dec[i + 1],
                        activation=self.activation))

            input_size = filters_dec[i + 1]

            if i == len(num_neighbors) - 2:
                input_size = filters_dec[-2]
                self.dconv.append(
                    PaiConv(self.x_neighbors[-2 - i].shape[0], input_size, num_neighbors[-2 - i], filters_dec[-1],
                            activation='identity'))

        self.dconv = nn.ModuleList(self.dconv)

    def poolwT(self, x, L):
        Mp = L.shape[0]
        N, M, Fin = x.shape
        # Rescale transform Matrix L and store as a TF sparse tensor. Copy to not modify the shared L.
        x = x.permute(1, 2, 0).contiguous()  # M x Fin x N
        x = x.view(M, Fin * N)  # M x Fin*N

        x = torch.spmm(L, x)  # Mp x Fin*N
        x = x.view(Mp, Fin, N)  # Mp x Fin x N
        x = x.permute(2, 0, 1).contiguous()  # N x Mp x Fin
        return x

    def encode(self, x):
        bsize = x.size(0)
        S = self.x_neighbors
        D = self.D
        t_vertices = self.t_vertices
        for i in range(len(self.num_neighbors) - 1):
            x = self.conv[i](x, t_vertices[i], S[i].repeat(bsize, 1, 1))
            # x = torch.matmul(D[i],x)
            x = self.poolwT(x, D[i])
        # x = self.conv[-1](x, t_vertices[-1], S[-1].repeat(bsize,1,1))
        x = x.view(bsize, -1)
        return self.fc_latent_enc(x)

    def decode(self, z):
        bsize = z.size(0)
        S = self.x_neighbors
        U = self.U
        t_vertices = self.t_vertices

        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1] + 1, -1)

        for i in range(len(self.num_neighbors) - 1):
            # x = torch.matmul(U[-1-i],x)
            x = self.poolwT(x, U[-1 - i])
            x = self.dconv[i](x, t_vertices[-2 - i], S[-2 - i].repeat(bsize, 1, 1))
        x = self.dconv[-1](x, t_vertices[0], S[0].repeat(bsize, 1, 1))
        return x

    def forward(self, ratio, x1, x2, x3=None):

        N, D = x1.shape[:]
        x1 = x1.reshape(1, -1, D)  # B, N, C
        x2 = x2.reshape(1, -1, D)  # B, N, C

        zeros1 = torch.zeros([1, N+1, D], dtype=x1.dtype, device=x1.device)
        zeros1[:, 0:N, :] = x1
        x1 = zeros1

        zeros2 = torch.zeros([1, N+1, D], dtype=x1.dtype, device=x1.device)
        zeros2[:, 0:N, :] = x2
        x2 = zeros2

        x3 = x3.reshape(1, -1, D)  # B, N, C
        zeros3 = torch.zeros([1, N + 1, D], dtype=x1.dtype, device=x1.device)
        zeros3[:, 0:N, :] = x3
        x3 = zeros3

        z1 = self.encode(x1)
        z2 = self.encode(x2)
        z3 = self.encode(x3)
        z0 = z3 + (z2 - z1)
        x = self.decode(z0)
        x = x.reshape(-1, D)  # N * 3

        return x