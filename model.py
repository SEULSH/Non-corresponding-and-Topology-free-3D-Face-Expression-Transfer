
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from xcit import XCABlock1D
from dgcnn_modular import DGCNN_MODULAR

class ex_encoder(nn.Module):
    def __init__(self):
        super(ex_encoder, self).__init__()

        in_channels = 3
        out_channels = [32, 64, 128]
        latent_channels = 128

        eta = 1e-5

        # self.ex_mlp = nn.Sequential(
        #     nn.Linear(in_channels, out_channels[0], bias=True),
        #     torch.nn.ReLU(),
        #     nn.Linear(out_channels[0], out_channels[1], bias=True),
        #     torch.nn.ReLU(),
        #     nn.Linear(out_channels[1], out_channels[2], bias=True),
        # )

        # self.ex_mlp = nn.Sequential(
        #     nn.Conv1d(in_channels, out_channels[0], kernel_size=1, bias=True),
        #     nn.BatchNorm1d(out_channels[0]), nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(out_channels[0], out_channels[1], kernel_size=1, bias=True),
        #     nn.BatchNorm1d(out_channels[1]), nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(out_channels[1], out_channels[2], kernel_size=1, bias=True),
        # )

        self.ex_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[0], kernel_size=1, bias=True),
            nn.BatchNorm1d(out_channels[0]), nn.ReLU(),
            nn.Conv1d(out_channels[0], out_channels[1], kernel_size=1, bias=True),
            nn.BatchNorm1d(out_channels[1]), nn.ReLU(),
            nn.Conv1d(out_channels[1], out_channels[2], kernel_size=1, bias=True),
        )

        self.attention_Blocks = nn.ModuleList(
            [XCABlock1D(latent_channels, 8, eta=eta),
             XCABlock1D(latent_channels, 8, eta=eta)])

        self.local_geometry = DGCNN_MODULAR(num_neighs=9, in_features_dim=3,
                                            nn_depth=3,
                                            bb_size=8, latent_dim=latent_channels)
        self.fusion = nn.Sequential(
            nn.Linear(latent_channels*3, latent_channels, bias=True),
            # torch.nn.ReLU(),
        )


    def forward(self, expression):

        # mlp_f = self.ex_mlp(expression)  # B, N, D
        mlp_f = self.ex_mlp(expression.transpose(1,2)).transpose(1,2)  # B, N, D

        attention_f = mlp_f
        for B in self.attention_Blocks:
            attention_f = B(attention_f)  # B, N, D

        gcn_f = self.local_geometry(expression.transpose(1, 2))  # B, N, D

        ex_f = self.fusion(torch.cat([mlp_f, attention_f, gcn_f], dim=-1))

        return ex_f

class id_encoder(nn.Module):
    def __init__(self):
        super(id_encoder, self).__init__()

        in_channels = 3
        out_channels = [32, 64, 128]
        latent_channels = 128

        eta = 1e-5

        # self.identity_mlp = nn.Sequential(
        #     nn.Linear(in_channels, out_channels[0], bias=True),
        #     torch.nn.ReLU(),
        #     nn.Linear(out_channels[0], out_channels[1], bias=True),
        #     torch.nn.ReLU(),
        #     nn.Linear(out_channels[1], out_channels[2], bias=True),
        # )

        # self.identity_mlp = nn.Sequential(
        #     nn.Conv1d(in_channels, out_channels[0], kernel_size=1, bias=True),
        #     nn.BatchNorm1d(out_channels[0]), nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(out_channels[0], out_channels[1], kernel_size=1, bias=True),
        #     nn.BatchNorm1d(out_channels[1]), nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(out_channels[1], out_channels[2], kernel_size=1, bias=True),
        # )

        self.identity_mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels[0], kernel_size=1, bias=True),
            nn.BatchNorm1d(out_channels[0]), nn.ReLU(),
            nn.Conv1d(out_channels[0], out_channels[1], kernel_size=1, bias=True),
            nn.BatchNorm1d(out_channels[1]), nn.ReLU(),
            nn.Conv1d(out_channels[1], out_channels[2], kernel_size=1, bias=True),
        )

        self.attention_Blocks = nn.ModuleList(
            [XCABlock1D(latent_channels, 8, eta=eta),
             XCABlock1D(latent_channels, 8, eta=eta)])

        self.local_geometry = DGCNN_MODULAR(num_neighs=9, in_features_dim=3,
                                            nn_depth=3,
                                            bb_size=8, latent_dim=latent_channels)
        self.fusion = nn.Sequential(
            nn.Linear(latent_channels * 4, latent_channels, bias=True),
            # torch.nn.ReLU(),
        )

        self.enc_token = torch.nn.Parameter(torch.empty(latent_channels), requires_grad=True)
        nn.init.trunc_normal_(self.enc_token, std=0.2)
        self.global_attention_Blocks = nn.ModuleList(
            [XCABlock1D(latent_channels, 8, eta=eta),
             XCABlock1D(latent_channels, 8, eta=eta)])

    def forward(self, identity):

        # mlp_f = self.identity_mlp(identity)  # B, N, D
        mlp_f = self.identity_mlp(identity.transpose(1, 2)).transpose(1, 2)  # B, N, D

        attention_f = mlp_f
        for B in self.attention_Blocks:
            attention_f = B(attention_f)  # B, N, D

        gcn_f = self.local_geometry(identity.transpose(1, 2))  # B, N, D

        global_f = self.enc_token.unsqueeze(0).unsqueeze(1).repeat(mlp_f.shape[0], 1, 1)
        global_f = torch.cat([mlp_f, global_f], dim=1)  # B, N+1, D
        for B in self.global_attention_Blocks:
            global_f = B(global_f)  # B, N+1, D
        global_f = global_f[:, -1, :].unsqueeze(1)  # B, 1, D

        global_f = global_f.repeat(1, mlp_f.shape[1], 1)  # B, N, D

        id_f = self.fusion(torch.cat([mlp_f, attention_f, gcn_f, global_f], dim=-1))

        return id_f

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        latent_channels = 128
        eta = 1e-5

        self.conv1 = torch.nn.Conv1d(latent_channels * 2, latent_channels, 1)
        self.conv2 = torch.nn.Conv1d(latent_channels, latent_channels, 1)
        self.conv3 = torch.nn.Conv1d(latent_channels, 3, 1)
        self.shape_dec1 = nn.ModuleList(
            [XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta)])
        self.shape_dec2 = nn.ModuleList(
            [XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta)])

        self.th = nn.Tanh()

    def forward(self, ex_f, id_f):  # B, N, D; B, N, D

        x = torch.cat([ex_f, id_f], dim=-1)

        x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
        for dec in self.shape_dec1:
            x = dec(x)
        x = self.conv2(F.relu(x).transpose(1, 2)).transpose(1, 2)
        for dec in self.shape_dec2:
            x = dec(x)

        x = 2 * self.th(self.conv3(x.transpose(1, 2)))

        return x.transpose(1, 2)  # B, N, 3


class ExpressionTransfer(nn.Module):
    def __init__(self):
        super(ExpressionTransfer, self).__init__()

        self.ex_encoder = ex_encoder()

        self.id_encoder = id_encoder()

        self.decoder = decoder()

    # input: B,N,3; B,N,3
    def forward(self, ex_shape, id_shape):

        ex_f = self.ex_encoder(ex_shape)  # B,N,D

        id_f = self.id_encoder(id_shape)  # B,N,D

        y = self.decoder(ex_f, id_f)  # B, N, 3

        return y

# # class PoseFeature(nn.Module):
# #     def __init__(self):
# #         super(PoseFeature, self).__init__()
# #
# #         # self.conv1 = torch.nn.Conv1d(3, 64, 1)
# #         # self.conv11 = XCABlock1D(64, 4, eta=1e-5)
# #         # self.conv2 = torch.nn.Conv1d(64, 128, 1)
# #         # self.conv21 = XCABlock1D(128, 8, eta=1e-5)
# #         # self.conv3 = torch.nn.Conv1d(128, 256, 1)
# #         # self.norm1 = torch.nn.InstanceNorm1d(64)
# #         # self.norm2 = torch.nn.InstanceNorm1d(128)
# #         # self.norm3 = torch.nn.InstanceNorm1d(256)
# #
# #         self.conv1 = torch.nn.Conv1d(3, 32, 1)
# #         self.conv11 = XCABlock1D(32, 2, eta=1e-5)
# #         self.conv2 = torch.nn.Conv1d(32, 64, 1)
# #         self.conv21 = XCABlock1D(64, 4, eta=1e-5)
# #         self.conv3 = torch.nn.Conv1d(64, 128, 1)
# #         self.norm1 = torch.nn.InstanceNorm1d(32)
# #         self.norm2 = torch.nn.InstanceNorm1d(64)
# #         self.norm3 = torch.nn.InstanceNorm1d(128)
# #
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = self.conv11(x.transpose(1, 2))
# #         x = F.relu(self.norm1(x.transpose(1, 2)))
# #         x = self.conv2(x)
# #         x = self.conv21(x.transpose(1, 2))
# #         x = F.relu(self.norm2(x.transpose(1, 2)))
# #         x = self.conv3(x)
# #         x = F.relu(self.norm3(x))
# #
# #         return x
# #
# # class ExpressionTransfer(nn.Module):
# #     def __init__(self):
# #         super(ExpressionTransfer, self).__init__()
# #         eta = 1e-5
# #         in_channels = 3
# #         # out_channels = [64, 128, 256]
# #         # latent_channels = 256
# #         out_channels = [32, 64, 128]
# #         latent_channels = 128
# #
# #         self.pose_only = PoseFeature()
# #
# #         self.identity_mlp = nn.Sequential(
# #             nn.Linear(in_channels, out_channels[0], bias=True),
# #             torch.nn.ReLU(),
# #             nn.Linear(out_channels[0], out_channels[1], bias=True),
# #             torch.nn.ReLU(),
# #             nn.Linear(out_channels[1], out_channels[2], bias=True),
# #         )
# #         # self.shape_enc = nn.ModuleList(
# #         #     [XCABlock1D(latent_channels, 16, eta=eta), XCABlock1D(latent_channels, 16, eta=eta)])
# #         self.shape_enc = nn.ModuleList(
# #             [XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta)])
# #
# #         # self.conv1 = torch.nn.Conv1d(latent_channels*2, latent_channels, 1)
# #         # self.conv2 = torch.nn.Conv1d(latent_channels, latent_channels // 2, 1)
# #         # self.conv4 = torch.nn.Conv1d(latent_channels // 2, 3, 1)
# #         # self.shape_dec1 = nn.ModuleList(
# #         #     [XCABlock1D(latent_channels, 16, eta=eta), XCABlock1D(latent_channels, 16, eta=eta)])
# #         # self.shape_dec2 = nn.ModuleList(
# #         #     [XCABlock1D(latent_channels // 2, 8, eta=eta), XCABlock1D(latent_channels // 2, 8, eta=eta)])
# #         self.conv1 = torch.nn.Conv1d(latent_channels*2, latent_channels, 1)
# #         self.conv2 = torch.nn.Conv1d(latent_channels, latent_channels, 1)
# #         self.conv4 = torch.nn.Conv1d(latent_channels, 3, 1)
# #         self.shape_dec1 = nn.ModuleList(
# #             [XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta)])
# #         self.shape_dec2 = nn.ModuleList(
# #             [XCABlock1D(latent_channels, 8, eta=eta), XCABlock1D(latent_channels, 8, eta=eta)])
# #
# #         self.th = nn.Tanh()
# #
# #
# #     def encode(self, identity):
# #         identity = self.inference_model(identity)
# #         return identity  #
# #     def inference_model(self, identity):
# #
# #         identity_f = self.identity_mlp(identity)
# #         for enc in self.shape_enc:
# #             identity_f = enc(identity_f)  # x: B, N+1, latent dimensions
# #
# #         return identity_f
# #
# #     def decode(self, identity_f, pose_f):
# #         logits = self.generative_model(identity_f, pose_f)
# #         return logits
# #
# #     def generative_model(self, identity_f, pose_f):
# #
# #         x = torch.cat([identity_f, pose_f], dim=-1)
# #         x = self.conv1(x.transpose(1, 2)).transpose(1, 2)
# #         for dec in self.shape_dec1:
# #             x = dec(x)
# #         x = self.conv2(F.relu(x).transpose(1,2)).transpose(1,2)
# #         for dec in self.shape_dec2:
# #             x = dec(x)
# #
# #         x = 2 * self.th(self.conv4(x.transpose(1, 2)))
# #
# #         return x.transpose(1, 2)
# #
# #     # input: B,N,3; B,N,3
# #     def forward(self, pose, identity):
# #
# #         pose_f = self.pose_only(pose.transpose(1, 2))  # B,D,N
# #
# #         identity_f = self.encode(identity)  # B, N, D
# #
# #         y = self.decode(identity_f, pose_f.transpose(1, 2))  # B, N, 3
# #
# #         return y