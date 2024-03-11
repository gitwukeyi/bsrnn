# !/user/bin/env python
# -*-coding:utf-8 -*-

"""
# File : local_axial_self_attention.py
# Time : 2023/8/24 下午5:01
# Author : wukeyi
# version : python3.9
"""
import torch
from thop import profile
from torch import nn, Tensor

from modules.masks import local_mask, casual_mask


class AxialSelfAttention(nn.Module):
    def __init__(self, sub_bands: int, chunk_size: int, encoded_dim: int):
        """
         Unlike pixel or patch level attention in computer vision, an ASA mechanism for speech is
         proposed in this paper:https://ieeexplore.ieee.org/document/9746610
         The ASA can reduce the need for memory and computation, which is more suitable for long sequence
         signals such as speech.
         frequency mask, cf become a port of module, it does not need to be passed in.
        :param sub_bands: (K) sub bands.
        :param encoded_dim: (N) frequency encoded dimension.
        """
        super(AxialSelfAttention, self).__init__()
        assert chunk_size < sub_bands, f"chunk_size{chunk_size} >= sub_bands{sub_bands} is not allowed!"

        fre_mask = local_mask()(shape=(1, 1, sub_bands, sub_bands), chunk_size=chunk_size)

        # unchanged value during train stage
        self.fre_mask = nn.Parameter(fre_mask, requires_grad=False)
        cf = torch.sqrt(torch.tensor(sub_bands * encoded_dim / 2))
        self.cf = nn.Parameter(cf, requires_grad=False)

        # frequency mask would not be changed even though different inputs until frequency dim changed.

        self.frequency_point_wise = nn.Sequential(
            nn.Conv2d(encoded_dim, encoded_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim // 2),
            nn.PReLU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(encoded_dim // 2, encoded_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim),
            nn.PReLU()
        )

        self.time_norm = nn.LayerNorm(normalized_shape=[encoded_dim], elementwise_affine=True)
        self.time_lstm = nn.GRU(input_size=encoded_dim, hidden_size=encoded_dim * 2, num_layers=1,
                                bidirectional=False, batch_first=True)
        self.time_fc = nn.Linear(in_features=encoded_dim * 2, out_features=encoded_dim // 2)

    def forward(self, inputs: Tensor):
        """
        :param inputs: shape=(B, N, T, K)
        :return:
        """

        # time LSTM
        B, N, T, K = inputs.shape
        out_t = torch.permute(inputs, dims=(0, 3, 2, 1)).contiguous()  # (B, K, T, N)
        out_t = self.time_norm(out_t)  # per norm

        out_t = out_t.reshape(B * K, T, N)
        out_t, _ = self.time_lstm(out_t)  # (B*K, T, N)
        out_t = self.time_fc(out_t)

        out_t = torch.reshape(out_t, shape=(B, K, T, N // 2))
        out_t = torch.permute(out_t, dims=(0, 2, 1, 3)).contiguous()  # (B, T, K, N)

        # sub band attention
        xf = self.frequency_point_wise(inputs)
        qurry = torch.permute(xf, dims=(0, 2, 3, 1)).contiguous()  # (B, T, K, N)
        key = torch.permute(xf, dims=(0, 2, 1, 3)).contiguous()  # (B, T, N, K)

        attention_f = torch.matmul(qurry, key) / self.cf  # (B, T, K, K)
        attention_f = torch.softmax(attention_f + self.fre_mask, dim=-1)
        out_f = torch.matmul(attention_f, out_t)  # (B, T, K, K) * (B, T, K, N) = (B, T, K, N)

        # output reshape and residual add
        out_f = torch.permute(out_f, dims=(0, 3, 1, 2)).contiguous()  # (B, N, T, K)
        out_f = self.out_conv(out_f)
        out = inputs + out_f

        return out


class AxialSelfAttention2(nn.Module):
    def __init__(self, encoded_dim: int, chunk_size: int, sub_bands: int):
        """
         Unlike pixel or patch level attention in computer vision, an ASA mechanism for speech is
         proposed in this paper:https://ieeexplore.ieee.org/document/9746610
         The ASA can reduce the need for memory and computation, which is more suitable for long sequence
         signals such as speech.
         difference in AxialSelfAttention: frequency mask, ct, cf become a port of module, it does not need to be
         passed in. time mask should move into the same device as modules firstly.
         as a parameter.
        :param encoded_dim:
        """
        super(AxialSelfAttention2, self).__init__()
        assert chunk_size < sub_bands, f"chunk_size{chunk_size} >= fre_dim{sub_bands} is not allowed!"

        fre_mask = local_mask()(shape=(1, 1, sub_bands, sub_bands), chunk_size=chunk_size)

        # unchanged value during train stage
        self.fre_mask = nn.Parameter(fre_mask, requires_grad=False)
        get_mask = casual_mask()
        time_mask = get_mask(shape=(1, 1, 300, 300))
        self.time_mask = nn.Parameter(time_mask, requires_grad=False)

        cf = torch.sqrt(torch.tensor(encoded_dim * sub_bands / 2))
        self.cf = nn.Parameter(cf, requires_grad=False)
        ct = torch.sqrt(torch.tensor(encoded_dim * 300 / 2))
        self.ct = nn.Parameter(ct, requires_grad=False)
        # frequency mask would not be changed even though different inputs if frequency dim not changes.

        self.frequency_point_wise = nn.Sequential(
            nn.Conv2d(encoded_dim, encoded_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim // 4),
            nn.PReLU()
        )

        self.time_point_wise = nn.Sequential(
            nn.Conv2d(encoded_dim, encoded_dim // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim // 4),
            nn.PReLU()
        )

        self.output_point_wise = nn.Sequential(
            nn.Conv2d(encoded_dim // 4, encoded_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim),
            nn.PReLU()
        )

    def forward(self, inputs: Tensor):
        """
        :param inputs: shape=(B, C, T, F)
        :return:
        """
        # frequency attention
        xf = self.frequency_point_wise(inputs)
        qurry = value = torch.permute(xf, dims=(0, 2, 3, 1))  # (B, T, F, C)
        key = torch.permute(xf, dims=(0, 2, 1, 3))  # (B, T, C, F)

        attention_f = torch.matmul(qurry, key) / self.cf  # (B, T, F, F)
        attention_f = torch.softmax(attention_f + self.fre_mask, dim=-1)
        out_f = torch.matmul(attention_f, value)  # (B, T, F, C)

        out_f = torch.transpose(out_f, dim0=1, dim1=2)  # ( B, F, T, C)

        # time attention
        xt = self.time_point_wise(inputs)
        qurry = torch.permute(xt, dims=(0, 3, 2, 1))  # (B, F, T, C)
        key = torch.permute(xt, dims=(0, 3, 1, 2))  # （B, F, C, T)

        attention_t = torch.matmul(qurry, key) / self.ct  # (B, F, T, T)
        attention_t = torch.softmax(attention_t + self.time_mask, dim=-1)
        out_t = torch.matmul(attention_t, out_f)  # (B, F, T, T) x (B, F, T, C) = (B, F, T, C)

        # output reshape and residual add
        out = torch.permute(out_t, dims=(0, 3, 2, 1))  # (B, C, T, F)
        out = self.output_point_wise(out)
        out = out + inputs

        return out


class AxialSelfAttention3(nn.Module):
    def __init__(self, sub_bands: int, chunk_size: int, encoded_dim: int):
        """
         Unlike pixel or patch level attention in computer vision, an ASA mechanism for speech is
         proposed in this paper:https://ieeexplore.ieee.org/document/9746610
         The ASA can reduce the need for memory and computation, which is more suitable for long sequence
         signals such as speech.
         frequency mask, cf become a port of module, it does not need to be passed in.
        :param sub_bands: (K) sub bands.
        :param encoded_dim: (N) frequency encoded dimension.
        """
        super(AxialSelfAttention3, self).__init__()
        assert chunk_size < sub_bands, f"chunk_size{chunk_size} >= sub_bands{sub_bands} is not allowed!"

        fre_mask = local_mask()(shape=(1, 1, sub_bands, sub_bands), chunk_size=chunk_size)

        # unchanged value during train stage
        self.fre_mask = nn.Parameter(fre_mask, requires_grad=False)
        cf = torch.sqrt(torch.tensor(sub_bands * encoded_dim / 2))
        self.cf = nn.Parameter(cf, requires_grad=False)

        # frequency mask would not be changed even though different inputs until frequency dim changed.

        self.sub_band_query = nn.Sequential(
            nn.Conv2d(encoded_dim, encoded_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim // 2),
            nn.PReLU()
        )

        self.sub_band_key = nn.Sequential(
            nn.Conv2d(encoded_dim, encoded_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim // 2),
            nn.PReLU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(encoded_dim // 2, encoded_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(encoded_dim),
            nn.PReLU()
        )

        self.time_norm = nn.LayerNorm(normalized_shape=[encoded_dim], elementwise_affine=True)
        self.time_lstm = nn.GRU(input_size=encoded_dim, hidden_size=encoded_dim * 2, num_layers=1,
                                bidirectional=False, batch_first=True)
        self.time_fc = nn.Linear(in_features=encoded_dim * 2, out_features=encoded_dim // 2)

    def forward(self, inputs: Tensor):
        """
        :param inputs: shape=(B, N, T, K)
        :return:
        """

        # time LSTM
        B, N, T, K = inputs.shape
        out_t = torch.permute(inputs, dims=(0, 3, 2, 1)).contiguous()  # (B, K, T, N)
        out_t = self.time_norm(out_t)  # per norm

        out_t = out_t.reshape(B * K, T, N)
        out_t, _ = self.time_lstm(out_t)  # (B*K, T, N)
        out_t = self.time_fc(out_t)

        out_t = torch.reshape(out_t, shape=(B, K, T, N // 2))
        out_t = torch.permute(out_t, dims=(0, 2, 1, 3)).contiguous()  # (B, T, K, N)

        # sub band attention
        query = self.sub_band_query(inputs)
        qurry = torch.permute(query, dims=(0, 2, 3, 1)).contiguous()  # (B, T, K, N)
        key = self.sub_band_key(inputs)
        key = torch.permute(key, dims=(0, 2, 1, 3)).contiguous()  # (B, T, N, K)

        attention_f = torch.matmul(qurry, key) / self.cf  # (B, T, K, K)
        attention_f = torch.softmax(attention_f + self.fre_mask, dim=-1)
        out_f = torch.matmul(attention_f, out_t)  # (B, T, K, K) * (B, T, K, N) = (B, T, K, N)

        # output reshape and residual add
        out_f = torch.permute(out_f, dims=(0, 3, 1, 2)).contiguous()  # (B, N, T, K)
        out_f = self.out_conv(out_f)
        out = inputs + out_f

        return out


class AxialSelfAttention4(nn.Module):
    def __init__(self, sub_bands: int, encoded_dim: int):
        """
         Unlike pixel or patch level attention in computer vision, an ASA mechanism for speech is
         proposed in this paper:https://ieeexplore.ieee.org/document/9746610
         The ASA can reduce the need for memory and computation, which is more suitable for long sequence
         signals such as speech.
         difference in AxialSelfAttention: frequency mask, cf become a port of module, it does not need to be
         passed in.
        :param sub_bands: (K) sub bands.
        :param encoded_dim: (N) frequency encoded dimension.
        """
        super(AxialSelfAttention4, self).__init__()

        self.time_norm = nn.LayerNorm(normalized_shape=[encoded_dim, sub_bands], elementwise_affine=True)
        self.time_gru = nn.GRU(input_size=encoded_dim, hidden_size=encoded_dim * 2, num_layers=1,
                               bidirectional=False, batch_first=True)
        self.time_fc = nn.Linear(in_features=encoded_dim * 2, out_features=encoded_dim)

        self.sub_band_norm = nn.LayerNorm(normalized_shape=[encoded_dim, sub_bands], elementwise_affine=True)
        self.sub_band_gru = nn.GRU(input_size=encoded_dim, hidden_size=encoded_dim, num_layers=1,
                                   bidirectional=True, batch_first=True)
        self.sub_band_fc = nn.Linear(in_features=encoded_dim * 2, out_features=encoded_dim)

    def forward(self, inputs: Tensor):
        """
        :param inputs: shape=(B, N, T, K)
        :return:
        """

        # time LSTM
        B, N, T, K = inputs.shape
        out_t = torch.permute(inputs, dims=(0, 2, 1, 3)).contiguous()  # (B, T, N, K)
        out_t = self.time_norm(out_t)  # per norm
        out_t = torch.permute(out_t, dims=(0, 3, 1, 2)).contiguous()  # (B, K, T, N)

        out_t = out_t.reshape(B * K, T, N)
        out_t, _ = self.time_gru(out_t)  # (B*K, T, N)
        out_t = self.time_fc(out_t)

        out_t = torch.reshape(out_t, shape=(B, K, T, N))
        out_t = torch.permute(out_t, dims=(0, 3, 2, 1)).contiguous()  # (B, N, T, K)
        out_t = inputs + out_t

        # sub band attention
        out_sub_band = torch.permute(out_t, dims=(0, 2, 1, 3)).contiguous()  # (B, T, N, K)
        out_sub_band = self.sub_band_norm(out_sub_band)  # per norm
        out_sub_band = torch.permute(out_sub_band, dims=(0, 1, 3, 2)).contiguous()  # (B, T, K, N)

        out_sub_band = torch.reshape(out_sub_band, shape=(B * T, K, N))
        out_sub_band, _ = self.sub_band_gru(out_sub_band)
        out_sub_band = self.sub_band_fc(out_sub_band)
        out_sub_band = torch.reshape(out_sub_band, (B, T, K, N))
        out_sub_band = torch.permute(out_sub_band, dims=(0, 3, 1, 2)).contiguous()

        out = out_sub_band + out_t

        return out


if __name__ == "__main__":

    in_data = torch.randn(1, 32, 300, 32)
    layer = AxialSelfAttention2(sub_bands=32, chunk_size=8, encoded_dim=32)
    out_data = layer(in_data)

    # example: ignore parameters do not need to be loaded when load modules parameters.
    torch.save(layer.state_dict(), f="test.pth.rar")

    model_dict = torch.load("test.pth.rar")

    new_dict = {}
    for keys, values in model_dict.items():
        if keys not in ["cf", "fre_mask"]:
            new_dict[keys] = values

    new_layer = AxialSelfAttention2(sub_bands=32, chunk_size=12, encoded_dim=32)

    new_layer.load_state_dict(new_dict, strict=False)
    macs, params = profile(layer, inputs=(in_data,))
    print(f"mac: {macs / 1e9} G \nparams: {params / 1e6}M")
