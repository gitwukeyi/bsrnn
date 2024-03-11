# !/user/bin/env python
# -*-coding:utf-8 -*-

"""
# File : bsrnn.py
# Time : 2024/2/25 下午3:45
# Author : wukeyi
# version : python3.9
"""

import torch
from thop import profile
from torch import nn, Tensor

from modules.en_decoder import SubBandEncoder, SubBandDecoder
from modules.local_axial_self_attention import AxialSelfAttention, AxialSelfAttention2, AxialSelfAttention3, \
    AxialSelfAttention4


class BandSplit(nn.Module):
    def __init__(self, sub_band_list: list, encode_dim: int):
        """
        divided full band spectra into sub band spectra with different width
        :param sub_band_list: a list of different sub band widths, from low frequency to high frequency.
        :param encode_dim: all sub bands with different width will be encoded into same dimension.
        """
        super().__init__()

        start_idx = 0
        self.sub_band_encoders = nn.ModuleList()

        for width in sub_band_list:
            end_idx = start_idx + width
            self.sub_band_encoders.append(
                SubBandEncoder(start_band_idx=start_idx, end_band_idx=end_idx, encode_dim=encode_dim)
            )

            start_idx = end_idx

    def forward(self, inputs: Tensor):
        """
        :param inputs: (B, 1, T, F)
        :return: (B, N, T, K)
        """
        encode_out = []
        for sub_encode_layer in self.sub_band_encoders:
            encode_out.append(
                sub_encode_layer(inputs)  # (B, 1, T, N)
            )

        encode_out = torch.cat(encode_out, dim=1)  # (B, K, T, N)
        encode_out = torch.permute(encode_out, dims=(0, 3, 2, 1)).contiguous()  # (B, N, T, K)

        return encode_out


class BandMerge(nn.Module):
    def __init__(self, sub_band_list: list, encode_dim: int):
        """
        merge sub band into full band
        :param sub_band_list:
        :param encode_dim:
        """
        super().__init__()

        self.sub_band_decoders = nn.ModuleList()
        for idx, width in enumerate(sub_band_list):
            self.sub_band_decoders.append(
                SubBandDecoder(sub_band_width=width, sub_band_idx=idx, encode_dim=encode_dim)
            )

    def forward(self, inputs):
        """
        :param inputs: (B, K, T, N)
        :return: (B, T, F, 2)
        """
        out_decode = []
        for mlp in self.sub_band_decoders:
            sub_out = mlp(inputs)  # (B, T, sub_band*2)
            B, T, sub_band = sub_out.shape
            sub_out = torch.reshape(sub_out, shape=(B, T, sub_band // 2, 2))  # (B, T, sub_band, 2)
            out_decode.append(sub_out)

        out_decode = torch.cat(tensors=out_decode, dim=2)  # (B, T, F, 2)

        return out_decode


class BandSplitRNN(nn.Module):
    def __init__(self, sub_band_list: list, encode_dim: int, num_sequence_module: int):
        super().__init__()

        self.band_split = BandSplit(sub_band_list=sub_band_list, encode_dim=encode_dim)

        # enhance layers
        self.enhance_layers = nn.ModuleList()
        for _ in range(num_sequence_module):
            self.enhance_layers.append(
                AxialSelfAttention4(sub_bands=len(sub_band_list), encoded_dim=encode_dim)
            )

        # decoder
        self.mask_decoder = BandMerge(sub_band_list=sub_band_list, encode_dim=encode_dim)

    def forward(self, real, imag):
        """
        :param real: (B, T, F, 1)
        :param imag: (B, T, F, 1)
        :return:
        """
        inputs = torch.cat([real, imag], dim=3)  # (B, T, F, 2)
        inputs = self.band_split(inputs)  # (B, N, T, K)

        for layer in self.enhance_layers:
            inputs = layer(inputs)

        # (B, N, T, K)
        inputs = torch.transpose(inputs, dim0=1, dim1=3).contiguous()  # (B, K ,T, N)
        mask = self.mask_decoder(inputs)  # (B, T, F, 2)

        # decode
        real_mask = mask[:, :, :, 0]  # (B, T, F)
        imag_mask = mask[:, :, :, 1]  # (B, T, F)

        real = real.squeeze(dim=3) * real_mask
        imag = imag.squeeze(dim=3) * imag_mask

        return real, imag


if __name__ == "__main__":
    sub_bands = [8, 8, 8, 8, 8, 8,
                 16, 16, 16, 16, 16,
                 32, 32, 32, 33]
    test_model = BandSplitRNN(sub_band_list=sub_bands, encode_dim=36, num_sequence_module=4)

    test_real = torch.rand(1, 62, 257, 1)
    test_imag = torch.rand(1, 62, 257, 1)

    test_out = test_model(test_real, test_imag)

    macs, params = profile(test_model, inputs=(test_real, test_imag))
    print(f"mac: {macs / 1e9} G \nparams: {params / 1e6}M")
