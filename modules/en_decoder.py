import torch
from torch import nn, Tensor
from modules.complex_layer import ComplexConv2d


class FeatureCompression(nn.Module):
    def __init__(self):
        """
        The value range of magnitude spectrum or real/imag spectrum are too large to highlighted their feature,
        so the value range needed to compressed.
        """
        super().__init__()
        self.complex_conv = ComplexConv2d(in_channels=1, out_channels=1, kernels=1, stride=1, padding=0)

    def forward(self, real: Tensor, imag: Tensor):
        """
        :param real: (B, 1, *, *)
        :param imag: same as real
        :return: (B, 1, *, *)
        """
        real, imag = self.complex_conv(real, imag)

        inputs = torch.sqrt(real ** 2 + imag ** 2 + 1e-12)
        inputs = torch.pow(inputs, 0.5)  # feature dynamic range compression

        return inputs


class FeatureUnCompression(nn.Module):
    def __init__(self, in_channels: int):
        """
        Inverse operation of FeatureCompression, the backbone network decodes compressed feature to obtain
        imaginary and real spectrum again.
        :param in_channels:
        """
        super().__init__()
        self.complex_conv = ComplexConv2d(in_channels=in_channels, out_channels=1, kernels=1, padding=0)

    def forward(self, real: Tensor, imag: Tensor):
        """
        :param real: (B, *, *, *)
        :param imag: (B, *, *, *)
        :return: real/imag spectrum
        """
        real, imag = self.complex_conv(real, imag)

        return real, imag


class SubBandEncoder(nn.Module):
    def __init__(self, start_band_idx: int, end_band_idx: int, encode_dim: int):
        """
        sub band spectra with different widths are encoded to the same width.
        :param start_band_idx:
        :param end_band_idx:
        :param encode_dim: all sub bands are encoded to the same width. The encode dimensions cannot be smaller than
        sub bands width.
        """
        super().__init__()
        fre_width = end_band_idx - start_band_idx
        self.norm = nn.LayerNorm(normalized_shape=fre_width*2)
        self.fc = nn.Linear(in_features=fre_width*2, out_features=encode_dim)

        self.start = start_band_idx
        self.end = end_band_idx

    def forward(self, inputs):
        """
        :param inputs: (B, T, F, 2)
        :return:
        """
        sub_spectra = inputs[:, :, self.start:self.end, :].contiguous()  # (B, T, sub_band_width, 2)
        B, T, _, _ = sub_spectra.shape
        sub_spectra = torch.reshape(sub_spectra, shape=(B, 1, T, -1))
        sub_spectra = self.norm(sub_spectra)
        sub_spectra = self.fc(sub_spectra)  # (B, 1, T, N)

        return sub_spectra


class SubBandDecoder(nn.Module):
    def __init__(self, sub_band_width: int, sub_band_idx: int, encode_dim: int):
        """
        encoded vectors are switched backed to sub band
        :param sub_band_width:
        :param encode_dim: all sub bands are encoded to the same width. The encode dimensions cannot be smaller than
        sub bands width.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(normalized_shape=encode_dim),
            nn.Linear(in_features=encode_dim, out_features=encode_dim * 4),
            nn.Tanh(),
            nn.Linear(in_features=encode_dim * 4, out_features=sub_band_width * 4),
            GLU(dim=-1)
        )

        self.sub_band_idx = sub_band_idx

    def forward(self, inputs):
        """
        :param inputs: (B, K, T, N)
        :return:
        """
        sub_spectra = inputs[:, self.sub_band_idx, :, :].contiguous()  # (B, T, encode_dim)
        sub_spectra = self.mlp(sub_spectra)  # (B, T, sub_band*2)

        return sub_spectra


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Tensor):
        divide_idx = inputs.shape[self.dim] // 2

        return inputs[..., :divide_idx].contiguous() * torch.sigmoid(inputs[..., divide_idx:].contiguous())
