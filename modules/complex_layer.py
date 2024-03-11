from typing import Union

from torch import nn, Tensor


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernels: Union[int, tuple] = 1,
                 stride: Union[int, tuple] = 1,
                 padding: Union[str, int, tuple] = (1, 0),
                 dilation: Union[int, tuple] = 1,
                 groups: int = 1,
                 bias: bool = False
                 ):
        super(ComplexConv2d, self).__init__()

        self.conv2d_r = nn.Conv2d(in_channels, out_channels, kernels, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)

        self.conv2d_i = nn.Conv2d(in_channels, out_channels, kernels, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)

    def forward(self, real: Tensor, imag: Tensor):
        """
        :param real: (B, *, *, *) we recommend using real value, not complex style, in deployment stage,
        complex value are most not supported by embedding machine.
        :param imag: (B, *, *, *)
        :return:
        """
        out_real = self.conv2d_r(real) - self.conv2d_i(imag)
        out_imag = self.conv2d_r(imag) + self.conv2d_i(real)

        return out_real, out_imag
