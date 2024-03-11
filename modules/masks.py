# !/user/bin/env python
# -*-coding:utf-8 -*-

"""
# File : masks.py
# Time : 2023/8/28 上午10:06
# Author : wukeyi
# version : python3.11
"""
import torch


class casual_mask:
    """
    casual mask for transformer decoder or causal sequence processing.

    >>attention_score = torch.matmul(qurry, key)
    >>attention_score = attention_score + attention_mask
    >>attention_score = torch.softmax(attention_score)
    >>out = torch.matmul(attention_score, value)
    """
    def __init__(self):
        self.mask_dict = {}

    def __call__(self, shape: tuple):
        """
        :param shape: reuse attention_mask if shape unchanged.
        :return:
        """

        matrix = torch.ones(shape, dtype=torch.bool)
        mask = torch.tril(matrix)
        unmask_value = torch.zeros(shape)
        mask_value = -torch.ones(shape) * 1e9
        attention_mask = torch.where(mask, unmask_value, mask_value)

        self.mask_dict.update({shape: attention_mask})

        return attention_mask


class casual_local_mask:
    """
    casual mask, moreover, only compute attention score adjustment.
    >>attention_score = torch.matmul(qurry, key)
    >>attention_score = attention_score + attention_mask
    >>attention_score = torch.softmax(attention_score)
    >>out = torch.matmul(attention_score, value)
    """
    def __init__(self):
        self.mask_dict = {}

    def __call__(self, shape: tuple, chunk_size: int):
        """
    :param shape: reuse attention_mask if shape unchanged.
    :param chunk_size: The number of adjacent frames/words used to calculate attention
    :return:
        """
        matrix = torch.ones(shape, dtype=torch.bool)
        mask1 = torch.tril(matrix)
        mask2 = torch.triu(matrix, diagonal=1 - chunk_size)
        mask = torch.logical_and(mask1, mask2)

        unmask_value = torch.zeros(shape)
        mask_value = -torch.ones(shape) * 1e9

        attention_mask = torch.where(mask, unmask_value, mask_value)

        self.mask_dict.update({shape: attention_mask})

        return attention_mask


class local_mask:
    """
    local mask, only compute attention score adjustment.
     paper: <LOCAL SPECTRAL ATTENTION FOR FULL-BAND SPEECH ENHANCEMENT>
    usually for frequency attention
    >>attention_score = torch.matmul(qurry, key)
    >>attention_score = attention_score + attention_mask
    >>attention_score = torch.softmax(attention_score)
    >>out = torch.matmul(attention_score, value)
    """
    def __init__(self):
        self.mask_dict = {}

    def __call__(self, shape: tuple, chunk_size: int):
        """
    :param shape: reuse attention_mask if shape unchanged.
    :param chunk_size: The number of adjacent frames/words used to calculate attention
    :return:
        """

        matrix = torch.ones(shape, dtype=torch.bool)

        mask2 = torch.triu(matrix, diagonal=(1 - chunk_size) // 2)
        mask1 = torch.tril(matrix, diagonal=chunk_size // 2)
        mask = torch.logical_and(mask1, mask2)

        unmask_value = torch.zeros(shape)
        mask_value = -torch.ones(shape) * 1e9

        attention_mask = torch.where(mask, unmask_value, mask_value)

        self.mask_dict.update({shape: attention_mask})

        return attention_mask


if __name__ == "__main__":
    get_masks = casual_local_mask()
    mask_out = get_masks(shape=(1, 5, 5), chunk_size=3)
    out = torch.softmax(mask_out, dim=-1)
