#!/usr/bin/env python
# -*- coding: utf-8 -*-
# description 利用生成器产生数据，并利用torch的DataLoader包装，达到多进程数据获取，generator用于数据动态混合
# time: 2023/2/22 10:43
# file: ve_generator_dataset.py
# author: KeYi WU
# email: wukeyi2665623088@gmail.com
from abc import ABC
from typing import Generator, List

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from dataloader.data_generator_v3 import build_set


class VE_Gen_Dataset(Dataset, ABC):
    def __init__(self, data_generator: Generator, iter_len: int = 20000):
        """
        利用生成器在线混合数据，然后用Dataset包装，以便多进程处理
        :param data_generator: 数据生成器，做了while循环无限生成数据
        :param iter_len: 迭代次数，必须设置，不然不知道什么时候停止数据生成。
        """
        super(VE_Gen_Dataset, self).__init__()

        self.data_generator = data_generator
        self.iter_len = iter_len

    def __len__(self):
        return self.iter_len

    def __getitem__(self, item):
        mix_data, clean_data, _ = next(self.data_generator)

        mix_data = mix_data  # 扩展通道维度

        return torch.tensor(mix_data), torch.tensor(clean_data)


class Val_Gen_Dataset(Dataset, ABC):
    def __init__(self, mix_data_paths: str, clean_data_paths: str, iter_len: int):
        """
        利用生成器在线混合数据，然后用Dataset包装，以便多进程处理
        :param mix_data_paths:
        :param clean_data_paths:
        :param iter_len: 迭代次数，必须设置，不然不知道什么时候停止数据生成。
        """
        super(Val_Gen_Dataset, self).__init__()

        self.mix_data_paths = mix_data_paths
        self.clean_data_paths = clean_data_paths
        self.iter_len = iter_len

    def __len__(self):
        return self.iter_len

    def __getitem__(self, item):
        mix_data = sf.read(self.mix_data_paths[item], dtype="float32")[0]
        clean_data = sf.read(self.clean_data_paths[item], dtype="float32")[0]

        return torch.tensor(mix_data), torch.tensor(clean_data)


def build_loader(clean_audio_folders: List[str],
                 noisy_audio_folders: List[str],
                 rir_path: str,
                 val_rate: float,
                 sample_points: int,
                 batch_size: int,
                 fs: int,
                 snr: tuple,
                 multi_gpu: bool = True,
                 load_pre_data: bool = False,
                 pre_data_pkl_path: str = ""):
    """
    构造tensorflow数据读取pipline, 利用tensorflow data加入批量读取, 数据预读取加入数据载入
    :param rir_path:
    :param clean_audio_folders: 干净语音音频文件夹
    :param noisy_audio_folders: 噪声音频文件夹
    :param pre_data_pkl_path: 预计完成过滤的文件数据路径，如果load_pre_data = True, 将从该路径加载数据
    :param load_pre_data: 是否利用预加载数据，将大大节省数据预处理时间
    :param multi_gpu: 如果使用多gpu, dataloader需要采用分布式采样
    :param val_rate: 干净语音文件划分训练数据与验证数据比例
    :param sample_points: 每个生成数据的点数
    :param batch_size: 每个批次读取的数据量
    :param fs: 采样率
    :param snr: 信噪比变化范围
    :return:
    """
    train_generator, mix_data_paths, clean_data_paths = build_set(
        clean_audio_folders=clean_audio_folders,
        noisy_audio_folders=noisy_audio_folders,
        rir_paths=rir_path,
        val_rate=val_rate,
        sample_points=sample_points,
        fs=fs,
        snr=snr,
        load_pre_data=load_pre_data,
        pre_data_pkl_path=pre_data_pkl_path)

    train_dataset = VE_Gen_Dataset(train_generator, iter_len=2000000)
    val_dataset = Val_Gen_Dataset(mix_data_paths, clean_data_paths, iter_len=20000)

    if multi_gpu is True:
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  sampler=DistributedSampler(train_dataset),
                                  num_workers=16, prefetch_factor=2)
        val_loader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                sampler=DistributedSampler(val_dataset),
                                num_workers=16, prefetch_factor=2)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=24, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=24, prefetch_factor=2)

    return train_loader, val_loader


if __name__ == "__main__":
    train, val = build_loader(clean_audio_folders=[r"/data3/wukeyi/dataset/clean_data/aishell2/S0003"],
                              noisy_audio_folders=[r"/data3/wukeyi/dataset/remove_client_noise_esc_50"],
                              rir_path="",
                              val_rate=0.8, sample_points=int(51072), batch_size=12, fs=16000, snr=(-5, 20),
                              multi_gpu=False)

    for mix, target in tqdm(val):
        print(f"mix.shape={mix.shape}")
