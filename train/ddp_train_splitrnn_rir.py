# !/user/bin/env python
# -*-coding:utf-8 -*-

"""
# File : train_ddp.py.py
# Time : 2023/11/15 上午10:04
# Author : wukeyi
# version : python3.9
"""
import os
import sys

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as Ddp
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.model_configs import train_config
from dataloader.ve_generator_dataset_v3 import build_loader as g_build_loader
from models.bsrnn import BandSplitRNN


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


class DdpTrainer:
    def __init__(self, model: Module, gpu_id: int, configs: train_config):
        self.configs = configs
        self.gpu_id = gpu_id
        self.epoch = configs.epoch
        self.window = torch.hann_window(configs.n_fft).to(self.gpu_id)

        self.optimizer = NoamOpt(model_size=32, factor=1., warmup=6000,
                                 optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        self.model = Ddp(model.to(gpu_id), device_ids=[gpu_id])  # model should move to gpu first
        if gpu_id == 0:
            self.log_writer = SummaryWriter(log_dir=r"/data3/wukeyi/code/attention_lstm_se/logs")

    def model_forward(self, data: Tensor):
        """
        :param data: (B, N)
        :return: (B, N)
        """
        noisy = data
        noisy = noisy.to(self.gpu_id)
        noisy_stft = torch.stft(noisy, self.configs.n_fft, self.configs.hop_length, win_length=self.configs.n_fft,
                                center=True, window=self.window,
                                return_complex=True)
        noisy_stft = torch.unsqueeze(noisy_stft, dim=3)
        noisy_stft = torch.transpose(noisy_stft, dim0=1, dim1=2)
        real = torch.real(noisy_stft)
        imag = torch.imag(noisy_stft)  # (B, T, F, 1)
        real, imag = self.model(real, imag)

        coarse_out = torch.complex(real, imag)
        coarse_out = torch.transpose(coarse_out, 1, 2)

        coarse_out = torch.istft(coarse_out, n_fft=self.configs.n_fft, hop_length=self.configs.hop_length,
                                 center=True, window=self.window, return_complex=False, onesided=True
                                 )

        return coarse_out

    def loss(self, y_pred: Tensor, y_true: Tensor):
        """
        In speech enhancement task, loss function would be changed frequently, thus, loss function usually defined
        in training scope.
        :param y_pred: (B, N)
        :param y_true: (B, N)
        :return:
        """
        pred_stft = torch.stft(y_pred, self.configs.n_fft, self.configs.hop_length, win_length=self.configs.n_fft,
                               center=True, window=self.window, return_complex=True)
        true_stft = torch.stft(y_true, self.configs.n_fft, self.configs.hop_length, win_length=self.configs.n_fft,
                               center=True, window=self.window, return_complex=True)

        pred_stft_real, pred_stft_imag = torch.real(pred_stft), torch.imag(pred_stft)
        true_stft_real, true_stft_imag = torch.real(true_stft), torch.imag(true_stft)

        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-9)

        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-9)

        pred_real = pred_stft_real / (pred_mag ** 0.7)
        pred_imag = pred_stft_imag / (pred_mag ** 0.7)
        true_real = true_stft_real / (true_mag ** 0.7)
        true_imag = true_stft_imag / (true_mag ** 0.7)
        # spectral loss
        real_loss = torch.mean((pred_real - true_real) ** 2)
        imag_loss = torch.mean((pred_imag - true_imag) ** 2)
        mag_loss = torch.mean((pred_mag ** 0.3 - true_mag ** 0.3) ** 2)

        wav_loss = torch.nn.functional.l1_loss(y_pred, y_true)

        total_loss = 0.3 * (real_loss + imag_loss) + 0.7 * mag_loss + 6 * wav_loss
        return total_loss

    def train_step(self, batch_inputs: tuple):
        noisy_audio = batch_inputs[0].to(self.gpu_id)
        clean_audio = batch_inputs[1].to(self.gpu_id)

        coarse_wav = self.model_forward(noisy_audio)
        coarse_loss = self.loss(coarse_wav, clean_audio)

        total_loss = coarse_loss
        # update model
        self.optimizer.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3)
        self.optimizer.step()

        return total_loss.item()

    @torch.no_grad()
    def test_step(self, batch_inputs: tuple):
        noisy_audio = batch_inputs[0].to(self.gpu_id)
        clean_audio = batch_inputs[1].to(self.gpu_id)

        coarse_wav = self.model_forward(noisy_audio)
        coarse_loss = self.loss(coarse_wav, clean_audio)

        total_loss = coarse_loss

        return total_loss.item()

    def run_epoch(self, dl: DataLoader, training: bool = True):
        loop = tqdm(enumerate(dl), total=len(dl), file=sys.stdout)
        total_loss = 0.0

        for idx, batch_data in loop:
            if training is True:
                self.model.train()
                loss = self.train_step(batch_data)

            else:
                self.model.eval()
                loss = self.test_step(batch_data)

            total_loss += loss
            loop.set_postfix({"loss": total_loss / (idx + 1)})

        return total_loss / len(dl)

    def train(self, train_dl: DataLoader, val_dl: DataLoader):

        min_loss = torch.inf

        for idx in range(self.epoch):
            print(f"\n<<<<<< Epoch {idx + 1} / {self.epoch}")

            _ = self.run_epoch(train_dl, training=True)
            val_loss = self.run_epoch(val_dl, training=False)

            if self.gpu_id == 0 and val_loss < min_loss:
                min_loss = val_loss
                path = os.path.join(r"/data3/wukeyi/code/attention_lstm_se/checkpoint",
                                    f"best_{np.round(val_loss, 4)}.pth.rar")
                torch.save(self.model.module.state_dict(), path)
                print("save successfully")
                # self.configs.train_logger.info(f"save model to {path}")
            if self.gpu_id == 0:
                path = os.path.join(r"/data3/wukeyi/code/attention_lstm_se/checkpoint",
                                    f"last_{np.round(val_loss, 4)}.pth.rar")
                torch.save(self.model.module.state_dict(), f=path)  # save last model

                self.log_writer.add_scalar("loss/val", np.round(val_loss, 4))

        print("<<<<< ending training")


def main(rank,
         world_size,
         configs):
    model = BandSplitRNN(sub_band_list=configs.sub_bands, encode_dim=configs.encode_dim, num_sequence_module=2)
    # checkpoint = torch.load(r"/data3/wukeyi/code/streamLSA/checkpoint/refine_lsa/best_0.0578.pth.rar")
    # model.load_state_dict(checkpoint, strict=False)
    init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    trainer = DdpTrainer(model, gpu_id=rank, configs=configs)
    train_loader, val_loader = g_build_loader(
        clean_audio_folders=configs.clean_path,
        noisy_audio_folders=configs.noisy_path,
        rir_path=configs.rir_path,
        val_rate=0.8, sample_points=configs.sample_points, batch_size=configs.batch_size, fs=configs.fs,
        snr=configs.train_snr, multi_gpu=True, load_pre_data=True, pre_data_pkl_path=configs.pre_data_pkl_path
    )
    trainer.train(train_loader, val_loader)
    destroy_process_group()


if __name__ == "__main__":
    config = train_config()

    # _, _ = g_build_loader(
    #     clean_audio_folders=config.clean_path,
    #     noisy_audio_folders=config.noisy_path,
    #     rir_path=config.rir_path,
    #     val_rate=0.8, sample_points=config.sample_points, batch_size=config.batch_size, fs=config.fs,
    #     snr=config.train_snr, multi_gpu=False, load_pre_data=False, pre_data_pkl_path=config.pre_data_pkl_path
    # )

    # # gpu setup
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ['MASTER_PORT'] = '8001'
    os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
    # multi gpu train
    import torch.multiprocessing as mp

    mp.spawn(main, args=(2, config), nprocs=2)
