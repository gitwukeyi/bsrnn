import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from configs.model_configs import test_config, train_config
from models.bsrnn import BandSplitRNN
from processing.audio_format import m4a_to_wav_in_memory

# from torch.utils.tensorboard import SummaryWriter

configs = test_config()
device = torch.device("cpu")
torch.set_default_tensor_type(torch.FloatTensor)

WINDOW = torch.hann_window(configs.n_fft, device=device)


# WINDOW = torch.sqrt(WINDOW+1e-12)


def load_model(model_obj, model_path: str):
    model_dict = torch.load(model_path, map_location="cpu")

    new_dict = {}
    for keys, values in model_dict.items():
        if keys.split(".")[-1] not in ["cf", "ct", "fre_mask", "time_mask"]:
            new_dict[keys] = values

    model_obj.load_state_dict(new_dict, strict=False)

    return model_obj


def main(test_file_path, out_path):
    """model"""

    '''model'''
    sample_points: int = (80 - 1) * configs.hop_length + configs.n_fft

    if test_file_path.split(".")[-1] in ["m4a", "M4A"]:
        inputs = m4a_to_wav_in_memory(test_file_path)
    else:
        inputs = librosa.load(test_file_path, sr=configs.fs)[0]

    # inputs = inputs/np.max(abs(inputs))
    # inputs = inputs[16000*60: 16000*120]
    inputs = np.append(inputs, np.zeros(sample_points))
    audio_len = len(inputs)
    inputs = torch.tensor(inputs).float()
    ola_len = configs.n_fft - configs.hop_length
    hop_length = sample_points

    out_result = np.zeros(audio_len)
    start_time = time.time()
    for start in tqdm(range(0, audio_len - sample_points, hop_length)):
        noisy = inputs[start: start+sample_points]
        noisy = noisy[None, :]

        with torch.no_grad():
            noisy = noisy.to(device)
            noisy_stft = torch.stft(noisy, configs.n_fft, configs.hop_length, win_length=configs.n_fft,
                                    center=True, window=WINDOW,
                                    return_complex=True)
            noisy_stft = torch.unsqueeze(noisy_stft, dim=3)
            noisy_stft = torch.transpose(noisy_stft, dim0=1, dim1=2)
            real = torch.real(noisy_stft)
            imag = torch.imag(noisy_stft)  # (B, T, F, 1)
            real, imag = model(real, imag)

            coarse_out = torch.complex(real, imag)
            coarse_out = torch.transpose(coarse_out, 1, 2)
            coarse_out = torch.istft(coarse_out, n_fft=configs.n_fft, hop_length=configs.hop_length,
                                     center=True, return_complex=False, onesided=True,
                                     window=WINDOW
                                     )

            enh_s = coarse_out[0, :].cpu().detach().numpy()
            # result = np.append(result, enh_s)
            out_result[start:start+sample_points] += enh_s
    end_time = time.time()
    print(f"average time {(end_time-start_time) / (audio_len/16000)}")
    out_file_path = test_file_path.split(os.sep)[-2:]
    out_file_path = os.sep.join(out_file_path)
    out_file_path = os.path.join(out_path, out_file_path)
    out_file_path = f"{'.'.join(out_file_path.split('.')[:-1])}.wav"
    out_file_folder = os.sep.join(out_file_path.split(os.sep)[:-1])
    if not os.path.exists(out_file_folder):
        os.makedirs(out_file_folder)
    sf.write(out_file_path, out_result[:-sample_points], samplerate=configs.fs)


if __name__ == "__main__":
    # exclude
    configs = train_config()
    model = BandSplitRNN(sub_band_list=configs.sub_bands, encode_dim=configs.encode_dim,
                         num_sequence_module=configs.num_sequence_module)
    model = load_model(model, model_path=r"/data3/wukeyi/code/attention_lstm_se/checkpoint/best_0.0538.pth.rar")
    model = model.to(device)

    model.eval()

    root_path = r"/data3/wukeyi/code/streamLSA/test_files/noisy-scrum.wav"
    out_path = r"/data3/wukeyi/dataset/SE_test_out32"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    paths = librosa.util.find_files(root_path, ext=["wav", "WAV"], recurse=True)
    if len(paths) < 1:
        paths = [root_path]
    for path in paths:
        main(path, out_path)
