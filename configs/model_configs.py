import datetime
import os
from dataclasses import dataclass


@dataclass(frozen=False)
class basic_config:
    fs: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    n_frames: int = 300
    encode_dim: int = 36
    num_sequence_module: int = 4
    sub_bands = [8, 8, 8, 8, 8, 8,
                 16, 16, 16, 16, 16,
                 32, 32, 32, 33]


@dataclass(frozen=False)
class train_config(basic_config):
    train_snr: tuple = (-5, 10)
    sample_points: int = (basic_config.n_frames - 1) * basic_config.hop_length
    # fit stft and i-stft
    epoch: int = 300
    batch_size: int = 50

    pre_data_pkl_path: str = r"/data3/wukeyi/code/streamLSA/dataloader/pre_data_all_no_speech_in_noise"
    if not os.path.exists(pre_data_pkl_path):
        os.makedirs(pre_data_pkl_path)

    clean_path = [
        r"/data3/wukeyi/dataset/aishell2/wav_data",
        r"/data3/wukeyi/dataset/clean_data/aishell",
        r"/data3/wukeyi/dataset/vctk/wav48_silence_trimmed"
    ]
    noisy_path = [
        r"/data3/wukeyi/dataset/remove_client_noise2",
        # r'/data3/wukeyi/dataset/nosie_dataset_no_speech/remove_client_noise_esc_50',
        # r'/data3/wukeyi/dataset/nosie_dataset_no_speech/FSD50K',
        # r'/data3/wukeyi/dataset/nosie_dataset_no_speech/{noise_fullband}'
    ]

    rir_path = r"/data3/wukeyi/dataset/rirs_noises/RIRS_NOISES"

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.abspath(os.path.join(current_file_dir, ".."))
    current_time = str(datetime.datetime.now())
    # checkpoint_path: str = os.path.join(work_dir, "checkpoint", current_time[:19])
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    #
    # train_logger: logging.Logger = get_logger(checkpoint_path)  # using for training logging

    del work_dir, current_file_dir, current_time


@dataclass(frozen=False)
class test_config(basic_config):
    test_snr: tuple = (0, 5)
    sample_points: int = (basic_config.n_frames - 1) * basic_config.hop_length + basic_config.n_fft
    # fit stft and i-stft
    pre_data_pkl_path: str = r"/data3/wukeyi/code/streamLSA/dataloader/pre_data_all_3"
    clean_path = [r"/data3/wukeyi/dataset/clean_data/aishell2/S0003"]
    noisy_path = ["/data3/wukeyi/dataset/remove_client_noise"]

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.abspath(os.path.join(current_file_dir, ".."))

    test_model_path: str = r"/data3/wukeyi/code/streamLSA/checkpoint/coarse_lsa/best_0.1395.pth.rar"
    save_test_path: str = os.path.join(work_dir, "test_out")
    save_test_single_path: str = os.path.join(work_dir, "test_single_out")
    # if not os.path.exists(save_test_path):
    #     os.makedirs(save_test_path)
    #
    # if not os.path.exists(save_test_single_path):
    #     os.makedirs(save_test_single_path)

    del work_dir, current_file_dir


if __name__ == "__main__":
    configs = train_config()
    # configs.train_logger.info(configs.__dict__)
