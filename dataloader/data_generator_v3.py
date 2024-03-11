import os
import sox
import wave
import joblib
import librosa
import soundfile as sf
import numpy as np

from typing import List, Dict
from scipy import stats, signal
from joblib import Parallel, delayed
from tqdm import tqdm


def file_filter(audio_folder: str):
    """
    过滤短音频文件，并输出文件字典，字典包含每个音频的绝对路径，音频时长，原始采样率
    :param audio_folder: 音频文件夹路径
    :return:
    """
    def _filter(idx: int, path: str):
        audio_obj = wave.open(path, mode="r")
        data_len = audio_obj.getnframes()
        sr = audio_obj.getframerate()
        audio_obj.close()
        if data_len > 16000:
            file_dict = {str(idx): [path, data_len, sr]}
            return file_dict
        else:
            return

    file_path_list = librosa.util.find_files(audio_folder, ext=["wav", "WAV"], recurse=True)  # 选取文件夹下特定文件
    print("\n<<<filter files")
    out_dict = Parallel(n_jobs=-1)(
        delayed(_filter)(idx, path) for idx, path in enumerate(file_path_list)
    )  # multithreading processing
    out_dicts = {}
    for out_d in out_dict:
        if out_d is not None:
            out_dicts.update(out_d)
    print("\n filter file ending!")

    return out_dicts


def padding(audio_data: np.ndarray, points: int):
    """
    填充数据，包括数据复制，填充数据两种方式
    :param audio_data: 被充填的数据
    :param points: 填充后的数据点数量
    :return:
    """
    origin_len = len(audio_data)
    if origin_len >= points:
        return audio_data[:points]

    multiple, remainder = divmod(points, origin_len)
    if multiple > 0:
        audio_data = np.tile(audio_data, multiple)  # (a, b, c) -> (a, b, c, a, b, c, ....)

    audio_data = np.pad(audio_data, pad_width=remainder, mode="wrap")

    return audio_data


class DataSynthesizer:
    def __init__(self,
                 clean_audio_dict: Dict[int, list],
                 noisy_audio_dict: Dict[int, list],
                 rir_paths: List[str],
                 sample_rate: int,
                 sample_points: int,
                 snr: tuple,
                 change_speed: bool,
                 change_gain: bool):
        """
        用于动态合成语音增强所需数据，每次都输入固定点数的合成带噪音频及其对应的干净音频。注意：训练时，测试数据需要固定，
        所以不能在训练时动态混合测试数据，而是在训练前混合一批测试数据。将实现数据的遍历，保证每个数据点都用于训练，例如上一条
        音频剩余长度不满足训练长度，将读取下一条音频。实现中，将利用截断高斯分布产生随机信噪比，默认使用的均值和方差是0和5
        :param clean_audio_dict: 纯净音频文件文件字典,其中一个元素列子{1: ["path1", 12345, 16000], ...}
        :param noisy_audio_dict: 噪声音频文件字典, 其中一个元素列子{1: ["path1", 12345, 16000], ...}
        :param rir_paths: reverberation file paths
        :param sample_rate: 每次迭代生成的音频点数
        :param sample_points: 输出音频长度(点)
        :param snr: 信噪比
        :param change_speed: 是否采用变速增广方法
        :param change_gain: 是否采用音量变换增广方法
        """
        self.sample_points = sample_points
        self.fs = sample_rate
        self.get_snr = stats.truncnorm(snr[0], snr[1], loc=3, scale=10)  # 截尾高斯分布
        self.change_speed = change_speed
        self.change_gain = change_gain

        self.iter_clean_nums = 0
        self.iter_noise_nums = 0
        self.clean_data = np.array([], dtype="float32")  # 记录当前音频，如果长度不够了，就读取下一条音频
        self.noisy_data = np.array([], dtype="float32")  # 同上
        self.clean_audio_dict = clean_audio_dict
        self.noisy_audio_dict = noisy_audio_dict
        self.rir_paths = rir_paths

        self.clean_audio_keys = list(clean_audio_dict.keys())
        self.noisy_audio_keys = list(noisy_audio_dict.keys())

    def __len__(self):
        return len(self.clean_audio_dict)  # 用于for循环停止

    def __iter__(self):
        self.iter_clean_nums = 0  # 用于for 循环
        self.iter_noise_nums = 0
        return self

    @staticmethod
    def add_reverberation(clean_data: np.ndarray, rir_data: np.ndarray):
        peak_origin = np.max(np.abs(clean_data))
        reverb_speech = signal.fftconvolve(clean_data, rir_data, mode="full")
        peak_after = np.max(np.abs(reverb_speech))
        reverb_speech = reverb_speech[0: clean_data.shape[0]]

        return reverb_speech / peak_after * peak_origin

    def speed_shift(self, audio: np.ndarray, speed: float):

        tmf = sox.Transformer()
        tmf.speed(speed)
        audio = tmf.build_array(input_array=audio, sample_rate_in=self.fs)
        tmf.clear_effects()
        del tmf

        return audio

    @staticmethod
    def normalize(waveforms: np.ndarray):
        eps = 1e-14
        den = np.max(np.abs(waveforms), axis=0, keepdims=True) + eps

        waveforms = waveforms / den
        db_coefficient = 10 ** (-30 / 20)   # normalize to -30 dB

        waveforms = waveforms * db_coefficient

        return waveforms

    def gain_shift(self, audio: np.ndarray,
                   min_gain: float = -15,
                   max_gain: float = 2,
                   p: float = 0.8):
        if np.random.rand() > p:
            return audio
        else:
            tmf = sox.Transformer()
            gain = np.round(np.random.uniform(min_gain, max_gain), 2)
            tmf.gain(gain)
            audio = tmf.build_array(input_array=audio, sample_rate_in=self.fs)
            tmf.clear_effects()
            del tmf  # some error would occur unless delete it

            return audio

    def get_data(self):
        """
        获取语音和噪声数据，并根据是否进行变速增广而改变需要采样的数据点数，同时根据实际音频采样率对其变换采样率处理。
        :return:
        """
        # get noisy data
        sample_points = self.sample_points
        if sample_points > len(self.noisy_data):
            while sample_points > len(self.noisy_data):
                # 通过key获取数据，虽然麻烦了点，但是训练需要利用keys打乱字典，其实是为了妥协
                noisy_key = self.noisy_audio_keys[self.iter_noise_nums]
                path, data_len, fc = self.noisy_audio_dict[noisy_key]
                self.iter_noise_nums += 1

                if self.iter_noise_nums > len(self.noisy_audio_dict) - 1:
                    self.iter_noise_nums = 0
                    np.random.shuffle(self.noisy_audio_keys)  # 遍历完噪声文件，需要重新打乱

                noisy_data, _ = sf.read(path, dtype="float32")
                if fc != self.fs:
                    noisy_data = librosa.resample(noisy_data, orig_sr=fc, target_sr=self.fs)

                self.noisy_data = np.append(self.noisy_data, noisy_data)

        noisy_data = self.noisy_data[:sample_points]
        self.noisy_data = self.noisy_data[sample_points:]  # shift data

        # get clean data
        if self.change_speed is True:
            speed = np.random.uniform(0.8, 1.2)
            speed = np.round(speed, 1)
            sample_points = int(np.ceil(self.sample_points * speed))  # up int for speed shift
        else:
            sample_points = self.sample_points
            speed = 1.0

        if sample_points > len(self.clean_data):
            while sample_points > len(self.clean_data):
                clean_key = self.clean_audio_keys[self.iter_clean_nums]
                path, data_len, fc = self.clean_audio_dict[clean_key]
                self.iter_clean_nums += 1

                if self.iter_clean_nums > len(self.clean_audio_dict) - 1:
                    self.iter_clean_nums = 0
                    np.random.shuffle(self.clean_audio_keys)  # 遍历完语音文件，需要打乱，增加随机性。

                clean_data, _ = sf.read(path, dtype="float32")
                if fc != self.fs:
                    clean_data = librosa.resample(clean_data, orig_sr=fc, target_sr=self.fs)

                self.clean_data = np.append(self.clean_data, clean_data)

        clean_data = self.clean_data[:sample_points]
        self.clean_data = self.clean_data[sample_points:]

        if self.change_speed is True:
            clean_data = self.speed_shift(clean_data, speed=speed)

        clean_data = padding(clean_data, self.sample_points)
        noisy_data = padding(noisy_data, self.sample_points)

        return clean_data, noisy_data

    @staticmethod
    def mix_data(clean_data: np.ndarray, noisy_data: np.ndarray, snr: float):
        """
        干净声音与噪声按给定信噪比混合，通过改变干净声音的幅值达到所需信噪比
        :param clean_data:
        :param noisy_data:
        :param snr: 信噪比
        :return:
        """
        p_clean = np.mean(clean_data ** 2)  # 纯净语音功率
        p_noise = np.mean(noisy_data ** 2)  # 噪声功率
        alpha = 10 ** (snr / 10)

        kapa = np.sqrt(p_clean / alpha / (p_noise + np.finfo(np.float32).eps))
        mix_data = clean_data + noisy_data * kapa  # 数据混合

        return mix_data, clean_data

    def __next__(self):
        """
        迭代产生数据， python生成器一直迭代，调用多少次就返回多少次数据，如果不调用则不返回数据
        :return:
        """
        clean_data, noisy_data = self.get_data()

        # if self.change_gain is True:
        #     clean_data = self.normalize(clean_data)
        #     noisy_data = self.normalize(noisy_data)
        # intervals 是非静音间隔
        # clean_data = self.pitch_shift(clean_data)
        current_snr = self.get_snr.rvs(1)[0]

        if np.random.rand() > 0.6:
            choice_file = np.random.choice(self.rir_paths)
            rir_data = librosa.load(choice_file, sr=self.fs)[0]
            clean_data_reverb = self.add_reverberation(clean_data, rir_data=rir_data)
            mix_data, clean_data = self.mix_data(clean_data_reverb, noisy_data, current_snr)  # 混合数据

        else:
            mix_data, clean_data = self.mix_data(clean_data, noisy_data, current_snr)  # 混合数据

        intervals = librosa.effects.split(y=clean_data, top_db=40, frame_length=1024, hop_length=256)

        vad_vector = np.zeros_like(clean_data)
        for start, end in intervals:
            vad_vector[start:end] = 1  # 构造标签

        return mix_data, clean_data, vad_vector


def build_set(clean_audio_folders: List[str],
              noisy_audio_folders: List[str],
              rir_paths: str,
              val_rate: float = 0.2,
              sample_points: int = 15872,
              fs: int = 8000,
              snr: tuple = (-5, 5),
              load_pre_data: bool = False,
              pre_data_pkl_path: str = ""):
    """
    构造tensorflow/pytorch数据读取pipline, 利用tensorflow data加入批量读取, 数据预读取加入数据载入
    :param rir_paths:
    :param pre_data_pkl_path:
    :param load_pre_data: 是否利用预加载数据，将大大节省数据预处理时间
    :param clean_audio_folders: 干净语音音频文件夹
    :param noisy_audio_folders: 噪声音频文件夹
    :param val_rate: 干净语音文件划分训练数据与验证数据比例
    :param sample_points: 每个生成数据的点数
    :param fs: 采样率
    :param snr: 信噪比变化范围
    :return:
    """
    if load_pre_data is True:
        return build_from_pre_data(sample_points, snr, fs, pre_data_pkl_path)

    clean_audio_dict = dict()
    for audio_folder in clean_audio_folders:
        clean_audio_dict.update(file_filter(audio_folder))

    clean_audio_keys = np.array(list(clean_audio_dict.keys()))
    np.random.shuffle(clean_audio_keys)

    # clean拆分数据
    segment_point = int(len(clean_audio_keys) * val_rate)
    train_keys, val_keys = np.split(clean_audio_keys, indices_or_sections=[segment_point])

    train_audio_dict = dict([(key, clean_audio_dict[key]) for key in train_keys])  # 字典切分
    val_audio_dict = dict([(key, clean_audio_dict[key]) for key in val_keys])

    joblib.dump(train_audio_dict,
                filename=os.path.join(pre_data_pkl_path, "train_audio_dict.pkl"))  # dump data for next training
    joblib.dump(val_audio_dict,
                filename=os.path.join(pre_data_pkl_path, "val_audio_dict.pkl"))

    noisy_audio_dict = dict()
    for audio_folder in noisy_audio_folders:
        noisy_audio_dict.update(file_filter(audio_folder))
    # noise拆分数据
    noise_audio_keys = np.array(list(noisy_audio_dict.keys()))
    np.random.shuffle(noise_audio_keys)
    segment_point = int(len(noise_audio_keys) * val_rate)
    train_keys, val_keys = np.split(noise_audio_keys, indices_or_sections=[segment_point])

    train_noise_dict = dict([(key, noisy_audio_dict[key]) for key in train_keys])  # 字典切分
    val_noise_dict = dict([(key, noisy_audio_dict[key]) for key in val_keys])

    joblib.dump(train_noise_dict,
                filename=os.path.join(pre_data_pkl_path, "train_noise_dict.pkl"))
    joblib.dump(val_noise_dict,
                filename=os.path.join(pre_data_pkl_path, "val_noise_dict.pkl"))

    # rir file split
    rir_file_paths = librosa.util.find_files(rir_paths, ext=["WAV", "wav"], recurse=True)
    segment_point = int(len(rir_file_paths) * val_rate)
    train_rir_paths, val_rir_paths = np.split(rir_file_paths, indices_or_sections=[segment_point])

    joblib.dump(train_rir_paths,
                filename=os.path.join(pre_data_pkl_path, "train_rir_paths.pkl"))
    joblib.dump(val_noise_dict,
                filename=os.path.join(pre_data_pkl_path, "val_rir_paths.pkl"))

    # 构造数据pipline
    train_data_generator = DataSynthesizer(train_audio_dict,
                                           train_noise_dict,
                                           train_rir_paths,
                                           fs,
                                           sample_points, snr,
                                           change_speed=True,
                                           change_gain=True)

    val_data_generator = DataSynthesizer(val_audio_dict,
                                         val_noise_dict,
                                         val_rir_paths,
                                         fs,
                                         sample_points, snr,
                                         change_speed=True,
                                         change_gain=True)

    mix_data_paths, clean_data_paths = write_val_data(
        val_data_generator, write_path=pre_data_pkl_path, sample_rate=fs, iter_nums=20000)

    joblib.dump(mix_data_paths,
                filename=os.path.join(pre_data_pkl_path, "mix_data_paths.pkl"))
    joblib.dump(clean_data_paths,
                filename=os.path.join(pre_data_pkl_path, "clean_data_paths.pkl"))

    return train_data_generator, mix_data_paths, clean_data_paths


def build_from_pre_data(sample_points: int, snr, fs: int, pre_data_pkl_path: str):
    """
    从记录中加载处理的文件数据，避免再次预处理消耗时间
    :param sample_points:
    :param snr:
    :param fs:
    :param pre_data_pkl_path: 文件记录所在地址
    :return:
    """
    audio_dict = joblib.load(
        filename=os.path.join(pre_data_pkl_path, "train_audio_dict.pkl"))  # dump data for next training
    noise_dict = joblib.load(
        filename=os.path.join(pre_data_pkl_path, "train_noise_dict.pkl"))

    rir_path = joblib.load(
        filename=os.path.join(pre_data_pkl_path, "train_rir_paths.pkl")
    )

    data_generator = DataSynthesizer(audio_dict,
                                     noise_dict,
                                     rir_path,
                                     fs,
                                     sample_points,
                                     snr,
                                     change_speed=True,
                                     change_gain=True)

    mix_data_paths = joblib.load(
        filename=os.path.join(pre_data_pkl_path, "mix_data_paths.pkl"))  # dump data for next training
    clean_data_paths = joblib.load(
        filename=os.path.join(pre_data_pkl_path, "clean_data_paths.pkl"))

    print("load data successfully!")

    return data_generator, mix_data_paths, clean_data_paths


def write_val_data(val_data_generator, write_path: str, sample_rate: int, iter_nums: int):
    """
    制作验证集/测试集
    :param val_data_generator: 数据生成器
    :param write_path: 写入目录
    :param sample_rate: 输出音频采样率
    :param iter_nums: 数据个数/音频样本数
    :return:
    """
    if not os.path.exists(os.path.join(write_path, "mix_data")):
        os.makedirs(os.path.join(write_path, "mix_data"))

    if not os.path.exists(os.path.join(write_path, "clean_data")):
        os.makedirs(os.path.join(write_path, "clean_data"))

    for idx in tqdm(range(iter_nums)):
        mix_data, clean_data, _ = next(val_data_generator)

        mix_data_path = os.path.join(write_path, "mix_data", f"{idx}.wav")
        clean_data_path = os.path.join(write_path, "clean_data", f"{idx}.wav")
        # 音频数据写入到文件
        sf.write(mix_data_path, mix_data, samplerate=sample_rate)
        sf.write(clean_data_path, clean_data, samplerate=sample_rate)

    mix_data_paths = librosa.util.find_files(
        os.path.join(write_path, "mix_data"), ext=["wav", "WAV"], recurse=True)
    clean_data_paths = librosa.util.find_files(
        os.path.join(write_path, "clean_data"), ext=["wav", "WAV"], recurse=True)

    return mix_data_paths, clean_data_paths
