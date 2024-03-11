import io
import os
import subprocess

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm


def mp3_to_wav_in_memory(file_path: str = None) -> np.ndarray:
    """
    读取MP3并转换为wav, 转换过程不产生中间文件
    Parameters
    ----------
    file_path

    Returns
    -------
    """
    wav_buffer = io.BytesIO()  # 申请内存，转换后的wav音频数据直接存储到内存
    audio_mp3 = AudioSegment.from_mp3(file_path)
    audio_mp3.export(wav_buffer, format="wav")
    audio_new = AudioSegment.from_file(wav_buffer, format="wav")  # 从内存中读取wav音频数据
    audio_data = audio_new.get_array_of_samples()
    wav_buffer.close()

    return np.divide(audio_data, 2 ** 15)


def m4a_to_wav_in_memory(file_path: str = None) -> np.ndarray:
    """
    读取m4a并转换为wav, 转换过程不产生中间文件
    Parameters
    ----------
    file_path

    Returns
    -------
    """
    wav_buffer = io.BytesIO()  # 申请内存，转换后的wav音频数据直接存储到内存
    audio_mp3 = AudioSegment.from_file(file_path, format="m4a")
    audio_mp3.export(wav_buffer, format="wav")
    audio_new = AudioSegment.from_file(wav_buffer, format="wav")  # 从内存中读取wav音频数据
    audio_new = audio_new.set_frame_rate(16000)
    audio_new = audio_new.set_sample_width(2)
    audio_new = audio_new.set_channels(1)
    audio_data = audio_new.get_array_of_samples()
    wav_buffer.close()

    return np.divide(audio_data, 2 ** 15)


def all_file_to_wav(in_path: str, output_path: str, sample_rate: int = 16000) -> None:
    """
    convert all files into wav format
    Parameters
    ----------
    in_path
    output_path
    sample_rate

    Returns
    -------

    """

    def to_wav(mp3_path: list) -> None:
        """
        convert all format audio into wav; using ffmpeg
        :param mp3_path:
        :return:
        """
        sep = os.path.sep  # path sep, in window sep is "\\" or "/", but only the style of "/" in linux
        for file in tqdm(mp3_path):
            file_folder_name = sep.join(file.split(sep)[-3:-1])
            print(file_folder_name)
            out_file_folder = os.path.join(output_path, file_folder_name)
            if not os.path.exists(out_file_folder):
                os.makedirs(out_file_folder)

            out_file_name = file.split(sep)[-1].split(".")[0] + '.wav'
            out_file_path = os.path.join(out_file_folder, out_file_name)
            print(out_file_path)
            subprocess.Popen([
                'ffmpeg', '-i', file, '-ar', str(sample_rate), '-ac', '1', '-y',
                out_file_path
            ])

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_list = []
    for root, dirs, files in os.walk(in_path):
        for f in files:
            if f.split(".")[-1] in ["mp3", "MP3", "flac"]:
                file_path = os.path.join(root, f)
                file_list.append(file_path)

    to_wav(file_list)


if __name__ == "__main__":
    test_path = r"/data3/wukeyi/dataset/clean_data/train-other-500/LibriSpeech"
    out_path = r"/data3/wukeyi/dataset/clean_data/librispeech"
    all_file_to_wav(test_path, output_path=out_path, sample_rate=16000)
    # test_paths = r"/data3/wukeyi/code/streamLSA/test_files/road.m4a"
    # out_wav = m4a_to_wav_in_memory(test_paths)
    #
    # # plot
    # import matplotlib.pyplot as plt
    # import librosa
    # import soundfile as sf
    # origin_data = librosa.load(test_paths, sr=16000)[0]
    # fig = plt.figure(figsize=(20, 5), dpi=100)
    # ax = fig.add_subplot(1, 2, 1)
    # ax.set_title("m4a")
    # ax.specgram(origin_data, Fs=16000)
    #
    # ax = fig.add_subplot(1, 2, 2)
    # ax.specgram(out_wav, Fs=16000)
    # ax.set_title(f"wav")
    # plt.savefig("to_wav.png")
    # sf.write("to_wav.wav", out_wav, samplerate=16000)
    # plt.show()

