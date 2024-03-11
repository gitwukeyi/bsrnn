import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import wave
import multiprocessing


def silent_part_remove(file: str,
                       out_folder: str = r'/data3/wukeyi/dataset/remove_client_noise2',
                       sample_rate: int = 16000):
    """
    用于训练vad或者SED网络的数据中，很多音频存在静默部分，需要将静部分去除在可以进行
    训练数据制作，本函数将处理文件夹里所有wav, mp3格式音频，去除其静默片段，并转换为
    单通道的wav格式音频。
    :param file: 文件路径
    :param out_folder: 输出文件夹
    :param sample_rate: 采样率
    :return:
    """
    out_file_path = file.split(os.sep)[-2:]
    out_file_path = os.sep.join(out_file_path)
    out_file_path = os.path.join(out_folder, out_file_path)
    out_file_folder = os.sep.join(out_file_path.split(os.sep)[:-1])
    if not os.path.exists(out_file_folder):
        os.makedirs(out_file_folder)

    audio_data, sr = sf.read(file, dtype="float32")

    if len(audio_data) < sample_rate:
        return

    if sample_rate != sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)  # 重采样

    intervals = librosa.effects.split(y=audio_data, top_db=50, frame_length=1024, hop_length=256)
    # intervals.shape = (m, 2)
    with wave.open(out_file_path, mode="w") as f:
        f.setframerate(sample_rate)
        f.setsampwidth(2)
        f.setnchannels(1)
        # 一个音频不同分段写入同一个文件
        for start, end in intervals:
            segment = (audio_data[start:end] * 32767).astype(np.int16)
            f.writeframesraw(segment.tobytes())


if __name__ == "__main__":
    # audio_folder = r"/AI_audio/ESC-50-master/audio"
    audio_folder = r"/data3/wukeyi/dataset/nosie_dataset_no_speech"
    audio_files = librosa.util.find_files(audio_folder, ext=["wav", "WAV", "mp3", "MP3"], recurse=True)

    with multiprocessing.Pool() as p:
        list(tqdm(
            p.imap(silent_part_remove, iterable=audio_files), total=len(audio_files))
        )
