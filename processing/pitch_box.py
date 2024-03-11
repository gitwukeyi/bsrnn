# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 20:13:21 2022

@author: WKY
"""
import os
import subprocess
import typing
import warnings
from multiprocessing import Pool
from typing import Tuple, Any

import joblib
import librosa
import librosa.display
import matplotlib
import matplotlib.patches as patches
import numpy as np
import matplotlib.pyplot as plt
from librosa import util
from tqdm import tqdm

# plt.rcParams['font.sans-serif'] = ['song']
# plt.rcParams['axes.unicode_minus'] = False
# warnings.filterwarnings("ignore")
# matplotlib.use("Agg")


def pitchTracking(
        y=None,
        sr=22050,
        S=None,
        n_fft=2048,
        hop_length=None,
        f_min=150.0,
        f_max=4000.0,
        threshold=0.1,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        ref=None,
):
    """Pitch tracking on threshold parabolic-ally-interpolated STFT.

    This implementation uses the parabolic interpolation method described by [#]_.

    .. [#] https://ccrma.stanford.edu/~jos/sasp/Sinusoidal_Peak_Interpolation.html

    Parameters
    ----------
    y: np.ndarray [shape=(n,)] or None
        audio signal

    sr : number > 0 [scalar]
        audio sampling rate of ``y``

    S: np.ndarray [shape=(d, t)] or None
        magnitude or power spectrogram

    n_fft : int > 0 [scalar] or None
        number of FFT bins to use, if ``y`` is provided.

    hop_length : int > 0 [scalar] or None
        number of samples to hop

    threshold : float in `(0, 1)`
        A bin in spectrum ``S`` is considered a pitch when it is greater than
        ``threshold * ref(S)``.

        By default, ``ref(S)`` is taken to be ``max(S, axis=0)`` (the maximum value in
        each column).

    f_min : float > 0 [scalar]
        lower frequency cutoff.

    f_max : float > 0 [scalar]
        upper frequency cutoff.

    win_length : int <= n_fft [scalar]
        Each frame of audio is windowed by ``window``.
        The window will be of length `win_length` and then padded
        with zeros to match ``n_fft``.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.windows.hann`
        - a vector or array of length ``n_fft``

        .. see also:: `filters.get_window`

    center : boolean
        - If ``True``, the signal ``y`` is padded so that frame
          ``t`` is centered at ``y[t * hop_length]``.
        - If ``False``, then frame ``t`` begins at ``y[t * hop_length]``

    pad_mode : string
        If ``center=True``, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.

    ref : scalar or callable [default=np.max]
        If scalar, the reference value against which ``S`` is compared for determining
        pitches.

        If callable, the reference value is computed as ``ref(S, axis=0)``.

    Returns
    -------
    pitches, magnitudes : np.ndarray [shape=(d, t)]
        Where ``d`` is the subset of FFT bins within ``fmin`` and ``fmax``.

        ``pitches[f, t]`` contains instantaneous frequency at bin
        ``f``, time ``t``

        ``magnitudes[f, t]`` contains the corresponding magnitudes.

        Both ``pitches`` and ``magnitudes`` take value 0 at bins
        of non-maximal magnitude.
    """

    # Check that we received an audio time series or STFT
    if S is None:
        S = librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )

    # Make sure we're dealing with magnitudes
    S = np.abs(S)

    # Truncate to feasible region
    f_min = np.maximum(f_min, 0)
    f_max = np.minimum(f_max, float(sr) / 2)

    fft_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  # 每个点所在频率 (n_fft, )

    # Do the parabolic interpolation everywhere,
    # then figure out where the peaks are
    # then restrict to the feasible range (fmin:fmax)
    # s[1:-1] is beta, S[:-2] is alpha S[2:] is gamma
    avg = 0.5 * (
            S[2:] - S[:-2]
    )  # (gamma - alpha)/2 , but the formula in reference is (alpha - gamma)/2

    shift = 2 * S[1:-1] - S[2:] - S[:-2]  # (alpha - -2beta + gamma)

    # Suppress divide-by-zeros.
    # Points where shift == 0 will never be selected by local anyway
    # calculate peak center, p=(alpha - gamma)/2 /(alpha - -2beta + gamma)
    shift = avg / (shift + (np.abs(shift) < util.tiny(shift)))

    # Pad back up to the same shape as S
    avg = np.pad(avg, ([1, 1], [0, 0]), mode="constant"
                 )  # padding zero to each item due to quadratic operation
    shift = np.pad(shift, ([1, 1], [0, 0]), mode="constant")

    peak = 0.5 * avg * shift  # 1/4 * (alpha - gamma) * p

    # Pre-allocate output
    pitches = np.zeros_like(S)
    mags = np.zeros_like(S)

    # Clip to the viable frequency range
    freq_mask = ((f_min <= fft_bins) & (fft_bins < f_max)
                 )[:,
                None]  # mask in [0  1]  shape=(n_fft) -> shape=(n_fft, 1)

    # Compute the column-wise local max of S after thresholding
    # Find the argmax coordinates
    if ref is None:
        ref = np.max

    if callable(ref):
        ref_value = threshold * ref(S, axis=0)
    else:
        ref_value = np.abs(ref)

    # find local max point which greater than reference value, then output their index
    idx = np.argwhere(
        freq_mask & util.localmax(S * (S > ref_value)))  # shape=(len(S), 2)

    # Store pitch and magnitude
    pitches[idx[:, 0], idx[:, 1]] = (
            (idx[:, 0] + shift[idx[:, 0], idx[:, 1]]) * float(sr) / n_fft
    )  # k_beta + p

    mags[idx[:, 0],
    idx[:,
    1]] = S[idx[:, 0],
    idx[:, 1]] + peak[idx[:, 0],
    idx[:, 1]]  # beta - 1/4(alpha-gamma）*p

    return pitches, mags


def _cumulative_mean_normalized_difference(y_frames, frame_length, win_length,
                                           min_period, max_period):
    """Cumulative mean normalized difference function (equation 8 in [#]_)

    .. [#] De Scheveningen, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.

    frame_length : int > 0 [scalar]
         length of the frames in samples.

    win_length : int > 0 [scalar]
        length of the window for calculating autocorrection in samples.

    min_period : int > 0 [scalar]
        minimum period.

    max_period : int > 0 [scalar]
        maximum period.

    Returns
    -------
    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
        Cumulative mean normalized difference function for each frame.
    """
    # Autocorrection.
    a = np.fft.rfft(y_frames, frame_length, axis=0)
    b = np.fft.rfft(y_frames[win_length::-1, :], frame_length, axis=0)  # 做翻转
    acf_frames = np.fft.irfft(
        a * b, frame_length,
        axis=0)[win_length:]  # 利用傅里叶变换对计算自相关,并抛弃(pi~2pi)重复的频率的部分
    acf_frames[np.abs(acf_frames) < 1e-6] = 0  # [win_length, frames]

    # Energy terms.
    energy_frames = np.cumsum(y_frames ** 2,
                              axis=0)  # 能量累加  [shape=(frame_length, n_frames)]
    energy_frames = energy_frames[
                    win_length:, :] - energy_frames[:
                                                    -win_length, :]  # 这是是算win_length内的能量累加
    energy_frames[np.abs(energy_frames) < 1e-6] = 0  # [win_length, n_frames]

    # Difference function.
    yin_frames = energy_frames[
                 0, :] + energy_frames - 2 * acf_frames  # [win_length, frames]

    # Cumulative mean normalized difference function.
    yin_numerator = yin_frames[
                    min_period:max_period +
                               1, :]  # [max_period-min_period+1, frames], 只对定义做运算
    tau_range = np.arange(1, max_period + 1)[:, None]
    # [max_period, 1],不是从零开始，因为从零开始就是没有延时，等于计算自身能量，并且会产生干扰，YIN算法都会跳过第一点
    cumulative_mean = np.cumsum(yin_frames[1:max_period + 1, :],
                                axis=0) / tau_range  # 这里从1开始算，与时延对应
    yin_denominator = cumulative_mean[min_period -
                                      1:max_period, :]  # 同样取定义的周期范围
    yin_frames = yin_numerator / (yin_denominator + util.tiny(yin_denominator)
                                  )  # 对应公式的累积均值归一化，不同时延采取不同归一化值
    return yin_frames


def _parabolic_interpolation(y_frames: np.ndarray) -> np.ndarray:
    """Piecewise parabolic interpolation for yin and pyin.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.

    Returns
    -------
    parabolic_shifts : np.ndarray [shape=(frame_length, n_frames)]
        position of the parabola optima
    """
    parabolic_shifts = np.zeros_like(y_frames)
    parabola_a = (y_frames[:-2, :] + y_frames[2:, :] -
                  2 * y_frames[1:-1, :]) / 2  # 二次差分，并对齐数据点（大家点数一样）
    parabola_b = (y_frames[2:, :] - y_frames[:-2, :]) / 2
    parabolic_shifts[1:-1, :] = -parabola_b / (2 * parabola_a +
                                               util.tiny(parabola_a))
    parabolic_shifts[np.abs(parabolic_shifts) > 1] = 0
    return parabolic_shifts


def yin(
        y_frames,
        fmin,
        fmax,
        sr=32000,
        frame_length=1024,
        win_length=None,
        trough_threshold=0.8,
):
    """Fundamental frequency (F0) estimation using the YIN algorithm.

    YIN is an autocorrection based method for fundamental frequency estimation [#]_.
    First, a normalized difference function is computed over short (overlapping) frames of audio.
    Next, the first minimum in the difference function below ``trough_threshold`` is selected as
    an estimate of the signal's period.
    Finally, the estimated period is refined using parabolic interpolation before converting
    into the corresponding frequency.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(n,)]
        audio time series.

    fmin: number > 0 [scalar]
        minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.

    fmax: number > 0 [scalar]
        maximum frequency in Hertz.
        The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
        though higher values may be feasible.

    sr : number > 0 [scalar]
        sampling rate of ``y`` in Hertz.

    frame_length : int > 0 [scalar]
         length of the frames in samples.
         By default, ``frame_length=2048`` corresponds to a timescale of about 93 ms at
         a sampling rate of 22050 Hz.

    win_length : None or int > 0 [scalar]
        length of the window for calculating autocorrection in samples.
        If ``None``, defaults to ``frame_length // 2``

    trough_threshold: number > 0 [scalar]
        absolute threshold for peak estimation.

    Returns
    -------
    f0: np.ndarray [shape=(n_frames,)]
        time series of fundamental frequencies in Hertz.
    """

    if fmin is None or fmax is None:
        raise print('both "fmin" and "fmax" must be provided')

    # Set the default window length if it is not already specified.
    if win_length is None:
        win_length = frame_length // 2

    if win_length >= frame_length:
        raise print("win_length={} cannot exceed given frame_length={}".format(
            win_length, frame_length))

    # Check that audio is valid.
    # util.valid_audio(y, mono=True)

    # Pad the time series so that frames are centered
    # if center:
    #     y = np.pad(y, frame_length // 2, mode=pad_mode)

    # Frame audio.
    # y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = max(int(np.floor(sr / fmax)), 1)  # 信号最小周期所在频率轴的位置（点数）
    max_period = min(int(np.ceil(sr / fmin)),
                     frame_length - win_length - 1)  # 信号最大周期所在频率轴的位置（点数）

    # Calculate cumulative mean normalized difference function.
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period)

    # Parabolic interpolation.
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find local minima.
    is_trough = util.localmin(yin_frames, axis=0)  # 理论上，最小值所在位置就是周期所在位置
    # 因为util.local min函数的缺点，每次都是比较三个值，第一个点默认是最小，这里需要在第一个点与第二个点再判断一次
    is_trough[0, :] = yin_frames[0, :] < yin_frames[1, :]

    # Find minima below peak threshold.
    # 极小值点，并且极小值小于阈值才算是基频点， 如果没有符合就找最小的作为基频点
    is_threshold_trough = np.logical_and(
        is_trough, yin_frames < trough_threshold)  # 一般有多个估值点，并且其中有部分小于阈值

    # Absolute threshold.
    # "The solution we propose is to set an absolute threshold and choose the
    # smallest value of tau that gives a minimum of d' deeper than
    # this threshold. If none is found, the global minimum is chosen instead."
    global_min = np.argmin(yin_frames, axis=0)

    yin_period = np.argmax(is_threshold_trough, axis=0)
    # 求最大值所在的位置，这里is_threshold都是False或True, 找的就是第一个基频点
    no_trough_below_threshold = np.all(~is_threshold_trough,
                                       axis=0)  # 寻找没有小于阈值的极小点的帧
    yin_period[no_trough_below_threshold] = global_min[
        no_trough_below_threshold]

    # 没有小于阈值的极小点的帧的基频点用该帧最小的点代替。

    # Refine peak by parabolic interpolation.
    # 因为上面运算选取了定义的周期[min_period, max_period]运算，yin_period第一个点就是最小周期，
    # 但是它的位置不对应，所以最后计算还要加上min_period, 抛物线插值是为了使得曲线更光滑, parabolic_shifts是二次插值微调的值，
    # 利用二次插值可以将精度提高到0.1%， 相当于补100万个零
    yin_period = (
            min_period + yin_period +
            parabolic_shifts[yin_period, range(yin_frames.shape[1])])

    # Convert period to fundamental frequency.
    # yin_period[no_trough_below_threshold] = 0
    ff = np.repeat(np.arange(is_trough.shape[0])[:, None],
                   is_trough.shape[1],
                   axis=1)
    ff[~is_trough] = 0

    f0 = sr / yin_period
    multiF0 = sr / (min_period + ff)
    multiF0 = multiF0[is_trough]
    IS_f0 = np.any(is_threshold_trough, axis=0)
    # f0[~IS_f0] = 0

    return f0, multiF0, IS_f0


def findPitch(S: np.ndarray,
              sr: int = 32000,
              ts_mag: float = 0.3) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    获取每帧音高，即基频，这里应该包括基频和各次谐波，最小的为基频（一次谐波），其他的依次为二次、三次...谐波
    各次谐波等于基频的对应倍数，因此基频也等于各次谐波除以对应的次数，精确些等于所有谐波之和除以谐波次数之和
    :param S:  wave spectrum data vector
    :param sr: sample rate
    :param ts_mag: threshold
    :return: 每帧基频及其对应峰的幅值(>0)，
             np.ndarray[shape=(1 + n_fft/2，n_frames), dtype=float32]，（257，全部采样点数/(512*2/3)+1）
    """
    # pitches:shape=(d,t)  magnitudes:shape=(d.t), Where d is the subset of FFT bins within fmin and fmax.
    # pitches[f,t] contains instantaneous frequency at bin f, time t
    # magnitudes[f,t] contains the corresponding magnitudes.
    # pitches和magnitudes大于maximal magnitude时认为是一个pitch，否则取0，maximal默认取threshold*ref(S)=1*mean(S, axis=0)
    pitches, magnitudes = librosa.piptrack(S=S,
                                           sr=sr,
                                           threshold=1.0,
                                           ref=np.mean,
                                           fmin=200,
                                           fmax=4000)
    ts = np.average(magnitudes[np.nonzero(magnitudes)]) * ts_mag
    pit_likely = pitches
    mag_likely = magnitudes
    # pit_likely[magnitudes < ts] = 0
    # mag_likely[magnitudes < ts] = 0
    return pit_likely, mag_likely


def findEnergyPeak(data: np.ndarray,
                   segment_frames: int = 4,
                   n_fft: int = 2048) -> np.ndarray:
    """通过能量初步捕获音节，包括能量低的音节，达到100%的检测率，虚警率高是必然的
    再利用其他步骤筛选, 所以不能用到全局阈值，只能使用局域阈值.

    :param data : np.ndarray [shape=(frame_length, n_frames)]framed audio time series
    :param segment_frames : the numbers of frames in a segment
    :param n_fft : fft size

    :return: f0 : fundamental frequency vector
    """

    S = librosa.stft(data, n_fft=n_fft, hop_length=64, center=False)
    S = abs(S ** 2)  # (F, T)

    S_cum = np.cumsum(S.max(axis=0))  # (T,)
    # Segmented statistical, Statistical segmented average energy
    segment_sum = S_cum[
                  segment_frames::segment_frames] - S_cum[:-segment_frames:segment_frames]
    # Each block energy threshold value is assigned to each frame within a segment
    segment_sum = np.repeat(segment_sum[:, None], segment_frames,
                            axis=1).reshape(-1)

    segment_mean = segment_sum / segment_frames / (n_fft // 2 + 1)

    if S.shape[-1] % segment_frames > 0:
        m = np.mean(np.max(S[:, -(S.shape[-1] % segment_frames):], axis=0))
        m = np.repeat(m, S.shape[-1] % segment_frames)
        segment_mean = np.append(segment_mean, m)

    fft_bins = librosa.fft_frequencies(sr=16000, n_fft=n_fft)

    # 统计配个分块的平均能量，每个分开的每个帧都公用该块的平均能量阈值
    f0 = S.argmax(axis=0)
    S_max = S.max(axis=0)
    # f0 = np.argsort(S, axis=0)
    f0[S_max < segment_mean] = 0
    f0[S_max < S[S < segment_mean].mean()] = 0
    f0 = fft_bins[f0]

    return f0


def multi_pitch_find(frames, frame_length, fmin, fmax):
    frenquency = [[200, 300], [300, 400], [400, 500], [500, 800]]
    f0, _, IS = yin(frames, frame_length=frame_length, fmin=200, fmax=1000)
    for i, fre in enumerate(frenquency[1:]):
        segment_f0_value, _, IS_f0 = yin(frames, frame_length=frame_length, fmin=200, fmax=fre[1])
        f0 = (f0 + segment_f0_value) / 2
        IS = np.logical_or(IS, IS_f0)

    return f0, 0, IS


def findEnergyLocalMax(
        data: np.ndarray,
        sr: int = 32000,
        segment_frames: int = 20,
        localMaxNum: int = 3,
        n_fft: int = 2048,
        hop_length: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    find each local maximum value along the frequency axis, In order to have adaptive thresholds for each
    frame signal, we segment long spectrum into little bock, each frame beyond a bock assignment the same
    threshold
    :param localMaxNum: the number of maximum values about energy in spectrum
    :param hop_length: stride size of frame
    :param data: audio data
    :param sr: sample rate
    :param segment_frames: the number of frames in a segment
    :param n_fft: fft size
    :return: f0 , IS_sound
    """
    S = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, center=False)
    frames = librosa.util.frame(data,
                                frame_length=n_fft,
                                hop_length=hop_length)
    assert S.shape[-1] == frames.shape[
        -1], "spectrum shape is not equal to frames"

    ZCR = librosa.feature.zero_crossing_rate(y=data,
                                             frame_length=n_fft,
                                             hop_length=hop_length,
                                             center=False)[0]
    S, _ = librosa.magphase(S)
    L = S.shape[-1]
    F_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # The average energy of each block is calculated, and the average energy threshold of each separate frame
    # is common to a block
    multi_f0 = np.zeros((localMaxNum, L))
    IS_sound = np.empty((L,), dtype=np.bool_)
    IS_flatness = np.empty((L,), dtype=np.bool_)
    IS_ZCR = np.empty((L,), dtype=np.bool_)

    # min_period = max(int(np.floor(sr / 10000)), 1)  # 信号最小周期所在频率轴的位置（点数）
    # max_period = min(int(np.ceil(sr / 200)), n_fft // 2 - 1)  # 信号最大周期所在频率轴的位置（点数）
    IS_f0 = np.empty((L,), dtype=np.bool_)
    wav_f0_value = np.empty((L,))

    segment_list = list(
        zip(range(0, L - segment_frames + 1, segment_frames),
            range(segment_frames, L + 1,
                  segment_frames)))  # segment index of each block
    if L % segment_frames > 0:
        #  if the last segment less than the segment_frames, it should be operated separately
        segment_list.append((L - (L % segment_frames) - 1, L))

    for start, end in segment_list:
        S_segment = S[:, start:end]
        # S_segment[S_segment < np.median(S_segment) * 3] = 0
        pitch, mag = findPitch(S_segment, sr=sr)
        index_top3 = np.argsort(mag, axis=0)[
                     -localMaxNum:, :]  # the top  energy point along frequency axis
        f_min = np.mean(F_bins[index_top3[localMaxNum -
                                          1]])  # min period using for yin
        f_min = np.clip(f_min, 200, 1000)
        y_axis = np.repeat(np.arange(end - start)[None, :],
                           localMaxNum,
                           axis=0)  # match the x-axis coordinate
        multi_f0[:, start:end] = pitch[index_top3, y_axis]

        S_flatness = librosa.feature.spectral_flatness(S=S_segment)
        threshold = np.mean(mag)
        # Only the frame whose maximum energy point is greater than the threshold of evenness energy
        # in the block is considered as a frame with sound
        IS_sound[start:end] = np.max(mag, axis=0) > threshold * 1.5
        # only the frame whose flatness is less than average of flatness within a block is considered as a frame
        # with pitch.
        IS_flatness[start:end] = S_flatness < 0.5

        segment_f0_value, _, IS_f0[start:end] = multi_pitch_find(frames[:, start:end],
                                                                 frame_length=n_fft,
                                                                 fmin=f_min,
                                                                 fmax=4000)
        wav_f0_value[start:end] = segment_f0_value

        IS_ZCR[start:end] = ZCR[start:end] < np.median(ZCR[start:end]) * 0.5

    # f0_swipe = pysptk.sptk.swipe(data.astype(np.float32), fs=sr, hopsize=hop_length, min=200, max=10000, otype="f0")

    level1 = np.logical_and(IS_flatness, IS_f0)
    level2 = np.logical_and(~level1, IS_sound)
    level2 = np.logical_and(level2, IS_ZCR)

    sound_flags = np.logical_or(level1, level2)

    return IS_flatness, sound_flags, wav_f0_value


def adjust_detection_mask(wav_mask: np.ndarray,
                          wav_f0: np.ndarray = None) -> np.ndarray:
    """
    Syllable detection, merging adjacent sounds
    :param wav_f0:
    :param wav_mask:
    :return: mask
    """
    mask = np.zeros_like(wav_mask)

    length = 0
    f0_value_last_syllable = 0
    start_index_last_syllable = 0
    end_index_last_syllable = 0
    last_syllable_flags = False

    for index, element in enumerate(wav_mask):
        if element and length < 1000:

            # 如果相邻音节小于阈值，就将前后两个音节合并
            if index - end_index_last_syllable < 30:
                length = index - end_index_last_syllable
                continue

            length = length + 1

        # 如果属于非音节帧，判断之前积累的音节帧是否大于阈值，符合就标记为音节段
        elif length > 12 or last_syllable_flags:
            f0_value_now_syllable = np.mean(wav_f0[index - length:index])
            if np.abs(f0_value_now_syllable - f0_value_last_syllable) < 800 \
                    and index - length - end_index_last_syllable < 30 and index - start_index_last_syllable < 20:
                length = index - start_index_last_syllable
                f0_value_last_syllable = np.mean(wav_f0[index - length:index])

            mask[index - length:index] = 1
            length = 0
            start_index_last_syllable = index - length
            end_index_last_syllable = index
            last_syllable_flags = False

        elif length > 3:
            f0_value_last_syllable = np.mean(wav_f0[index - length:index])
            length = 0
            end_index_last_syllable = index
            start_index_last_syllable = index - length
            last_syllable_flags = True

        # 如果音节长度小于阈值
        else:
            length = 0
            f0_value_last_syllable = 0
            end_index_last_syllable = index

    mask[end_index_last_syllable:length + end_index_last_syllable] = 1
    mask[-1] = 0
    mask[0] = 0
    return mask


def syllable_detection(wav_path: str,
                       sr: int = 16000,
                       segment_frames: int = 100,
                       n_fft: int = 2048,
                       hop_length: int = 256,
                       Draw_TAG: bool = True) -> Any:
    """
    bird syllable detection rules setting and segment single syllable from long audio record
    :param Draw_TAG: whether to draw picture
    :param segment_frames: the number of frames in a segment
    :param hop_length:
    :param sr: sample rate
    :param wav_path: str audio path
    :param n_fft: int fft size using for one frame
    :return: None
    """

    # file_path = wav_path
    # wav_obj = AudioSegment.from_file(file_path, format=file_path.split(".")[-1])
    # wav_obj = wav_obj.set_channels(1)
    # wav_obj = wav_obj.set_frame_rate(32000)
    # wav_obj = wav_obj.set_sample_width(2)
    #
    # data = np.divide(wav_obj.get_array_of_samples(), 32768)
    # data = librosa.effects.preemphasis(data)
    data = librosa.load(wav_path, sr=sr)[0]
    if librosa.get_duration(y=data, sr=sr) < 1:
        return  # limit audio length
    # data = nr.reduce_noise(data, sr=sr, stationary=True, time_mask_smooth_ms=100, freq_mask_smooth_hz=100)  # 降噪
    # data = librosa.effects.harmonic(data)
    data = librosa.effects.preemphasis(data, coef=0.98)
    # multi_f0 = findEnergyPeak(data, n_fft)
    multi_f0, IS_sound, wav_f0_value = findEnergyLocalMax(
        data,
        sr=sr,
        segment_frames=segment_frames,
        n_fft=n_fft,
        hop_length=hop_length)

    mask = adjust_detection_mask(IS_sound, wav_f0_value)

    # multi_f0 = librosa.yin(
    #     data, fmin=400,
    #     fmax=8000,
    #     sr=32000,
    #     frame_length=1024,
    #     hop_length=128,
    #     trough_threshold=0.8)
    start_point_num = librosa.util.localmax(mask)
    rectangle_point = np.zeros((int(sum(start_point_num)), 4))
    position = np.zeros(
        (int(sum(start_point_num)), 2))  # used to log onset and offset

    syllable_f0 = np.zeros(
        (int(sum(start_point_num)),))  # used to log f0 of syllable

    if np.sum(start_point_num) < 1:
        return

    index = 0
    point_index = 0
    while index < len(mask):
        if mask[index] > 0:
            start = index

            while mask[index] > 0 and index < len(mask):
                index = index + 1

            syllable_f0[point_index] = np.mean(wav_f0_value[start:index])
            # add xy coordinate, width, height to array
            rectangle_point[point_index] = (
                start, np.mean(wav_f0_value[start:index]) // 5, index,
                min(
                    np.mean(wav_f0_value[start:index]) * 4,
                    sr // 2 - wav_f0_value[index] // 5 - 2000))
            position[point_index] = librosa.frames_to_samples(
                (start, index), hop_length=hop_length, n_fft=n_fft)
            point_index = point_index + 1

        index = index + 1

    filename = wav_path.split(os.path.sep)[-1]
    label = np.zeros((int(sum(start_point_num)))).astype(str)

    if not Draw_TAG:
        return {filename: (position, label, wav_path, syllable_f0)}

    # data = librosa.load(wav_path, sr=sr)[0]
    fig = plt.figure(figsize=(50, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    t = ax.specgram(data, NFFT=n_fft, Fs=sr, noverlap=n_fft - hop_length)[2]

    # ax.plot(t, wav_f0_value, lw=1)
    # ax.plot(t, multi_f0 * 8000, lw=1)
    # ax.plot(t, multi_f0[1], lw=0.3)
    # ax.plot(t, multi_f0[2], lw=0.6, color="blue")
    for point_tuple in rectangle_point:
        rect = patches.Rectangle(
            (t[int(point_tuple[0])], point_tuple[1]),
            width=t[int(point_tuple[2])] - t[int(point_tuple[0])],
            height=point_tuple[3],
            linewidth=1.5,
            edgecolor='white',
            fill=False)
        ax.add_patch(rect)
        # ax.text(t[int(point_tuple[0])], point_tuple[1] * 5,
        #         s=str(int(point_tuple[1] * 5)),
        #         fantasize=10,
        #         color='white',
        #         va="center",
        #         bbox={'facecolor': 'red', 'alpha': 0.1, "edgecolor": "white"})
    ax.set_xticks(range(0, int(t[-1])))
    ax.set_xlabel("时间(s)", fontsize=20)
    ax.set_ylabel("频率(HZ)", fontsize=20)
    svg_file_path = "/data3/wukeyi/code/streamLSA/pitch_svg/" + "-".join(
        wav_path.split(".")[0].split(os.path.sep)[-3:]) + ".svg"
    fig.savefig(svg_file_path, dpi=100)
    plt.close(fig)


def mutilProcessDetect(root_path: str,
                       num_workers: int = 4,
                       files_per_class: int = 30) -> None:
    """
    using mutil process to deal with huge amount of audio files
    :param root_path: root path of audio files , variety of classes of birds catalogue should be included
    :param num_workers: the number of processes used to deal with files
    :param files_per_class: Files to be processed in each category
    :return: None
    """
    bird_list = os.listdir(root_path)
    bird_list = [os.path.join(root_path, name) for name in bird_list]

    if os.path.isfile(bird_list[0]):
        files_per_class = len(
            bird_list
        )  # if root path is a single class folder, each file there should be retrieval
        bird_list = [root_path]
    wav_list = list()

    for bird_path in bird_list:
        count = 0
        for root, dirs, files in os.walk(bird_path):
            for f in files:
                if f.split(".")[-1] in ["WAV", "wav", "ogg", "MP3", "mp3"
                                        ] and count < files_per_class:
                    wav_list.append(os.path.join(root, f))
                    count = count + 1

    pool = Pool(num_workers)
    Dict_list = list(
        tqdm(pool.imap(func=syllable_detection, iterable=wav_list),
             total=len(wav_list)))
    pool.close()
    pool.join()

    dict_name = root_path.split(os.path.sep)[-1]
    Dict = dict()
    for element in Dict_list:
        if element:
            Dict.update(element)

    Dict_all = dict()
    Dict_all[dict_name] = Dict

    joblib.dump(Dict_all, f"{dict_name}.pkl")


def toWav(mp3_path: list, output_path: str) -> None:
    """
    convert all format audio into wav; using ffmpeg
    :param output_path:
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
            'ffmpeg', '-i', file, '-ar', '32000', '-ac', '1', '-y',
            out_file_path
        ])


def allFileToWav(in_path: str, output_path: str) -> None:
    """
    convert all files into wav format
    :param output_path:
    :param in_path: root path
    :return:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_list = []
    for root, dirs, files in os.walk(in_path):
        for f in files:
            if f.split(".")[-1] in ["mp3", "MP3"]:
                file_path = os.path.join(root, f)
                file_list.append(file_path)

    toWav(file_list, output_path=output_path)


if __name__ == "__main__":
    # convert file into wav format
    # path = r"D:\WKY\syllable_clustering_byol2\wav_baiyun"
    out_path = r"/data3/wukeyi/code/streamLSA/test_files"
    # allFileToWav(path, output_path=out_path)

    # note: execute this script in command window, instead of pycharm console
    mutilProcessDetect(
        out_path,
        files_per_class=100)  # mutil process must be executed in main function
