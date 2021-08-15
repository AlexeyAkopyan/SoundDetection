from matplotlib import pyplot as plt
import librosa
import numpy as np
import librosa.display as display


def get_centers(seq):
    centers = []
    last_st = -1
    len_sseq = 0
    for i in range(len(seq)):
        if seq[i] == 1:
            len_sseq += 1
            if last_st == -1:
                last_st = i
        else:
            if len_sseq > 0:
                centers.append((last_st + i - 1) // 2)
            len_sseq = 0
            last_st = -1
    if seq[-1] == 1:
        centers.append((last_st + len(seq) - 1) // 2)
    return centers


def correlation(x, y):
    corr = []
    a = x / np.linalg.norm(x)
    x_median = np.quantile(np.abs(x), 0.0)
    len_x = len(x)
    for i in range(len(y) - len_x):
        b = y[i : i + len_x]
        if np.abs(b).max() > x_median:
            b = b / np.linalg.norm(b)
            corr.append(np.sum(a * b))
        else:
            corr.append(0)
    return corr


def cut_pattern(pattern, quant=0.96):
    high_values = np.arange(0, len(pattern))[np.abs(pattern) > np.quantile(np.abs(pattern), quant)]
    return pattern[high_values[0]:high_values[-1]]


def find_patterns(pattern, audio, sr=22050, n_mfcc=20, threshold=0.8, q=0.96):
    if isinstance(pattern, str):
        pattern, _ = librosa.load(pattern, sr=sr)
    if isinstance(audio, str):
        audio, _ = librosa.load(audio, sr=sr)

    pattern = cut_pattern(pattern, quant=q)
    f_pattern = librosa.feature.mfcc(pattern, sr=sr, n_mfcc=n_mfcc)
    f_audio = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    f_len = int(f_pattern.shape[1] * 1.5)

    corr = correlation(f_pattern.T, f_audio.T)
    display.waveplot(audio)
    x_coors = np.linspace(0, audio.shape[0] / sr, len(corr))
    centers = get_centers((np.array(corr) > threshold))

    for cent in centers:
        plt.axvline(x_coors[cent], c='r')
        plt.axvspan(x_coors[max(0, cent - (f_len // 2))], x_coors[min(len(x_coors) - 1, cent + (f_len // 2))],
                    alpha=0.5, color='g')
