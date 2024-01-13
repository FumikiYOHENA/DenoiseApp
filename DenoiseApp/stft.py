import numpy as np
import matplotlib.pyplot as plt
import wave_func as wf


# STFT
def STFT(s, Lf, noverlap):
    l = s.shape[0]
    win = np.hanning(Lf)
    Mf = Lf//2 + 1
    Nf = int(np.ceil((l-noverlap)/(Lf-noverlap)))-1
    S = np.empty([Mf, Nf], dtype=np.complex128)
    for n in range(Nf):
        S[:,n] = np.fft.rfft(s[(Lf-noverlap)*n:(Lf-noverlap)*n+Lf] * win, n=Lf, axis=0)
    return S

# 逆STFT
def ISTFT(S, Lf, noverlap):
    Mf, Nf = S.shape
    L = (Nf - 1) * (Lf - noverlap) + Lf
    win = np.hanning(Lf)
    s = np.zeros(L, dtype=np.float64)

    for n in range(Nf):
        x = np.fft.irfft(S[:, n], n=Lf, axis=0)
        s[n * (Lf - noverlap):n * (Lf - noverlap) + Lf] += x * win

    return s

# drow spectrogram
def plot_spectrogram(fs, s,  Lf, noverlap=None):
    S = STFT(s, Lf, noverlap)
    S = S + 1e-18
    P = 20 * np.log10(np.abs(S))
    P = P - np.max(P) # normalization
    vmin = -150
    if np.min(P) > vmin:
        vmin = np.min(P)
    m = np.linspace(0, s.shape[0]/fs, num=P.shape[1])
    k = np.linspace(0, fs/2, num=P.shape[0])
    plt.figure()
    plt.pcolormesh(m, k, P, cmap = 'jet', vmin=-150, vmax=0)
    plt.title("Spectrogram of Sound")
    plt.xlabel("time[s]")
    plt.ylabel("frequency[Hz]")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
