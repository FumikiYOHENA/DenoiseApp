# import stft
import wave_func as wf
import numpy as np
import scipy
import sounddevice as sd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.signal as sp


Fs = 16000
file_name = 'static/dereverb.wav'
start = 0 # [s]
end = 0 # [s] (if end==0 =&amp;amp;amp;gt; full size)
Lf = 2**9 # length of frame(window)
noverlap = 2**7
s = wf.read_wave(file_name, start, end)

f, t, S = sp.stft(s, fs = 16000, window = "hann", nperseg=Lf, noverlap = noverlap)
print(S.shape)


def dereverb(x):
    Fq, T = x.shape
    p = 40
    D = 2
    z = np.zeros((Fq, T), dtype=complex)  # 複素数型に変更
    G = np.zeros((p, Fq), dtype=complex)   # 複素数型に変更

    for f in tqdm(range(Fq)):
        x_hat = np.zeros((p, T, Fq), dtype=complex)  # 複素数型に変更
        max_l2norm = 0
        for t in range(T):
            l2norm = np.abs(x[f, t]) ** 2
            if l2norm > max_l2norm:
                max_l2norm = l2norm

        epsilon = max_l2norm * 10 ** -5
 

        for t in range(p + D):
            x_hat[:, t, f] = np.zeros(p)

        for t in range(p + D + 1, T):
            for p_1 in range(p):
                x_hat[p_1, t, f] = x[f, t - p_1 - D + 1]

        count = 0
        while count < 10:
            count += 1
            # step 7
            for t in range(T):
                z[f, t] = x[f, t] - np.dot(G[:, f].conj().T, x_hat[:, t, f])

            lambda_vals = np.zeros(T, dtype=complex)  # 複素数型に変更
            for t in range(T):
                lambda_vals[t] = max(np.linalg.norm(z[f, t]) ** 2, epsilon)

            # R = np.zeros((p, p), dtype=complex)  # 複素数型に変更
            # P = np.zeros((p, 1), dtype=complex)  # 複素数型に変更
            R = 0
            P = 0
            for t in range(T):
                R += np.outer(x_hat[:, t, f], x_hat[:, t, f]).conj() / lambda_vals[t]
                P += np.outer(x_hat[:, t, f], x[f, t]).conj() / lambda_vals[t]
            R /= T
            P /= T
            R += 1e-5 * np.trace(R) * np.eye(p, dtype=complex)  # 複素数型に変更
            G[:, f] = np.linalg.inv(R).dot(P[:,0])
    
    return z

result = dereverb(S)

t, dereverbed = sp.istft(result, fs = 16000, window="hann", nperseg = Lf, noverlap = noverlap)
# 配列を再生
sd.play(dereverbed, Fs)
sd.wait()  # 再生が完了するまで待機
plt.plot(dereverbed)
plt.show()
