import stft
import wave_func as wf
import numpy as np
import scipy
import sounddevice as sd
from tqdm import tqdm
import matplotlib.pyplot as plt


Fs = 16000
file_name = 'static/input_audio.wav'
start = 0 # [s]
end = 0 # [s] (if end==0 =&amp;amp;amp;gt; full size)
Lf = 2**9 # length of frame(window)
noverlap = 2**7
s = wf.read_wave(file_name, start, end)

# S = stft.STFT(s, Lf, noverlap)
# print(S.shape)
# dereverbed = stft.ISTFT(S, Lf, noverlap)
# # 配列を再生
# sd.play(dereverbed, Fs)
# sd.wait()  # 再生が完了するまで待機
# plt.plot(dereverbed)
# plt.show()

# stft.plot_spectrogram(Fs, dereverbed,  Lf, noverlap)


import scipy.signal as sp
f, t, stft_data = sp.stft(s, fs = 16000, window = "hann", nperseg=512, noverlap = 128)

print(np.shape(stft_data))


t, data_post = sp.istft(stft_data, fs = 16000, window="hann", nperseg = 512, noverlap = 128)
print(t)
print(len(data_post))

sd.play(data_post, Fs)
sd.wait()  # 再生が完了するまで待機
plt.plot(data_post)
plt.show()


