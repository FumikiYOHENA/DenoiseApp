# import wave_func as wf
# import scipy
# from scipy.ndimage import maximum_filter1d
# import numpy as np
# import wave
# import sounddevice as sd
# import matplotlib.pyplot as plt
# import scipy.signal as sp

# def envelope(y, rate, threshold):
#     """
#     Args:
#         - y: 信号データ
#         - rate: サンプリング周波数
#         - threshold: 雑音判断するしきい値
#     Returns:
#         - mask: 振幅がしきい値以上か否か
#         - y_mean: Sound Envelop
#     """
#     y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
#     mask = [mean > threshold for mean in y_mean]
#     return mask, y_mean

# wav = wave.open("static/input_audio.wav", "rb")
# frames = wav.readframes(-1)
# noisy_signal = np.frombuffer(frames, dtype="int16")

# mask, y_mean = envelope(noisy_signal, 16000, 5000)
# # filtered = np.array(noisy_signal) * np.array(mask)
# filtered = np.array(noisy_signal) * np.where(mask, 1, 0.1)
# filtered = filtered.astype(np.float32)
# print(mask)
# print(len(mask))
# print(len(noisy_signal))


# Fs = 16000
# sd.play(noisy_signal, Fs)
# sd.wait()  # 再生が完了するまで待機
# plt.plot(noisy_signal)
# plt.show()

# sd.play(filtered, Fs)
# sd.wait()  # 再生が完了するまで待機
# plt.plot(filtered)
# plt.show()


import wave
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import nara_wpe
import tensorflow as tf
from nara_wpe import tf_wpe as wpe
from nara_wpe.utils import stft, istft


wav = wave.open("static/input_audio.wav", "rb")
frames = wav.readframes(-1)
reverb_signal = np.frombuffer(frames, dtype="int16")
Y = stft(reverb_signal,512,256)
with tf.Session() as session:
    Y_tf = tf.placeholder(
        tf.complex128, shape=(None, None, None))
    Z_tf = wpe(Y_tf)
    Z = session.run(Z_tf, {Y_tf: Y})
audio = istft(Z.transpose(1, 2, 0))



print(len(reverb_signal))
print(len(Y))
print(len(Z))
print(len(audio))


Fs = 16000
sd.play(reverb_signal, Fs)
sd.wait()  # 再生が完了するまで待機
plt.plot(reverb_signal)
plt.show()

sd.play(processed_signal, Fs)
sd.wait()  # 再生が完了するまで待機
plt.plot(processed_signal)
plt.show()