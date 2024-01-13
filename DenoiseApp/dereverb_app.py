from flask import Flask, request, render_template, send_file, jsonify, send_from_directory
import os
import numpy as np
import wave
import wave_func as wf
import scipy
from scipy.ndimage import maximum_filter1d
import matplotlib.pyplot as plt
import scipy.signal as sp
import librosa

app = Flask(__name__, template_folder='templates')

# 音声ファイルの保存ディレクトリを指定
UPLOAD_FOLDER = 'static'  # 音声ファイルを保存するディレクトリ

# アップロードディレクトリを設定
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/')
def index():
    return render_template('dereverb.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio_file' not in request.files:
        return jsonify(message="音声ファイルがアップロードされていません"), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify(message="ファイル名が空です"), 400

    # 入力音声ファイルを保存
    input_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_audio.wav')
    audio_file.save(input_audio_path)

    # 出力音声ファイルを保存
    output_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dereverb.wav')

    ######音声処理########
    # 入力ファイルを開く
    with wave.open(input_audio_path, "rb") as wav:
        frames = wav.readframes(-1)
        noisy_signal = np.frombuffer(frames, dtype="int16")
        # audioLen = len(audio)
        # t = np.linspace(0, 1, audioLen, endpoint=False)  # 時間軸
        # noise = np.random.normal(0, 0.5, t.shape)  # ノイズ
        # noisy_signal = audio + noise

    def envelope(y, rate, threshold):
        """
        Args:
            - y: 信号データ
            - rate: サンプリング周波数
            - threshold: 雑音判断するしきい値
        Returns:
            - mask: 振幅がしきい値以上か否か
            - y_mean: Sound Envelop
        """
        y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
        mask = [mean > threshold for mean in y_mean]
        return mask, y_mean
    
    mask, y_mean = envelope(noisy_signal, wav.getframerate(), 6000)
    audio = np.array(noisy_signal) * np.array(mask)

    # 出力ファイルを開く
    with wave.open(output_audio_path, "wb") as out:
        out.setnchannels(wav.getnchannels())
        out.setsampwidth(wav.getsampwidth())
        out.setframerate(wav.getframerate())
        out.writeframes(audio.tobytes())

    ########################

    # 音声ファイルへのURLを返す
    input_audio_url = '/static/input_audio.wav'
    output_audio_url = '/static/dereverb.wav'
    return jsonify(output_audio_url=output_audio_url, input_audio_url=input_audio_url)

@app.route('/delete_static_wav', methods=['POST'])
def delete_static_wav():
    try:
        for f in os.listdir("static"):
            if f.endswith(".wav"):
                os.remove(os.path.join("static", f))
        return jsonify(message="静的なWAVファイルを削除しました")
    except Exception as e:
        return jsonify(message="エラーが発生しました", error=str(e)), 500

def serve_wav():
    response = make_response(send_from_directory('static', 'dereverb.wav'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True)


