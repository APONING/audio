import numpy as np
import matplotlib.pyplot as plt
import librosa.display

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 载入双声道音频文件
audio_file = '375milk_1_1s.wav'
y, sr = librosa.load(audio_file, mono=False)

# 合并为单声道
y_mono = librosa.to_mono(y)

# 计算短时傅里叶变换
n_fft = 2048
hop_length = 512
S = np.abs(librosa.stft(y_mono, n_fft=n_fft, hop_length=hop_length))

# 绘制初始频谱图
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
plt.title('初始频谱图', fontsize=20)
plt.xlabel('时间 (秒)', fontsize=16)
plt.ylabel('频率 (Hz)', fontsize=16)
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# 绘制波形图
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y_mono, sr=sr)
plt.title('波形图', fontsize=20)
plt.xlabel('时间 (秒)', fontsize=16)
plt.ylabel('振幅', fontsize=16)
plt.tight_layout()
plt.show()

# 计算逆短时傅里叶变换并绘制变换后的频谱图
y_inv = librosa.istft(S, hop_length=hop_length)
S_inv = np.abs(librosa.stft(y_inv, n_fft=n_fft, hop_length=hop_length))

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time')
plt.title('初始频谱图', fontsize=20)
plt.xlabel('时间 (秒)', fontsize=16)
plt.ylabel('频率 (Hz)', fontsize=16)
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 2, 2)
librosa.display.waveshow(y_mono, sr=sr)
plt.title('波形图', fontsize=20)
plt.xlabel('时间 (秒)', fontsize=16)
plt.ylabel('振幅', fontsize=16)

plt.subplot(2, 2, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_inv, ref=np.max), y_axis='log', x_axis='time')
plt.title('变换后频谱图', fontsize=20)
plt.xlabel('时间 (秒)', fontsize=16)
plt.ylabel('频率 (Hz)', fontsize=16)
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 2, 4)
librosa.display.waveshow(y_inv, sr=sr)
plt.title('变换后波形图', fontsize=20)
plt.xlabel('时间 (秒)', fontsize=16)
plt.ylabel('振幅', fontsize=16)

plt.tight_layout()
plt.savefig('stft.png')
plt.show()