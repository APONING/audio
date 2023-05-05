import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 载入音频文件并将其转换为单声道
audio_file = '375milk_1_1s.wav'
y, sr = librosa.load(audio_file, mono=True)

# 合并双声道
if y.ndim > 1:
    y = np.mean(y, axis=1)

# 绘制初始频谱图和波形图
plt.figure(figsize=(10, 8))


plt.subplot(2, 2, 1)
D = np.abs(librosa.stft(y))
S_db = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr, cmap='cool')
plt.title('音频频谱图', fontsize=24)
plt.xlabel('时间 (秒)', fontsize=18)
plt.ylabel('频率 (Hz)', fontsize=18)
plt.colorbar(format='%+2.0f dB')

plt.subplot(2, 2, 2)
librosa.display.waveshow(y, sr=sr)
plt.title('音频波形图', fontsize=24)
plt.xlabel('时间 (秒)', fontsize=18)
plt.ylabel('振幅', fontsize=18)


# 计算MFCC变换
n_mfcc = 13
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

# 绘制MFCC变换后的频谱图和波形图
plt.subplot(2, 2, 3)
librosa.display.specshow(mfccs, x_axis='time', cmap='cool', sr=sr)
plt.title('MFCC变换后频谱图', fontsize=24)
plt.xlabel('时间 (秒)', fontsize=18)
plt.ylabel('MFCC系数', fontsize=18)
plt.colorbar()

plt.subplot(2, 2, 4)
librosa.display.waveshow(librosa.istft(mfccs), sr=sr)
plt.title('MFCC变换后波形图', fontsize=24)
plt.xlabel('时间 (秒)', fontsize=18)
plt.ylabel('振幅', fontsize=18)

plt.tight_layout()
plt.savefig('mfcc.png')
plt.show()