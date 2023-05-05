import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取音频文件
rate, data = wavfile.read('375milk_1_1s.wav')

# 合并双声道
if len(data.shape) > 1:
    data = np.mean(data, axis=1)

# 小波变换
wavelet = pywt.Wavelet('sym4')
coeffs = pywt.wavedec(data, wavelet)

# 绘制初始频谱图
plt.subplot(2, 2, 1)
plt.specgram(data, Fs=rate)
plt.title('初始频谱图', fontsize=16)

# 绘制波形图
plt.subplot(2, 2, 2)
plt.plot(data)
plt.title('初始波形图', fontsize=16)

# 绘制小波变换后的频谱图
plt.subplot(2, 2, 3)
plt.specgram(pywt.waverec(coeffs, wavelet), Fs=rate)
plt.title('小波变换后的频谱图', fontsize=16)

# 绘制小波变换后的波形图
plt.subplot(2, 2, 4)
plt.plot(pywt.waverec(coeffs, wavelet))
plt.title('小波变换后的波形图', fontsize=16)
plt.tight_layout()
plt.savefig('xiaobo.png')
# 显示图形
plt.show()