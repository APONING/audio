import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import wiener

# 1. 读取音频文件
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 1. 读取音频文件
audio_file = '375milk_1_1s.wav'
data, sample_rate = sf.read(audio_file)
if data.ndim > 1:
    data = np.mean(data, axis=1)
time = np.linspace(0, len(data) / sample_rate, len(data))

# 2. 向音频添加随机噪声
noise = np.random.normal(0, 0.1, len(data))
noisy_data = data + noise

# 3. 应用德纳滤波器以降低噪声
filtered_data = wiener(noisy_data)

# 4. 绘制原始音频、噪声音频和降噪后的音频
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time, data)
plt.title('原始音频', fontsize=20)

plt.subplot(3, 1, 2)
plt.plot(time, noisy_data)
plt.title('噪声音频', fontsize=20)

plt.subplot(3, 1, 3)
plt.plot(time, filtered_data)
plt.title('降噪后的音频', fontsize=20)

plt.tight_layout()
plt.savefig('dena.png')
plt.show()
