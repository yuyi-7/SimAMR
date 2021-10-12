from modulation import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

T = 1               #基带信号宽度
nb = 128            #定义传输的比特数
delta_T = T/200     #采样间隔
fs = 1/delta_T      #采样频率
fc = 4/T           #载波频率
SNR = 10             #信噪比
#

# m = Modulation(nb, fs, fc, SNR)
# qpsk_wave, data, idata, qdata = m.modulate_16QAM()
#
# plt.plot(qpsk_wave[:1000], label='SNR='+str(SNR))
# plt.title('16QAM Signal')
# plt.legend(loc='upper right')
# plt.show()
# print(data[:20])
# print(idata[:5])
# print(qdata[:5])
# print(len(qpsk_wave))

# m = Modulation(nb, fs, fc, SNR)
# bpsk_wave, data, idata, qdata = m.modulate_16QAM()
#
# print(len(idata))

a = np.array([0,2,1,2,3,3])
b = np.array([0,2,1,1,2,2])
c = confusion_matrix(a, b)
fig, ax = plt.subplots()
sns.heatmap(c, annot=True, ax=ax)
ax.set_title('confusion matrix SNR='+str(SNR))
ax.set_xlabel('Predict')
ax.set_ylabel('True')
ax.set_xticklabels(['bpsk', 'qpsk', '8psk', '16qam'])
ax.set_yticklabels(['bpsk', 'qpsk', '8psk', '16qam'])
plt.show()