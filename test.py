from modulation import *

T = 1               #基带信号宽度
nb = 128            #定义传输的比特数
delta_T = T/200     #采样间隔
fs = 1/delta_T      #采样频率
fc = 4/T           #载波频率
SNR = 10             #信噪比
#

m = Modulation(nb, fs, fc, SNR)
qpsk_wave, data, idata, qdata = m.modulate_16QAM()

idata_noise = m.add_noise(idata, SNR)
qdata_noise = m.add_noise(qdata, SNR)
iqdata = m.create_psk_wave(idata_noise, qdata_noise, 4)

plt.plot(qpsk_wave[:1000], label='SNR='+str(SNR))
plt.title('16QAM Signal')
plt.legend(loc='upper right')
plt.show()
plt.plot(iqdata[:1000], label='SNR='+str(SNR))
plt.title('I+Q Signal')
plt.legend(loc='upper right')
plt.show()
print(data[:20])
print(idata[:5])
print(qdata[:5])
print(len(qpsk_wave))

# m = Modulation(nb, fs, fc, SNR)
# bpsk_wave, data, data_nrz = m.modulate_BPSK()
#
# plt.plot(bpsk_wave[:1000])
# plt.title('BPSK Signal')
# plt.show()
# print(data[:10])
# print(data_nrz[:5])
# print(len(bpsk_wave))
