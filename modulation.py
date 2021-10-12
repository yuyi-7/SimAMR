import numpy as np
from matplotlib import pyplot as plt
from functools import reduce


class Modulation:
    def __init__(self, nb, fs, fc, SNR):
        """
        init function
        :param nb: number of bits of signal
        :param fs: sample frequency
        :param fc: carrier frequency
        :param SNR: signal noise ratio
        """
        self.__nb = nb
        self.__fs = fs
        self.__fc = fc
        self.__SNR = SNR
    
    def get_nb(self):
        return self.__nb
    
    def get_fs(self):
        return self.__fs
    
    def get_fc(self):
        return self.__fc
    
    def get_SNR(self):
        return self.__SNR
    
    def set_nb(self, nb):
        self.__nb = nb
        return self.__nb
    
    def set_fs(self, fs):
        self.__fs = fs
        return self.__fs
    
    def set_fc(self, fc):
        self.__fc = fc
        return self.__fc
    
    def set_SNR(self, SNR):
        self.__SNR = SNR
        return self.__SNR
    
    def add_noise(self, data, SNR):
        """
        add gaussian noise to data acorrding to SNR
        :param data:
        :param SNR:
        :return:
        """
        data_lens = len(data)
        noise = np.random.randn(data_lens)
        
        # 信号平均功率
        signal_power = reduce(
            lambda x, y: x + y, map(lambda x: x * x, data)) / data_lens
        # 期望的噪声
        noise_variance = signal_power / np.power(10, SNR / 10)
        noise = noise * np.sqrt(noise_variance)
        # 信号与噪声相加
        data_noise = data + noise
        return data_noise
    
    def create_baseband(self, nb=None):
        """
        create baseband signal in randomly
        :return: randomly baseband signal
        """
        self.__t = np.arange(0, self.get_nb(), 1 / self.get_fs())
        self.N = len(self.__t)
        
        # 调用随机函数产生任意在0到1的1*nb的矩阵，大于0.5显示为1，小于0.5显示为0
        if nb is None:
            num = self.__nb
        else:
            num = nb
        data = [1 if x > 0.5 else 0 for x in np.random.randn(1, num)[0]]
        return data
    
    def create_qpsk_mod_signal(self, data):
        """
        create QPSK modulation signal
        :param data: baseband data
        :return: in_phase signal and quadrature signal
        """
        # 产生调制信号
        h = 1 / np.math.sqrt(2)
        l = -1 / np.math.sqrt(2)
        idata = []
        qdata = []
        # 进行串并变换
        for i in range(int(self.__nb / 2)):
            if data[i * 2:i * 2 + 2] == [0, 0]:
                idata.append(h)
                qdata.append(h)
            elif data[i * 2:i * 2 + 2] == [0, 1]:
                idata.append(l)
                qdata.append(h)
            elif data[i * 2:i * 2 + 2] == [1, 1]:
                idata.append(l)
                qdata.append(l)
            elif data[i * 2:i * 2 + 2] == [1, 0]:
                idata.append(h)
                qdata.append(l)
        return idata, qdata
    
    def create_psk_wave(self, idata, qdata, n=2):
        """
        create wave of any psk and qam
        :param idata: in-phase signal
        :param qdata: quadrature signal
        :param n: ues n bit as one symbol
        :return: wave sigal
        """
        # 调制
        psk_wave = []
        for i in range(int(self.N / n)):
            try:
                wave_temp = idata[int(np.floor(i / self.__fs))] * \
                            np.cos(2 * np.pi * self.__fc * self.__t[i]) \
                            - qdata[int(np.floor(i / self.__fs))] * \
                            np.sin(2 * np.pi * self.__fc * self.__t[i])
                psk_wave.append(wave_temp)
            except:
                break
        return psk_wave
    
    def modulate_QPSK(self):
        """
        a signal function to create modulated wave of QPSK
        :return: wave of qpsk, baseband data, in_phase data and quadrature data
        """
        data = self.create_baseband()
        idata, qdata = self.create_qpsk_mod_signal(data)
        qpsk_wave = self.create_psk_wave(idata, qdata)
        qpsk_wave_noise = self.add_noise(qpsk_wave, self.__SNR)
        return qpsk_wave_noise, data, idata, qdata
    
    def get_QPSK_iq(self):
        """
        get QPSK IQ signal with noise
        :return: wave of qpsk, baseband data, in_phase data and quadrature data
        """
        data = self.create_baseband()
        idata, qdata = self.create_qpsk_mod_signal(data)
        idata_noise = self.add_noise(idata, self.__SNR)
        qdata_noise = self.add_noise(qdata, self.__SNR)
        return idata_noise, qdata_noise
    
    @staticmethod
    def create_bpsk_mod_signal(data):
        """
        create bpsk modulation signal -- no return zero signal
        :param data: baseband data
        :return: NRZ signal
        """
        return list(map(lambda x: x * 2 - 1, data))
    
    def create_bpsk_wave(self, data_nrz):
        """
        create bpsk wave through NRZ signal
        :param data_nrz:
        :return: wave
        """
        bpsk_wave = []
        for i in range(int(self.N)):
            wave_temp = data_nrz[int(np.floor(i / self.__fs))] * \
                        np.cos(2 * np.pi * self.__fc * self.__t[i])
            bpsk_wave.append(wave_temp)
        return bpsk_wave
    
    def modulate_BPSK(self):
        """
        a sigle function to create wave of BPSK
        :return: wave of bpsk, baseband data, NRZ data
        """
        data = self.create_baseband()
        data_nrz = self.create_bpsk_mod_signal(data)
        bpsk_wave = self.create_bpsk_wave(data_nrz)
        bpsk_wave_noise = self.add_noise(bpsk_wave, self.__SNR)
        return bpsk_wave_noise, data, data_nrz
    
    def get_BPSK_iq(self):
        """
        get BPSK IQ signal with noise
        :return: wave of qpsk, baseband data, in_phase data and quadrature data
        """
        data = self.create_baseband()
        qdata = self.create_bpsk_mod_signal(data)
        idata_noise = self.add_noise(np.zeros(len(qdata)), self.__SNR)
        qdata_noise = self.add_noise(qdata, self.__SNR)
        return idata_noise, qdata_noise
    
    def create_8psk_mod_signal(self, data):
        """
        create 8PSK modulation signal through IQ signal
        :param data: baseband data
        :return: in_phase signal and quadrature signal
        """
        s = np.sin(np.pi / 8)
        c = np.cos(np.pi / 8)
        idata = []
        qdata = []
        # 星座图映射
        for i in range(int(self.__nb / 3)):
            j = i * 3
            if data[j:j + 3] == [0, 0, 0]:
                idata.append(c)
                qdata.append(s)
            elif data[j:j + 3] == [0, 0, 1]:
                idata.append(s)
                qdata.append(c)
            elif data[j:j + 3] == [0, 1, 1]:
                idata.append(-s)
                qdata.append(c)
            elif data[j:j + 3] == [0, 1, 0]:
                idata.append(-c)
                qdata.append(s)
            elif data[j:j + 3] == [1, 1, 0]:
                idata.append(-c)
                qdata.append(-s)
            elif data[j:j + 3] == [1, 1, 1]:
                idata.append(-s)
                qdata.append(-c)
            elif data[j:j + 3] == [1, 0, 1]:
                idata.append(s)
                qdata.append(-c)
            elif data[j:j + 3] == [1, 0, 0]:
                idata.append(c)
                qdata.append(-s)
        
        return idata, qdata
    
    def modulate_8PSK(self):
        """
        a signal function to create modulated wave of 8PSK
        :return: wave of 8psk, baseband data, in_phase data and quadrature data
        """
        data = self.create_baseband()
        idata, qdata = self.create_8psk_mod_signal(data)
        psk8_wave = self.create_psk_wave(idata, qdata, n=3)
        psk8_wave_noise = self.add_noise(psk8_wave, self.__SNR)
        return psk8_wave_noise, data, idata, qdata
    
    def get_8PSK_iq(self):
        """
        get 8PSK IQ signal with noise
        :return: wave of qpsk, baseband data, in_phase data and quadrature data
        """
        data = self.create_baseband()
        idata, qdata = self.create_8psk_mod_signal(data)
        idata_noise = self.add_noise(idata, self.__SNR)
        qdata_noise = self.add_noise(qdata, self.__SNR)
        return idata_noise, qdata_noise
    
    def create_16qam_mod_signal(self, data):
        """
        create 16qam modulation signal through IQ signal
        :param data: baseband data
        :return: in_phase signal and quadrature signal
        """
        a = 1 / np.sqrt(10)
        n = 4
        idata = []
        qdata = []
        # 星座图映射
        for i in range(int(self.__nb / n)):
            j = i * n
            if data[j:j + n] == [0, 0, 0, 0]:
                idata.append(3 * a)
                qdata.append(3 * a)
            elif data[j:j + n] == [0, 0, 0, 1]:
                idata.append(a)
                qdata.append(3 * a)
            elif data[j:j + n] == [0, 0, 1, 0]:
                idata.append(-3 * a)
                qdata.append(3 * a)
            elif data[j:j + n] == [0, 0, 1, 1]:
                idata.append(-a)
                qdata.append(3 * a)
            elif data[j:j + n] == [0, 1, 0, 0]:
                idata.append(3 * a)
                qdata.append(a)
            elif data[j:j + n] == [0, 1, 0, 1]:
                idata.append(a)
                qdata.append(a)
            elif data[j:j + n] == [0, 1, 1, 0]:
                idata.append(-3 * a)
                qdata.append(a)
            elif data[j:j + n] == [0, 1, 1, 1]:
                idata.append(-a)
                qdata.append(a)
            elif data[j:j + n] == [1, 0, 0, 0]:
                idata.append(3 * a)
                qdata.append(-3 * a)
            elif data[j:j + n] == [1, 0, 0, 1]:
                idata.append(a)
                qdata.append(-3 * a)
            elif data[j:j + n] == [1, 0, 1, 0]:
                idata.append(-3 * a)
                qdata.append(-3 * a)
            elif data[j:j + n] == [1, 0, 1, 1]:
                idata.append(-a)
                qdata.append(-3 * a)
            elif data[j:j + n] == [1, 1, 0, 0]:
                idata.append(3 * a)
                qdata.append(-a)
            elif data[j:j + n] == [1, 1, 0, 1]:
                idata.append(a)
                qdata.append(-a)
            elif data[j:j + n] == [1, 1, 1, 0]:
                idata.append(-3 * a)
                qdata.append(-a)
            elif data[j:j + n] == [1, 1, 1, 1]:
                idata.append(-a)
                qdata.append(-a)
        
        return idata, qdata
    
    def modulate_16QAM(self):
        """
        a signal function to create modulated wave of 8PSK
        :return: wave of 8psk, baseband data, in_phase data and quadrature data
        """
        data = self.create_baseband()
        idata, qdata = self.create_16qam_mod_signal(data)
        qam16_wave = self.create_psk_wave(idata, qdata, n=4)
        qam16_wave_noise = self.add_noise(qam16_wave, self.__SNR)
        return qam16_wave_noise, data, idata, qdata
    
    def get_16QAM_iq(self):
        """
        get 16QAM IQ signal with noise
        :return: wave of qpsk, baseband data, in_phase data and quadrature data
        """
        data = self.create_baseband()
        idata, qdata = self.create_16qam_mod_signal(data)
        idata_noise = self.add_noise(idata, self.__SNR)
        qdata_noise = self.add_noise(qdata, self.__SNR)
        return idata_noise, qdata_noise
    