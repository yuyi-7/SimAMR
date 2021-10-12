from modulation import *
import numpy as np


def mods_create_data(mods, Modulate, nb, fs, fc, SNR, data_num):
    for i in range(data_num):
        # 先示例化
        m = Modulate(nb, fs, fc, SNR)
        # 判断需要哪种调制的数据
        if mods == 'bpsk':
            idata, qdata = m.get_BPSK_iq()
            y_temp = 0
        elif mods == 'qpsk':
            idata, qdata = m.get_QPSK_iq()
            y_temp = 1
        elif mods == '8psk':
            idata, qdata = m.get_8PSK_iq()
            y_temp = 2
        elif mods == '16qam':
            idata, qdata = m.get_16QAM_iq()
            y_temp = 3
        else:
            raise Exception('At present we dot have the modulation method '+mods)
        idata = np.array(idata).reshape((1, 1, 1, -1))
        qdata = np.array(qdata).reshape((1, 1, 1, -1))
        x_temp = np.concatenate((idata, qdata), axis=2)
        
        if i == 0:
            x = x_temp
            y = np.array([y_temp])
        else:
            x = np.concatenate((x, x_temp), axis=0)
            y = np.concatenate((y, np.array([y_temp])), axis=0)
    print(mods+' x shape:'+str(x.shape)+' y shape:'+str(y.shape))
    print(mods+' data done')
    return x, y
    
    
def generate_data(fs, fc, SNR, sample_num, sample_long):
    mods = ['bpsk', 'qpsk', '8psk', '16qam']
    nb = [sample_long, sample_long*2, sample_long*3, sample_long*4]
    for index, value in enumerate(mods):
        x_temp, y_temp = mods_create_data(
            value, Modulation, nb[index], fs, fc, SNR, sample_num)
        if index == 0:
            x = x_temp
            y = y_temp
        else:
            x = np.concatenate((x, x_temp), axis=0)
            y = np.concatenate((y, y_temp), axis=0)
    print('data generate done!')
    print(x.shape, y.shape)
    return x, y
        
        
    