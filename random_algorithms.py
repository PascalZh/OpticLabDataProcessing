from bitarray import bitarray
import numpy as np


def identity(x):
    return x


def delayed_xor(x, delay=101):

    x_bytes = x.tobytes()
    ba = bitarray()
    ba.frombytes(x_bytes)
    ba_ret = bitarray()
    for i in range(delay, len(ba)//8):
        i_ = i - delay  # index for delayed values
        res = [ba[i*8+j] ^ ba[i_*8+j] for j in range(8)]

        for j in range(8):
            ba_ret.append(res[j])

    return np.frombuffer(ba_ret.tobytes(), dtype='uint8')


def delayed_reversed_xor(x, delay=101):

    x_bytes = x.tobytes()
    ba = bitarray()
    ba.frombytes(x_bytes)
    ba_ret = bitarray()
    for i in range(delay, len(ba)//8):
        i_ = i - delay  # index for delayed values
        res = [ba[i*8+j] ^ ba[i_*8+7-j] for j in range(8)]

        for j in range(8):
            ba_ret.append(res[j])

    return np.frombuffer(ba_ret.tobytes(), dtype='uint8')


def delayed_reversed_xor_then_select_4bits(x, delay=101):

    x_bytes = x.tobytes()
    ba = bitarray()
    ba.frombytes(x_bytes)
    ba_ret = bitarray()
    for i in range(delay, len(ba)//8):
        i_ = i - delay  # index for delayed values
        res = [ba[i*8+j] ^ ba[i_*8+7-j] for j in range(8)]

        for j in [0, 1, 6, 7]:
            ba_ret.append(res[j])

    return np.frombuffer(ba_ret.tobytes(), dtype='uint8')


def delayed_reversed_xor_then_mLSB(x, delay=101, m=4):

    x_bytes = x.tobytes()
    ba = bitarray()
    ba.frombytes(x_bytes)
    ba_ret = bitarray()
    for i in range(delay, len(ba)//8):
        i_ = i - delay  # index for delayed values
        res = [ba[i*8+j] ^ ba[i_*8+7-j] for j in range(8)]

        for j in range(m):
            ba_ret.append(res[7-j])

    return np.frombuffer(ba_ret.tobytes(), dtype='uint8')


def cdf_method(x, cdf):
    return (cdf(x) * 256).astype('uint8')


def box_muller(x_bytes, delay=101):
    x_ = np.frombuffer(x_bytes, dtype='uint8')
    x = (x_[delay:] - 128)/40
    y = (x_[:-delay] - 128)/40
    b = np.exp(- (x**2+y**2)/2) * 256
    # plt.figure('Box-Muller time series')
    # plt.plot(b)
    return b.astype('uint8')


def diff(x, n=1):
    return np.diff(x, n=n).astype('uint8')


def subtract_delayed(x, delay):
    return x[delay:] - x[:-delay]


def diff_delayed_reversed_xor(x):
    return delayed_reversed_xor(np.diff(x))
