import numpy as np
import matplotlib.pyplot as plt
from bitarray import bitarray


def generate_cdf(x):
    hist = np.bincount(x, minlength=256).astype(float)
    hist /= len(x)
    hist = np.cumsum(hist)
    plt.figure('CDF')
    plt.plot(hist)
    assert(len(hist) == 256)

    def cdf(y):
        return hist[int(y) % 256]

    return np.vectorize(cdf)


def calculate_bias(x_bytes):
    ba = bitarray()
    ba.frombytes(x_bytes)
    N = len(ba)//8
    cnt = np.array([0 for _ in range(8)], dtype=float)

    for i in range(N):
        for j in range(8):
            cnt[j] += ba[i*8+7-j]

    cnt /= N
    cnt = np.abs(cnt - 0.5)
    return cnt


def plot_bias(ax, x_bytes, label=None):
    cnt = calculate_bias(x_bytes)
    print(f'label: {label}, cnt = {cnt}')
    ax.plot(range(1,9), cnt, label=label)
    ax.set_yscale('log')
    ax.set_title('Bias for every bits')
    ax.legend(fontsize=6)


def H_min(pdf):
    """Unit: bit"""
    return -np.log2(np.max(pdf))
