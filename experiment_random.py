import numpy as np
import matplotlib.pyplot as plt
from bitarray import bitarray
from numpy.random import default_rng
from random_toolbox import pack_bits
from numpy.random import default_rng
from random_algorithms import *
from random_utils import *

rng = default_rng()
bit_width = 8
vals = rng.normal(120, 40, size=100000).astype('uint8')
N = vals.shape[0]//2
bins = 2**bit_width

cdf = generate_cdf(vals)

plt.figure('Normal distribution')
plt.hist(vals, bins=2**8)
# for delay in [1, 2, 3, 5, 7, 29, 73, 101, 257, 503, 977]:
fig_bias = plt.figure('Bias', dpi=300)
ax_bias = fig_bias.add_subplot()


def try_algo(algo, plot_func=None, **kwargs):
    global vals, bins, bit_width, ax_bias, N
    r = algo(vals, **kwargs)[:N]
    print(f'len(r) = {len(r)}')
    plot_bias(ax_bias, r, label=f'{algo.__name__}({kwargs})')
    if plot_func is not None:
        plot_func(r, **kwargs)
    else:
        plt.figure(f'{algo.__name__}({kwargs})')
        plt.hist(r, bins=bins)


try_algo(identity)

try_algo(delayed_xor, delay=101)

for delay in [101]:
    try_algo(delayed_reversed_xor, delay=delay)

def plot_func_cdf(r, **kwargs):
    plt.figure(f'cdf_method({kwargs})')
    plt.subplot(121)
    plt.hist(r, bins=bins)
    plt.subplot(122)
    plt.hist(cdf(vals), bins=np.arange(0, 1, 0.01))


try_algo(cdf_method, plot_func=plot_func_cdf, cdf=cdf)

# try_algo(diff_delayed_reversed_xor)

try_algo(delayed_reversed_xor_then_mLSB, delay=101, m=4)
try_algo(delayed_reversed_xor_then_mLSB, delay=101, m=6)

try_algo(delayed_reversed_xor_then_select_4bits, delay=101)

def plot_func_diff(r, **kwargs):
    plt.figure(f'diff({kwargs})')
    plt.hist(r.astype('int8'), bins=bins)

try_algo(diff, plot_func=plot_func_diff, n=1)
try_algo(diff, plot_func=plot_func_diff, n=2)

plt.show()