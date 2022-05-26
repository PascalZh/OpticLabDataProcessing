import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
from numpy.random import default_rng
from random_algorithms import *
from random_utils import *

def RED(t):
    return colored(t, 'red')

def YELLOW(t):
    return colored(t, 'yellow')

def BLUE(t):
    return colored(t, 'blue')

def CYAN(t):
    return colored(t, 'cyan')

def PURPLE(t):
    return colored(t, 'magenta')

rng = default_rng()
bit_width = 8
sigma = 40
mu = 128
rand_seq = rng.normal(mu, sigma, size=2000000).astype('uint8')
N = rand_seq.shape[0]//2  # the length of the post-processed 8 bit number sequence to be analyzed
bins = 2**bit_width

cdf = generate_cdf(rand_seq)

plt.figure('Original Sequence')
plt.subplot(121)
plt.plot(rand_seq[0:500])
plt.ylim(0, 256)
plt.subplot(122)
n, _, _ = plt.hist(rand_seq, bins=2**8, range=(0, bins-1))

print(BLUE(f'The entropy of origin series is {H_min(n/rand_seq.shape[0])}.'))

# for delay in [1, 2, 3, 5, 7, 29, 73, 101, 257, 503, 977]:
fig_bias = plt.figure('Bias', dpi=150)
ax_bias = fig_bias.add_subplot()
ax_bias.plot(range(1,9), 3 * np.repeat(0.5/np.sqrt(N), 8), color='black', linestyle='dashdot', label='$3\sigma$')


def try_algo(algo, plot_func=None, **kwargs):
    global rand_seq, bins, bit_width, ax_bias, N
    r = algo(rand_seq, **kwargs)[:N]
    print(f'len(r) = {len(r)}')
    plot_bias(ax_bias, r, label=f'{algo.__name__}({kwargs})')
    if plot_func is not None:
        plot_func(r, **kwargs)
    else:
        plt.figure(f'{algo.__name__}({kwargs})')
        plt.hist(r, bins=bins, range=(0, bins-1))
    print()


try_algo(identity)

# try_algo(delayed_xor, delay=101)

for delay in [101]:
    try_algo(delayed_reversed_xor, delay=delay)

def plot_func_cdf(r, **kwargs):
    plt.figure(f'cdf_method({kwargs})')
    plt.subplot(121)
    plt.hist(r, bins=bins, range=(0, bins-1))
    plt.subplot(122)
    plt.hist(cdf(rand_seq), bins=np.arange(0, 1, 0.01))


# try_algo(cdf_method, plot_func=plot_func_cdf, cdf=cdf)

# try_algo(diff_delayed_reversed_xor)

try_algo(delayed_reversed_xor_then_mLSB, delay=101, m=4)
try_algo(delayed_reversed_xor_then_mLSB, delay=101, m=6)

# try_algo(delayed_reversed_xor_then_select_4bits, delay=101)

def plot_func_diff(r, **kwargs):
    plt.figure(f'diff({kwargs})')
    plt.hist(r.astype('int8'), bins=bins, range=(0, bins-1))

# try_algo(diff, plot_func=plot_func_diff, n=1)
# try_algo(diff, plot_func=plot_func_diff, n=2)

plt.show()