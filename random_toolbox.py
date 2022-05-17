#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import functools
from bitarray import bitarray


class ArgumentParserWithDefaultsHelpFormatter(argparse.ArgumentParser):
    def __init__(self, *args, formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs):
        kwargs['formatter_class'] = formatter_class
        super().__init__(*args, **kwargs)


def read_osc_data(filename):
    with open(filename, 'r') as f:
        return np.array([float(x) for x in f.read().splitlines()])


def pack_bits(bits, bit_width=1, dtype='int', big_endian=True):
    """Notice: ensure a variable of dtype can contain any integer number of width `bit_width`"""
    ba = bitarray()
    ba.frombytes(bits)

    N = len(ba)
    assert(N % bit_width == 0)

    ret = np.zeros(N // bit_width, dtype=dtype)
    for i in range(N // bit_width):
        for j in range(bit_width):
            if big_endian:
                ret[i] += ba[i*bit_width+j] << (bit_width - j - 1)
            else:  # little-endian
                ret[i] += ba[i*bit_width+j] << j
    return ret


def plot_hist(args):

    bit_width = args.bit_width
    file = args.FILE
    save_fig_file = args.save_fig
    file_format = args.file_format
    x = None

    with open(file, 'rb' if file_format == 'bin' else 'r') as f:
        b = f.read()
        if file_format == 'plain_hex_dump':
            b = bytes.fromhex(b[:len(b)//2*2])
        elif file_format == 'bit_dump':
            b = b[:len(b)//8*8]
            ba = bitarray(b)
            b = ba.tobytes()

        trunc_len = len(b) // bit_width * bit_width
        b = b[:trunc_len]

        if bit_width == 8:
            x = np.frombuffer(b, dtype='>u1')  # big-endian
        elif bit_width == 16:
            x = np.frombuffer(b, dtype='>u2')
        elif bit_width == 32:
            x = np.frombuffer(b, dtype='>u4')
        elif bit_width == 64:
            x = np.frombuffer(b, dtype='>u8')
        else:
            x = pack_bits(b, bit_width=bit_width)

    bins = 2**bit_width
    plt.hist(x, bins=bins)
    if save_fig_file == '':
        plt.show()
    else:
        plt.savefig(save_fig_file)


def main(args):

    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot()
    # x = read_osc_data('1S.txt')
    # x_binary, x_compare = demodulate_osc_data(x, compare=True)
    # ax2.hist(np.packbits(x_binary), bins=2**8)
    plt.show()


def demodulate_osc_data(x, T=64, compare=True):
    L = x.shape[0] // T
    bin_seq = np.array([(x[i*T:(i+1)*T] > 0).sum() for i in range(L)])
    bin_seq = (bin_seq > T/2)

    if compare:
        bin_seq_for_plot = np.amin(x) + (np.amax(x) - np.amin(x)) * functools.reduce(lambda a, b: np.concatenate((a, b)), [np.repeat(bin_seq[i], T) for i in range(L)])
        return bin_seq, bin_seq_for_plot
    return bin_seq, None


if __name__ == '__main__':

    parser = ArgumentParserWithDefaultsHelpFormatter(description='Random number generation lab toolbox')
    subparsers = parser.add_subparsers(description='subcommands contains various tools',
            required=True,
            parser_class=type(parser))

    parser_plot_hist = subparsers.add_parser('plot_hist',
            help='plot histogram of a random sequence, parsing as W-bit number (big-endian)')
    parser_plot_hist.add_argument('FILE', default='test.bin',
            help='inputted random sequence file')
    parser_plot_hist.add_argument('--save_fig', metavar='IMAGE_NAME', default='',
            help='save fig as image')
    parser_plot_hist.add_argument('--bit_width', metavar='W', type=int, default=4,
            help='random bit sequence will be viewed as W-bit number sequence (big-endian)')
    parser_plot_hist.add_argument('--file_format', default='bin',
            choices=['bin', 'plain_hex_dump', 'bit_dump'],
            help='bin: binary file; plain_hex_dump: txt files, see outputs of xxd -p; bit_dump: txt files, see outputs of xxd -b')
    parser_plot_hist.set_defaults(func=plot_hist)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)

