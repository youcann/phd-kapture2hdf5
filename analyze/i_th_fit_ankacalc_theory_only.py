#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
import scipy.optimize
import scipy.constants
import i_th_fit_ankacalc as ankacalc


# TODO: check if sigma in m or mm!
def sigma(energy, fs, vrf):
    ankacalc.e_spread = e_spread(energy)
    ankacalc.energy = energy
    return ankacalc.sigma(fs, vrf)


# def sigma(fs, vrf):
#    vrf = vrf * 1e3
#    return scipy.constants.c * e_spread *1e3 * energy*constants.ANKA.h/((f_rf)**2*np.sqrt((vrf)**2 - (8.8575e-5*(1e-9**3)/5.559*(energy)**4)**2))*fs


def alpha(energy, fs, vrf):
    return sigma(energy, fs, vrf) * 2 * np.pi * fs * 1e3 / (scipy.constants.c * e_spread(energy))


def ith_bunchbeam(energy, fs, vrf):
    ankacalc.energy = energy
    return ankacalc.ith_bunchbeam_cai(fs, vrf)


def e_spread(energy):
    return 0.9e-3 / 2.5e9 * energy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='find the bursting threshold using ankacalc equations')
    parser.add_argument('vrf', type=int, nargs='+')
    parser.add_argument('--energy', type=float, nargs='+', default=[1.3, ], help='Energy in GeV')
    parser.add_argument('--fs-list', type=float, nargs='+', help='fs in kHz')
    parser.add_argument('--squeeze', type=float, nargs='+', help='squeeze steps in k')

    args = parser.parse_args()

    font = {'size': 19}
    matplotlib.rc('font', **font)

    vrf_list = np.asarray(args.vrf)
    energy_list = np.asarray(args.energy) * 1e9

    fs = np.linspace(0, 40, 200)

    ## I_th over fs
    plt.figure(figsize=(12, 6), tight_layout=True)
    col = matplotlib.cm.rainbow(np.linspace(0, 1, len(args.vrf)))
    import itertools

    marker = itertools.cycle(('v', 'o', 's', 'D'))
    line = ['-', '--', '-.', ':']

    for j, energy in enumerate(energy_list):
        for i, vrf in enumerate(vrf_list):
            plt.plot(fs, ith_bunchbeam(energy, fs, vrf) * 1000, color=col[i],
                     label='%i kV - %.1f GeV' % (vrf, energy / 1.0e9), linestyle=line[j])

    plt.ylim(5e-3, 1)
    plt.xlim(1, 30)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    legend = plt.legend(loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.2, handletextpad=0.2,
                        handlelength=1, frameon=False, borderpad=0.2, fontsize=19, title='RF voltage')
    legend.draggable(state=True)

    plt.xlabel('Synchrotron frequency / kHz')
    plt.ylabel('Bursting threshold / mA')
    # plt.locator_params(axis='y', nbins=6)

    from matplotlib.ticker import FuncFormatter


    def minor_format(x, i=None):
        return int(x)


    plt.gca().xaxis.set_minor_formatter(FuncFormatter(minor_format))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(minor_format))
    # plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.tick_params('x', which='minor', direction='inout')
    ax.get_xaxis().set_tick_params(which='minor', pad=3)
    plt.minorticks_on()
    # plt.savefig('i_th_over_fs_theory.png')

    ## alpha für verschiedene Einstellungen:
    plt.figure(figsize=(12, 6), tight_layout=True)
    for j, energy in enumerate(energy_list):
        for i, vrf in enumerate(vrf_list):
            print(args.fs_list[i], vrf, alpha(energy, args.fs_list[i], vrf))
            plt.scatter(args.squeeze[i], alpha(energy, args.fs_list[i], vrf), color=col[i],
                        label='%i kV - %.1f GeV' % (vrf, energy / 1.0e9), linestyle=line[j])

    plt.xlim(22, 32)
    plt.ylim(0, 9e-4)
    plt.xlabel('Squeeze State / ksteps')
    plt.ylabel('alpha')

    plt.show()
