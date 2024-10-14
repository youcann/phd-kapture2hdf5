#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import i_th_fit_ankacalc as ankacalc
import json


def parse_data_after_readin(data_new):
    for key, value in data_new.iteritems():
        if isinstance(value, dict):
            for key1, value1 in value.iteritems():
                if isinstance(value1, list):
                    value[key1] = np.array(value1)
        if isinstance(value, list):
            data_new[key] = np.array(value)
    return data_new

def fs_error(fs):
    fs_err_fac = 0.016
    fs_err = 0.5
    fs_err = 0.1  # 100Hz
    # return fs*fs_err_fac
    return fs_err

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='find the bursting threshold using ankacalc equations')
    parser.add_argument('vrf', type=int, nargs='+')

    args = parser.parse_args()

    font = {'size': 19}
    matplotlib.rc('font', **font)

    vrf_list = np.asarray(args.vrf)
    fs = np.linspace(0, 20, 200)


    ##Daten
    # import von custom json is possible mit json and yaml!!
    with open("threshold_data_incl_sbb_new_json.json", 'r') as f:
        data_json = json.loads(f.read())
    data_json = parse_data_after_readin(data_json)

    # neue messung f05731
    # neue messung f05732


    ## I_th over fs
    plt.figure(figsize=(12, 6), tight_layout=True)
    # plt.figure(figsize=(10, 5), tight_layout=True)

    col = matplotlib.cm.rainbow(np.linspace(0, 1, len(args.vrf)))
    # import itertools
    # marker = itertools.cycle(('v', 'o', 's', 'D'))

    for i, vrf in enumerate(vrf_list):
        ax = plt.gca()
        for key in ['5731', '5732']:
            index = np.where(data_json[key]['vrf'] == vrf)

            plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index], xerr=fs_error(data_json[key]['fs'][index]), yerr=0.015, color='k',
                         ecolor=col[i], fmt='*', elinewidth=2)

    if True:
        ax = plt.gca()
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        for i, vrf in enumerate(vrf_list):
            plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf) * 1000, color=col[i], label='%i kV' % vrf)
            # FixMe: not 1 sigma error of all parameters, only vrf considered!
            plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf*1.01) * 1000, color=col[i], linestyle=':')
            plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf*0.99) * 1000, color=col[i], linestyle=':')

        plt.ylim(ylim)
        plt.xlim(xlim)

    # plt.xlim(xmin=4, xmax=14)
    # plt.ylim(ymax=0.5, ymin=0.05)

    plt.legend(loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.2, handletextpad=0.2,
               handlelength=0.2, frameon=False, borderpad=0.2, fontsize=19, title='RF voltage')

    handles, labels = ax.get_legend_handles_labels()

    plt.xlabel('Synchrotron frequency / kHz')
    plt.ylabel('Bursting threshold / mA')
    plt.locator_params(axis='y', nbins=6)
    # plt.savefig('f05731-f05732_i_th_over_fs_both.png')


    plt.show()
