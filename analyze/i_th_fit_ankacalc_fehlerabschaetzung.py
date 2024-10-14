#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='find the bursting threshold using ankacalc equations')
    parser.add_argument('vrf', type=int, nargs='+')

    args = parser.parse_args()

    font = {'size': 19}
    matplotlib.rc('font', **font)

    vrf_list = np.asarray(args.vrf)


    def sigma(fs, vrf, e_spread, f_rf):
        return ankacalc.sigma_exact(fs,vrf, e_spread, f_rf)


    def alpha(fs, vrf, e_spread, f_rf):
        ankacalc.sigma = ankacalc.sigma_exact
        return ankacalc.alpha(fs,vrf, e_spread, f_rf)


    def ith_bunchbeam(fs, vrf, e_spread, f_rf):
        ankacalc.sigma = ankacalc.sigma_exact
        return ankacalc.ith_bunchbeam_cai(fs,vrf, e_spread, f_rf)


    def montecarlo_fehler(mu, sigma, montecarlo=True, plot=False):
        s = np.random.normal(mu, sigma, 1000)

        if not montecarlo:
            s = np.ones(1000) * mu

        if plot:
            plt.figure()
            count, bins, ignored = plt.hist(s, 30, normed=True)
            plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                     linewidth=2, color='r')

        return s

    ## Data:
    # import von custom json is possible mit json and yaml!!
    with open("threshold_data_incl_sbb_new_json.json", 'r') as f:
        data_json = json.loads(f.read())
    data_json = parse_data_after_readin(data_json)

    # Daten squeeze scan f05110
    data_json['5110']['sigma'] = sigma(data_json['5110']['fs'], data_json['5110']['vrf'] * ankacalc.voltage_factor().nominal_value, ankacalc.e_spread, ankacalc.f_rf)

    # Daten rf-squeeze scan f05136
    data_json['5136']['sigma'] = sigma(data_json['5136']['fs'], data_json['5136']['vrf'] * ankacalc.voltage_factor().nominal_value, ankacalc.e_spread, ankacalc.f_rf)


    fs_theo = np.linspace(1, 20, 100)
    # sigmamm = scipy.constants.c*0.47e-3 *1e3 * 1.3e9*constants.ANKA.h/((499.71e6)**2*4*vrf*1e3)*fs

    ## fehler montecarlo
    if True:
        np.random.seed(42)
        montecarlo_vrf = False
        montecarlo_fs = False
        montecarlo_e_spread = False
        montecarlo_f_rf = False

        fehler_vrf = data_json['5110']['vrf'][0] * 0.05
        # fehler_fs = [fs*0.05 for fs in data_json['5110']['fs']]
        # fehler_e_spread = e_spread*0.05
        # fehler_f_rf = f_rf*0.00001

        # fehler_vrf = 10
        fehler_fs = [0.1 for fs in data_json['5110']['fs']]
        fehler_e_spread = 0.01e-3
        fehler_f_rf = 0.01e6

        vrf_montec = montecarlo_fehler(data_json['5110']['vrf'][0], fehler_vrf, montecarlo=montecarlo_vrf)
        fs_montec_array = [montecarlo_fehler(fs, fehler_fs[i], montecarlo=montecarlo_fs) for i, fs in enumerate(data_json['5110']['fs'])]
        e_spread_montec = montecarlo_fehler(ankacalc.e_spread, fehler_e_spread, montecarlo=montecarlo_e_spread)
        f_rf_montec = montecarlo_fehler(ankacalc.f_rf, fehler_f_rf, montecarlo=montecarlo_f_rf)

        ith_bunchbeam_montec_mean = np.array(
            [np.array(ith_bunchbeam(fs_montec_array[i], vrf_montec* ankacalc.voltage_factor().nominal_value, e_spread_montec, f_rf_montec) * 1000).mean() for
             i, fs in enumerate(data_json['5110']['fs'])])
        ith_bunchbeam_montec_std = np.array(
            [np.array(ith_bunchbeam(fs_montec_array[i], vrf_montec* ankacalc.voltage_factor().nominal_value, e_spread_montec, f_rf_montec) * 1000).std() for
             i, fs in enumerate(data_json['5110']['fs'])])
        sigma_mm_d_montec_mean_list = np.array(
            [np.array(sigma(fs_list, vrf_montec* ankacalc.voltage_factor().nominal_value, e_spread_montec, f_rf_montec) * 1e3).mean() for fs_list in
             fs_montec_array])
        sigma_mm_d_montec_std_list = np.array(
            [np.array(sigma(fs_list, vrf_montec* ankacalc.voltage_factor().nominal_value, e_spread_montec, f_rf_montec) * 1e3).std() for fs_list in
             fs_montec_array])


    ## I_th over bunch length with theory curves and points
    plt.figure(figsize=(12, 6), tight_layout=True)
    col = matplotlib.cm.rainbow(np.linspace(0, 1, len(args.vrf)))

    for i, vrf in enumerate(np.sort(vrf_list)):
        index = np.where(data_json['5136']['vrf'] == vrf)
        if vrf == 300:  # plot für reinen squeeze scan
            plt.errorbar(data_json['5110']['sigma'] * 1e3, data_json['5110']['ith'], yerr=0.0094, fmt='o', elinewidth=2, color=col[i], marker='^',
                         markersize=6)
            plt.errorbar(sigma_mm_d_montec_mean_list, ith_bunchbeam_montec_mean, xerr=sigma_mm_d_montec_std_list,
                         yerr=ith_bunchbeam_montec_std, color='gray', fmt='o', elinewidth=2, label='montecarlo')

        plt.errorbar(data_json['5136']['sigma'][index] * 1e3, data_json['5136']['ith'][index], yerr=0.0094, color=col[i], fmt='o', elinewidth=2)
        # plt.scatter(sigma(data_json['5136']['fs'][index], vrf) * 1e3, ith_bunchbeam(data_json['5136']['fs'][index], vrf, e_spread, f_rf)*1000, marker='x', color=col[i])
        plt.plot(sigma(fs_theo, vrf * ankacalc.voltage_factor().nominal_value, ankacalc.e_spread, ankacalc.f_rf) * 1e3, ith_bunchbeam(fs_theo, vrf * ankacalc.voltage_factor().nominal_value, ankacalc.e_spread, ankacalc.f_rf) * 1000,
                 label=r'4$\mathdefault{\times}$ %i kV' % vrf,
                 color=col[i])


    plt.legend(loc='upper left', ncol=1, frameon=False)
    plt.xlabel(u'natürliche Bunchlänge / mm')
    plt.xlim(xmin=0.7, xmax=1.75)
    plt.ylim(ymax=0.5, ymin=0.05)
    plt.ylabel('Burstingschwelle / mA')
    # plt.title('gleichungen von theorie')


    ## I_th over fs with theory points
    plt.figure(figsize=(12, 6), tight_layout=True)
    plt.errorbar(data_json['5110']['fs'], data_json['5110']['ith'], yerr=0.0094, fmt='o', elinewidth=2, color='k')
    plt.errorbar(data_json['5110']['fs'], ith_bunchbeam_montec_mean, xerr=[fs_list.std() for fs_list in fs_montec_array],
                 yerr=ith_bunchbeam_montec_std, color='gray', fmt='o', elinewidth=2, label='montecarlo')

    col = matplotlib.cm.rainbow(np.linspace(0, 1, len(args.vrf)))
    plt.plot(data_json['5136']['fs'][:2], data_json['5136']['ith'][:2], '--', color='k')
    plt.plot(data_json['5136']['fs'][2:5], data_json['5136']['ith'][2:5], '--', color='k')
    plt.plot(data_json['5136']['fs'][5:9], data_json['5136']['ith'][5:9], '--', color='k')
    plt.plot(data_json['5136']['fs'][9:13], data_json['5136']['ith'][9:13], '--', color='k')
    for i, vrf in enumerate(np.sort(vrf_list)):
        index = np.where(data_json['5136']['vrf'] == vrf)
        plt.errorbar(data_json['5136']['fs'][index], data_json['5136']['ith'][index], yerr=0.0094, color=col[i], fmt='o', elinewidth=2,
                     label=r'4$\mathdefault{\times}$ %i kV' % vrf)

        plt.plot(fs_theo, ith_bunchbeam(fs_theo, vrf* ankacalc.voltage_factor().nominal_value, ankacalc.e_spread, ankacalc.f_rf) * 1000,
                 label=r'4$\mathdefault{\times}$ %i kV' % vrf, color=col[i])

    plt.legend(loc='upper left', ncol=1, numpoints=1, frameon=False)
    plt.xlabel('Synchrotronfrequenz / kHz')
    plt.ylabel('Burstingschwelle / mA')
    # plt.legend(loc='upper left', ncol=1)
    # plt.title('gleichungen von theorie')


    plt.show()
