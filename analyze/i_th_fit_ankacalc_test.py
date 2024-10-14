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
from uncertainties import unumpy
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
    # fs_err = 0.01  # 10Hz
    # return fs*fs_err_fac
    return fs_err

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='find the bursting threshold using ankacalc equations')
    parser.add_argument('vrf', type=int, nargs='+')
    parser.add_argument('--loglog', action='store_true')
    parser.add_argument('--plot1', action='store_true')
    parser.add_argument('--plot2', action='store_true')
    parser.add_argument('--plot3', action='store_true')
    parser.add_argument('--plot4', action='store_true')
    parser.add_argument('--plot5', action='store_true')
    parser.add_argument('--plot6', action='store_true')
    parser.add_argument('--write', action='store_true')

    args = parser.parse_args()

    font = {'size': 19}
    matplotlib.rc('font', **font)

    vrf_list = np.asarray(args.vrf)
    fs = np.linspace(0, 20, 100)

    ##Daten
    # import von custom json is possible mit json and yaml!!
    with open("threshold_data_incl_sbb_new_json.json", 'r') as f:
        data_json = json.loads(f.read())
    data_json = parse_data_after_readin(data_json)

    # # Daten squeeze scan f05110
    data_json['5110']['sigma'] = ankacalc.sigma(data_json['5110']['fs'], data_json['5110']['vrf']*ankacalc.voltage_factor().nominal_value)
    data_json['5110']['shielding'] = ankacalc.shielding_cai(data_json['5110']['fs'], data_json['5110']['vrf']*ankacalc.voltage_factor().nominal_value)

    # Daten rf-squeeze scan f05136
    data_json['5136']['sigma'] = ankacalc.sigma(data_json['5136']['fs'], data_json['5136']['vrf']*ankacalc.voltage_factor().nominal_value)
    data_json['5136']['shielding'] = ankacalc.shielding_cai(data_json['5136']['fs'], data_json['5136']['vrf']*ankacalc.voltage_factor().nominal_value)

    # # Daten rf-squeeze scan f05606 - noch grob abgelesene Schwelle aus inst. Spektrogram
    data_json['5606']['sigma'] = ankacalc.sigma(data_json['5606']['fs'], data_json['5606']['vrf']*ankacalc.voltage_factor().nominal_value)
    data_json['5606']['shielding'] = ankacalc.shielding_cai(data_json['5606']['fs'], data_json['5606']['vrf']*ankacalc.voltage_factor().nominal_value)


    if args.write:
        print('squeeze scan:')
        for i in range(len(data_json['5110']['fs'])):
            print(data_json['5110']['fs'][i], data_json['5110']['ith'][i], data_json['5110']['sigma'][i])
        print('suqeeze rf scan:')
        for i in range(len(data_json['5136']['fs'])):
            print(data_json['5136']['fs'][i], data_json['5136']['ith'][i], data_json['5136']['sigma'][i])
        print('squeeze rf scan f05606:')
        for i in range(len(data_json['5606']['fs'])):
            print(data_json['5606']['fs'][i], data_json['5606']['ith'][i], data_json['5606']['sigma'][i])

        # shielding
        print(data_json['5110']['shielding'], data_json['5136']['shielding'], data_json['5606']['shielding'])
        print('Max shielding: ', max(max(data_json['5110']['shielding']), max(data_json['5136']['shielding']), max(data_json['5606']['shielding'])), 'Min shielding: ',
              min(min(data_json['5110']['shielding']), min(data_json['5136']['shielding']), min(data_json['5606']['shielding'])))



    # Bunchlänge:
    # f = np.load('/mnt/linux_data/2015_04_30_SD_squeeze-rf-scan_f05605/csv/bunchlength.npy')[()]
    # bunchlength = np.asarray(f['fwhm /ps']) * 1e-12 * scipy.constants.c / 2.35
    # bunchlength_usable = bunchlength[[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]]
    # plt.figure()
    # plt.plot(data_json['5606']['vrf'], ankacalc.sigma(data_json['5606']['fs'], data_json['5606']['vrf']), marker='*')
    # plt.plot(f['V_RF /kV'], bunchlength, marker='o')

    # data_json['5606']['sigma'][:15]=bunchlength_usable

    # plt.plot(data_json['5606']['vrf'], data_json['5606']['sigma'], marker='s')
    # plt.xlabel('voltage / kV')
    # plt.ylabel('bunchlength (rms) / m')
    # plt.show()
    # exit()

    if args.plot1:
        ## I_th over bunch length with theory curves
        fig = plt.figure(figsize=(12, 6), tight_layout=True)
        ax = plt.gca()
        col = matplotlib.cm.rainbow(np.linspace(0, 1, len(args.vrf)))
        # plt.scatter(data_json['5136']['sigma'] * 1e3, data_json['5136']['ith_d'], color='b')
        # plt.scatter(data_json['5136']['sigma'] * 1e3, data_json['5136']['ith'], color='g')
        # plt.scatter(data_json['5110']['sigma'] * 1e3, data_json['5110']['ith_d'], color='k')
        # plt.errorbar(data_json['5110']['sigma'] * 1e3, data_json['5110']['ith'],yerr=data_json['5110']['ith']*0.032, fmt='o',elinewidth=2,color='k',)

        for i, vrf in enumerate(vrf_list):
            marker = ['^', 'o', 's']
            for j, key in enumerate(['5110', '5136', '5606']):
                index = np.where(data_json[key]['vrf'] == vrf)

                plt.errorbar(unumpy.nominal_values(ankacalc.sigma(data_json[key]['fs'][index], data_json[key]['vrf'][index] * ankacalc.voltage_factor())) * 1e3, data_json[key]['ith'][index],
                             xerr=unumpy.std_devs(ankacalc.sigma(data_json[key]['fs'][index], data_json[key]['vrf'][index] * ankacalc.voltage_factor())) * 1e3, yerr=0.0094, fmt='o',
                             elinewidth=2, color=col[i], marker=marker[j], markersize=6)

            plt.plot(unumpy.nominal_values(ankacalc.sigma(fs, vrf * ankacalc.voltage_factor())) * 1e3,
                     unumpy.nominal_values(ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor())) * 1000, color=col[i], linestyle='-', label='%i kV' % (vrf * ankacalc.voltage_factor().nominal_value))
            plt.plot(ankacalc.sigma(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev)) * 1e3,
                     ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i], linestyle=':')
            plt.plot(ankacalc.sigma(fs, vrf * (ankacalc.voltage_factor().nominal_value - ankacalc.voltage_factor().std_dev)) * 1e3,
                     ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i], linestyle=':')

        handles, labels = ax.get_legend_handles_labels()

        # Create custom artists
        theoryArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')
        errorArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-')

        # Create legend from custom artist/label lists
        plt.legend([handle for i, handle in enumerate(handles)] + [theoryArtist, errorArtist],
                   [label for i, label in enumerate(labels)] + ['theory', '1-sigma error band'], loc='upper left', ncol=1, numpoints=1,
                   scatterpoints=1, labelspacing=0.2, frameon=False)

        # plt.legend(loc='upper left', ncol=1)

        plt.xlabel(u'natürliche Bunchlänge / mm')
        plt.xlabel('Calculated natural bunch length / mm')
        plt.xlim(xmin=0.7, xmax=1.6)
        plt.ylim(ymax=0.5, ymin=0.05)
        # plt.ylabel('Burstingschwelle / mA')
        plt.ylabel('Bursting threshold / mA')

        # plt.title('gleichungen von theorie')
        fig.text(0.9, 0.2, 'Preliminary',
                 fontsize=50, color='gray',
                 ha='right', va='bottom', alpha=0.6)
        # plt.savefig('i_th_over_sigma_both.png')


    if args.plot2:
        ## I_th over fs
        # plt.figure(figsize=(12, 6), tight_layout=True)
        plt.figure(figsize=(10, 5), tight_layout=True)

        col = ['blueviolet', 'cyan', 'darkorange', 'red']

        plt.plot(data_json['5136']['fs'][:2], data_json['5136']['ith'][:2], '--', color='k')
        plt.plot(data_json['5136']['fs'][2:5], data_json['5136']['ith'][2:5], '--', color='k')
        plt.plot(data_json['5136']['fs'][5:9], data_json['5136']['ith'][5:9], '--', color='k')
        plt.plot(data_json['5136']['fs'][9:13], data_json['5136']['ith'][9:13], '--', color='k')

        # plt.plot(data_json['5606']['fs'][:4], data_json['5606']['ith'][:4], ':', color='k')
        # plt.plot(data_json['5606']['fs'][4:8], data_json['5606']['ith'][4:8], ':', color='k')
        # plt.plot(data_json['5606']['fs'][8:12], data_json['5606']['ith'][8:12], ':', color='k')
        # plt.plot(data_json['5606']['fs'][12:16], data_json['5606']['ith'][12:16], ':', color='k')
        # plt.plot(data_json['5606']['fs'][16:20], data_json['5606']['ith'][16:20], ':', color='k')

        for i, vrf in enumerate(vrf_list):
            ax = plt.gca()
            marker = ['^', 'o', 's']
            color = ['k', col[i], col[i]]
            for j, key in enumerate(['5110', '5136', '5606']):
                index = np.where(data_json[key]['vrf'] == vrf)

                plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index], xerr=fs_error(data_json[key]['fs'][index]), yerr=0.0094, fmt='o', elinewidth=1, color=color[j], ecolor=col[i], marker=marker[j])

        if True:
            ax = plt.gca()
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            for i, vrf in enumerate(vrf_list):
                index = np.where(data_json['5136']['vrf'] == vrf)

                ##cbs_alpha für spannungs kalibrierung genutzt:
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                         label=r'4$\mathdefault{\times}$ %i kV' % (vrf * ankacalc.voltage_factor().nominal_value))
                # FixMe: not 1 sigma error of all parameters, only vrf considered!
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value - ankacalc.voltage_factor().std_dev)) * 1000,
                         color=col[i], linestyle=':')
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev)) * 1000,
                         color=col[i], linestyle=':')
                print(ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev )) -
                      ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value)* 1000)
                # plt.plot(fs, ankacalc.ith_bunchbeam_sbb_lower(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                #          linestyle='--')


            plt.ylim(ylim)
            plt.xlim(xlim)

        plt.xlim(xmin=4, xmax=14)
        plt.ylim(ymax=0.5, ymin=0.05)

        ### alpha gegeben:
        # for i, step in enumerate(cbs_steps[:-1]):
        #    index = np.where(data_json['5606']['steps'] == step)
        #    #print(index, data_json['5606']['fs'][index])
        #    cbs_sigma = cbs_alpha[i] * scipy.constants.c * e_spread / (2 * np.pi * data_json['5606']['fs'][index] * 1e3)
        #    plt.scatter(data_json['5606']['fs'][index], 1000* vorfaktor * (energy / 511e3) * cbs_alpha[i] * (e_spread) ** 2 * 17045 * (
        #        0.5 * np.power(R, -1.0 / 3.0) * np.power(cbs_sigma, 1.0 / 3.0)
        #        + 0.34 * np.power(R, 1.0 / 6.0) * np.power(cbs_sigma, 4.0 / 3.0)
        #        * np.power(h, -3.0 / 2.0)), color='g', marker='*')

        # plt.legend(loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.2, handletextpad=0.2,
        #            handlelength=0.2, frameon=False, borderpad=0.2, fontsize=19, title='RF voltage')

        handles, labels = ax.get_legend_handles_labels()
        theoryArtist = plt.Line2D((0, 1), (0, 0), color='k')
        measurementArtist1 = plt.Line2D((0, 1), (0, 0), color='k', marker='^', linestyle='')
        measurementArtist2 = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='')
        measurementArtist3 = plt.Line2D((0, 1), (0, 0), color='k', marker='s', linestyle='')
        constOpticArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')

        extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

        l = plt.legend([measurementArtist1, measurementArtist2, measurementArtist3, constOpticArtist, theoryArtist, extra,
                        extra] + handles,
                       ['Measurement 1', 'Measurement 2', 'Measurement 3', 'constant Magnet Optics', 'Theory', ' ',
                        'RF Voltage'] + labels,
                       loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.5, handletextpad=0.5,
                       handlelength=1.5, frameon=False, borderpad=0.5, fontsize=12)  # , title='RF voltage')

        plt.xlabel('Synchrotron frequency / kHz')
        plt.ylabel('Bursting threshold / mA')
        plt.locator_params(axis='y', nbins=6)
        # plt.title('gleichungen von theorie')
        # plt.savefig('i_th_over_fs_both1.png')

        if args.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')

    if args.plot3:
        ## I_th over fs
        plt.figure(figsize=(12, 6), tight_layout=True)
        # plt.figure(figsize=(10, 5), tight_layout=True)

        # col = ['blueviolet', 'cyan', 'darkorange', 'red']
        col = matplotlib.cm.inferno(np.linspace(0, 1, len(args.vrf) + 2))[1:-1]


        plt.plot(data_json['5136']['fs'][:2], data_json['5136']['ith'][:2], '--', color='k')
        plt.plot(data_json['5136']['fs'][2:5], data_json['5136']['ith'][2:5], '--', color='k')
        plt.plot(data_json['5136']['fs'][5:9], data_json['5136']['ith'][5:9], '--', color='k')
        plt.plot(data_json['5136']['fs'][9:13], data_json['5136']['ith'][9:13], '--', color='k')

        # annotation at x (fs) and calculate y (i_th) corresponding to this x ?
        # print(ankacalc.alpha(data_json['5136']['fs'][:2], data_json['5136']['vrf'][:2]*ankacalc.voltage_factor().nominal_value),
        #       ankacalc.alpha(data_json['5136']['fs'][2:5], data_json['5136']['vrf'][2:5]*ankacalc.voltage_factor().nominal_value),
        #       ankacalc.alpha(data_json['5136']['fs'][5:9], data_json['5136']['vrf'][5:9]*ankacalc.voltage_factor().nominal_value),
        #       ankacalc.alpha(data_json['5136']['fs'][9:13], data_json['5136']['vrf'][9:13]*ankacalc.voltage_factor().nominal_value))

        annotation_text_alpha={
            0: np.mean(ankacalc.alpha(data_json['5136']['fs'][:2], data_json['5136']['vrf'][:2]*ankacalc.voltage_factor().nominal_value)),
            1: np.mean(ankacalc.alpha(data_json['5136']['fs'][2:5], data_json['5136']['vrf'][2:5]*ankacalc.voltage_factor().nominal_value)),
            2: np.mean(ankacalc.alpha(data_json['5136']['fs'][5:9], data_json['5136']['vrf'][5:9]*ankacalc.voltage_factor().nominal_value)),
            3: np.mean(ankacalc.alpha(data_json['5136']['fs'][9:13], data_json['5136']['vrf'][9:13]*ankacalc.voltage_factor().nominal_value))}

        # annotation_pos_alpha = {
        #     0: (13, ankacalc.ith_bunchbeam_cai(13., ankacalc.vrf(13.,annotation_text_alpha[0]))*1000),
        #     1: (12.5, ankacalc.ith_bunchbeam_cai(12.5, ankacalc.vrf(12.5,annotation_text_alpha[1])) * 1000),
        #     2: (8., ankacalc.ith_bunchbeam_cai(8., ankacalc.vrf(8,annotation_text_alpha[2])) * 1000),
        #     3: (6., ankacalc.ith_bunchbeam_cai(6., ankacalc.vrf(6,annotation_text_alpha[3])) * 1000)}

        annotation_pos_alpha = {
            0: (np.mean(data_json['5136']['fs'][:2]), np.mean(data_json['5136']['ith'][:2])),
            1: (np.mean(data_json['5136']['fs'][2:4]), np.mean(data_json['5136']['ith'][2:4])),
            2: (np.mean(data_json['5136']['fs'][5:7]), np.mean(data_json['5136']['ith'][5:7])),
            3: (np.mean(data_json['5136']['fs'][11:13]), np.mean(data_json['5136']['ith'][11:13]))}

        annotation_pos_text_alpha = {
            0: (-35, 40),
            1: (90, -20),
            2: (40, 40),
            3: (60, 40)}

        ax = plt.gca()
        for l in range(4):
            print(annotation_text_alpha[l], annotation_pos_alpha[l])
            ax.annotate(r'%.1f$\mathregular{\cdot 10^{-4}}$' % (annotation_text_alpha[l]*10000), xy=annotation_pos_alpha[l], xycoords='data',
                        xytext=annotation_pos_text_alpha[l], textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"),
                        horizontalalignment='right', verticalalignment='top', fontsize=15)

        # plt.plot(data_json['5606']['fs'][:4], data_json['5606']['ith'][:4], ':', color='k')
        # plt.plot(data_json['5606']['fs'][4:8], data_json['5606']['ith'][4:8], ':', color='k')
        # plt.plot(data_json['5606']['fs'][8:12], data_json['5606']['ith'][8:12], ':', color='k')
        # plt.plot(data_json['5606']['fs'][12:16], data_json['5606']['ith'][12:16], ':', color='k')
        # plt.plot(data_json['5606']['fs'][16:20], data_json['5606']['ith'][16:20], ':', color='k')

        for i, vrf in enumerate(vrf_list):
            ax = plt.gca()
            marker = ['^', 'o', 's']
            color = ['k', col[i], col[i]]
            for j, key in enumerate(['5110', '5136', '5606']):
                index = np.where(data_json[key]['vrf'] == vrf)

                plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index],
                             xerr=fs_error(data_json[key]['fs'][index]), yerr=0.0094, fmt='o', elinewidth=1,
                             color=color[j], ecolor=col[i], marker=marker[j])

        if True:
            ax = plt.gca()
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            for i, vrf in enumerate(vrf_list):
                index = np.where(data_json['5136']['vrf'] == vrf)

                ##cbs_alpha für spannungs kalibrierung genutzt:
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                         label=r'4$\mathdefault{\times}$ %i kV' % (vrf * ankacalc.voltage_factor().nominal_value))
                # FixMe: not 1 sigma error of all parameters, only vrf considered!
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value - ankacalc.voltage_factor().std_dev)) * 1000,
                         color=col[i], linestyle=':')
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev)) * 1000,
                         color=col[i], linestyle=':')
                print(ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev )) -
                      ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value)* 1000)
                # plt.plot(fs, ankacalc.ith_bunchbeam_sbb_lower(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                #          linestyle='--')

                #annotation at x (fs=8,11,13,12) and calculate y (i_th) corresponding to this x for each vrf
                annotation_pos = {150: (8, ankacalc.ith_bunchbeam_cai(8., vrf * ankacalc.voltage_factor().nominal_value) * 1000),
                                  300: (11, ankacalc.ith_bunchbeam_cai(11., vrf * ankacalc.voltage_factor().nominal_value) * 1000),
                                  330: (13, ankacalc.ith_bunchbeam_cai(13., vrf * ankacalc.voltage_factor().nominal_value) * 1000),
                                  400: (9.5, ankacalc.ith_bunchbeam_cai(9.5, vrf * ankacalc.voltage_factor().nominal_value) * 1000)}

                annotation_pos_text = {150: (40, -20),
                                       300: (-20, 30),
                                       330: (40, -20),
                                       400: (40, -20)}

                ax.annotate('%i kV' % (vrf * ankacalc.voltage_factor().nominal_value), xy=annotation_pos[vrf], xycoords='data',
                            xytext=annotation_pos_text[vrf], textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", color=col[i]),
                            horizontalalignment='right', verticalalignment='top', fontsize=15, color=col[i])

            plt.ylim(ylim)
            plt.xlim(xlim)



        plt.xlim(xmin=4, xmax=14)
        plt.ylim(ymax=0.5, ymin=0.05)



        handles, labels = ax.get_legend_handles_labels()
        theoryArtist = plt.Line2D((0, 1), (0, 0), color='k')
        measurementArtist1 = plt.Line2D((0, 1), (0, 0), color='k', marker='^', linestyle='')
        measurementArtist2 = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='')
        measurementArtist3 = plt.Line2D((0, 1), (0, 0), color='k', marker='s', linestyle='')
        constOpticArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')

        extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


        l = plt.legend([measurementArtist1, measurementArtist2, measurementArtist3, constOpticArtist, theoryArtist],
                       ['Magnet Sweep', 'Combined Sweep (1)', 'Combined Sweep (2)', 'Constant Magnet Optics', 'Theory'],
                       loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.5, handletextpad=0.5,
                       handlelength=1.5, frameon=False, borderpad=0.5, fontsize=15)  # , title='RF voltage')

        plt.xlabel('Synchrotron Frequency / kHz')
        plt.ylabel('Threshold Current / mA')
        plt.locator_params(axis='y', nbins=6)
        # plt.title('gleichungen von theorie')
        # plt.savefig('/home/miriam/Documents/plots_for_diss/i_th_over_fs_both.png')

        if args.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')

    if args.plot4:
        ## I_th over fs
        plt.figure(figsize=(7, 5), tight_layout=True)
        # plt.figure(figsize=(10, 5), tight_layout=True)

        # col = ['blueviolet', 'cyan', 'darkorange', 'red']
        col = matplotlib.cm.inferno(np.linspace(0, 1, len(args.vrf) + 2))[1:-1]

        plt.plot(data_json['5136']['fs'][:2], data_json['5136']['ith'][:2], '--', color='k')
        plt.plot(data_json['5136']['fs'][2:5], data_json['5136']['ith'][2:5], '--', color='k')
        plt.plot(data_json['5136']['fs'][5:9], data_json['5136']['ith'][5:9], '--', color='k')
        plt.plot(data_json['5136']['fs'][9:13], data_json['5136']['ith'][9:13], '--', color='k')

        # annotation at x (fs) and calculate y (i_th) corresponding to this x ?
        # print(ankacalc.alpha(data_json['5136']['fs'][:2], data_json['5136']['vrf'][:2]*ankacalc.voltage_factor().nominal_value),
        #       ankacalc.alpha(data_json['5136']['fs'][2:5], data_json['5136']['vrf'][2:5]*ankacalc.voltage_factor().nominal_value),
        #       ankacalc.alpha(data_json['5136']['fs'][5:9], data_json['5136']['vrf'][5:9]*ankacalc.voltage_factor().nominal_value),
        #       ankacalc.alpha(data_json['5136']['fs'][9:13], data_json['5136']['vrf'][9:13]*ankacalc.voltage_factor().nominal_value))

        annotation_text_alpha = {
            0: np.mean(ankacalc.alpha(data_json['5136']['fs'][:2],
                                      data_json['5136']['vrf'][:2] * ankacalc.voltage_factor().nominal_value)),
            1: np.mean(ankacalc.alpha(data_json['5136']['fs'][2:5],
                                      data_json['5136']['vrf'][2:5] * ankacalc.voltage_factor().nominal_value)),
            2: np.mean(ankacalc.alpha(data_json['5136']['fs'][5:9],
                                      data_json['5136']['vrf'][5:9] * ankacalc.voltage_factor().nominal_value)),
            3: np.mean(ankacalc.alpha(data_json['5136']['fs'][9:13],
                                      data_json['5136']['vrf'][9:13] * ankacalc.voltage_factor().nominal_value))}

        # annotation_pos_alpha = {
        #     0: (13, ankacalc.ith_bunchbeam_cai(13., ankacalc.vrf(13.,annotation_text_alpha[0]))*1000),
        #     1: (12.5, ankacalc.ith_bunchbeam_cai(12.5, ankacalc.vrf(12.5,annotation_text_alpha[1])) * 1000),
        #     2: (8., ankacalc.ith_bunchbeam_cai(8., ankacalc.vrf(8,annotation_text_alpha[2])) * 1000),
        #     3: (6., ankacalc.ith_bunchbeam_cai(6., ankacalc.vrf(6,annotation_text_alpha[3])) * 1000)}

        annotation_pos_alpha = {
            0: (np.mean(data_json['5136']['fs'][:2]), np.mean(data_json['5136']['ith'][:2])),
            1: (np.mean(data_json['5136']['fs'][2:4]), np.mean(data_json['5136']['ith'][2:4])),
            2: (np.mean(data_json['5136']['fs'][5:7]), np.mean(data_json['5136']['ith'][5:7])),
            3: (np.mean(data_json['5136']['fs'][11:13]), np.mean(data_json['5136']['ith'][11:13]))}

        annotation_pos_text_alpha = {
            0: (-10, 30),
            1: (50, -20),
            2: (40, 40),
            3: (60, 40)}

        ax = plt.gca()
        for l in range(4):
            print(annotation_text_alpha[l], annotation_pos_alpha[l])
            ax.annotate(r'%.1f$\mathregular{\cdot 10^{-4}}$' % (annotation_text_alpha[l] * 10000),
                        xy=annotation_pos_alpha[l], xycoords='data',
                        xytext=annotation_pos_text_alpha[l], textcoords='offset points',
                        arrowprops=dict(arrowstyle="->"),
                        horizontalalignment='right', verticalalignment='top', fontsize=16)

        # plt.plot(data_json['5606']['fs'][:4], data_json['5606']['ith'][:4], ':', color='k')
        # plt.plot(data_json['5606']['fs'][4:8], data_json['5606']['ith'][4:8], ':', color='k')
        # plt.plot(data_json['5606']['fs'][8:12], data_json['5606']['ith'][8:12], ':', color='k')
        # plt.plot(data_json['5606']['fs'][12:16], data_json['5606']['ith'][12:16], ':', color='k')
        # plt.plot(data_json['5606']['fs'][16:20], data_json['5606']['ith'][16:20], ':', color='k')
        rflabelartist=list()
        rflabelartisthandle = list()
        for i, vrf in enumerate(vrf_list):
            ax = plt.gca()
            marker = ['^', 'o', 's']
            color = ['k', col[i], col[i]]
            for j, key in enumerate(['5110', '5136']):
                index = np.where(data_json[key]['vrf'] == vrf)

                plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index],
                             xerr=fs_error(data_json[key]['fs'][index]), yerr=0.0094, fmt='o', elinewidth=1,
                             color=color[j], ecolor=col[i], marker=marker[j])

            rflabelartist.append(plt.Line2D((0, 1), (0, 0), color=col[i], marker='o', linestyle=''))
            rflabelartisthandle.append(vrf * ankacalc.voltage_factor().nominal_value)

        plt.xlim(xmin=4, xmax=14)
        plt.ylim(ymax=0.5, ymin=0.05)

        handles, labels = ax.get_legend_handles_labels()
        measurementArtist1 = plt.Line2D((0, 1), (0, 0), color='k', marker='^', linestyle='')
        measurementArtist2 = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='')
        constOpticArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')

        extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

        l = plt.legend([measurementArtist1, measurementArtist2, constOpticArtist],
                       ['Magnet Sweep', 'Combined Sweep', 'Constant Magnet Optics'],
                       loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.3, handletextpad=0.2,
                       handlelength=1.5, frameon=False, borderpad=0.5, fontsize=16)  # , title='RF voltage')

        l2 = plt.legend(rflabelartist, ['%i kV' % i for i in rflabelartisthandle],
                       loc='upper left', bbox_to_anchor=(0,0.76), ncol=2, numpoints=1, scatterpoints=1, labelspacing=0.3, handletextpad=0.1,
                       handlelength=1.5, frameon=False, borderpad=0.5, fontsize=16, columnspacing=0)  # , title='RF voltage')
        plt.gca().add_artist(l)

        plt.xlabel('Synchrotron Frequency / kHz')
        plt.ylabel('Threshold Current / mA')
        plt.locator_params(axis='y', nbins=6)
        # plt.title('gleichungen von theorie')
        # plt.savefig('/home/miriam/Documents/plots_for_diss/i_th_over_fs_small.pdf')

        if args.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')


    if args.plot5:
        ## I_th over fs
        plt.figure(figsize=(12, 6), tight_layout=True)
        # plt.figure(figsize=(10, 5), tight_layout=True)

        # col = ['blueviolet', 'cyan', 'darkorange', 'red']
        col = matplotlib.cm.inferno(np.linspace(0, 1, len(args.vrf) + 2))[1:-1]

        rflabelartist = list()
        rflabelartisthandle = list()

        for i, vrf in enumerate(vrf_list):
            ax = plt.gca()
            marker = ['^', 'o', 's']
            color = ['k', col[i], col[i]]
            for j, key in enumerate(['5110', '5136', '5606']):
                index = np.where(data_json[key]['vrf'] == vrf)

                plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index],
                             xerr=fs_error(data_json[key]['fs'][index]), yerr=0.0094, fmt='o', elinewidth=1,
                             color=color[j], ecolor=col[i], marker=marker[j])

            rflabelartist.append(plt.Line2D((0, 1), (0, 0), color=col[i], marker='o', linestyle=''))
            rflabelartisthandle.append(vrf * ankacalc.voltage_factor().nominal_value)

        if True:
            ax = plt.gca()
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            for i, vrf in enumerate(vrf_list):
                index = np.where(data_json['5136']['vrf'] == vrf)

                ##cbs_alpha für spannungs kalibrierung genutzt:
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                         label=r'4$\mathdefault{\times}$ %i kV' % (vrf * ankacalc.voltage_factor().nominal_value))
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * 4) * 1000,
                         color=col[i], label=r'4$\mathdefault{\times}$ %i kV' % (vrf * 4), linestyle='--')
                # FixMe: not 1 sigma error of all parameters, only vrf considered!
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value - ankacalc.voltage_factor().std_dev)) * 1000,
                         color=col[i], linestyle=':')
                plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev)) * 1000,
                         color=col[i], linestyle=':')
                print(ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev )) -
                      ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value)* 1000)
                # plt.plot(fs, ankacalc.ith_bunchbeam_sbb_lower(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                #          linestyle='--')

            plt.ylim(ylim)
            plt.xlim(xlim)

        plt.xlim(xmin=4, xmax=14)
        plt.ylim(ymax=0.5, ymin=0.05)

        handles, labels = ax.get_legend_handles_labels()
        theoryArtist = plt.Line2D((0, 1), (0, 0), color='k')
        theoryArtist2 = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')
        # measurementArtist1 = plt.Line2D((0, 1), (0, 0), color='k', marker='^', linestyle='')
        # measurementArtist2 = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='')
        # measurementArtist3 = plt.Line2D((0, 1), (0, 0), color='k', marker='s', linestyle='')
        # constOpticArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')

        extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


        l = plt.legend([theoryArtist, theoryArtist2],
                       ['Theory (with calibrated RF voltage)', 'Theory (with uncalibrated\nRF voltage set-values)'],
                       loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.5, handletextpad=0.5,
                       handlelength=1.5, frameon=False, borderpad=0.5, fontsize=15)  # , title='RF voltage')
        l2 = plt.legend(rflabelartist, ['%i kV' % i for i in rflabelartisthandle],
                        loc='upper left', bbox_to_anchor=(0, 0.815), ncol=2, numpoints=1, scatterpoints=1,
                        labelspacing=0.3, handletextpad=0.4,
                        handlelength=1.5, frameon=False, borderpad=0.5, fontsize=16,
                        columnspacing=0)
        l2.set_title(title='Measurements:', prop={'size': 16})
        l2._legend_box.align = "left"
        plt.gca().add_artist(l)

        plt.xlabel('Synchrotron Frequency / kHz')
        plt.ylabel('Threshold Current / mA')
        plt.locator_params(axis='y', nbins=6)
        # plt.title('gleichungen von theorie')
        # plt.savefig('/home/miriam/Documents/plots_for_diss/i_th_over_fs_with_uncalibrated.png')

        if args.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')



    if args.plot6:
        ## I_th over fs
        plt.figure(figsize=(12, 6), tight_layout=True)
        # plt.figure(figsize=(10, 5), tight_layout=True)

        # col = ['blueviolet', 'cyan', 'darkorange', 'red']
        col = matplotlib.cm.inferno(np.linspace(0, 1, len(args.vrf) + 2))[1:-1]


        # plt.plot(data_json['5136']['fs'][:2], data_json['5136']['ith'][:2], '--', color='k')
        # plt.plot(data_json['5136']['fs'][2:5], data_json['5136']['ith'][2:5], '--', color='k')
        # plt.plot(data_json['5136']['fs'][5:9], data_json['5136']['ith'][5:9], '--', color='k')
        # plt.plot(data_json['5136']['fs'][9:13], data_json['5136']['ith'][9:13], '--', color='k')
        # annotation_text_alpha={
        #     0: np.mean(ankacalc.alpha(data_json['5136']['fs'][:2], data_json['5136']['vrf'][:2]*ankacalc.voltage_factor().nominal_value)),
        #     1: np.mean(ankacalc.alpha(data_json['5136']['fs'][2:5], data_json['5136']['vrf'][2:5]*ankacalc.voltage_factor().nominal_value)),
        #     2: np.mean(ankacalc.alpha(data_json['5136']['fs'][5:9], data_json['5136']['vrf'][5:9]*ankacalc.voltage_factor().nominal_value)),
        #     3: np.mean(ankacalc.alpha(data_json['5136']['fs'][9:13], data_json['5136']['vrf'][9:13]*ankacalc.voltage_factor().nominal_value))}
        #
        #
        # annotation_pos_alpha = {
        #     0: (np.mean(data_json['5136']['fs'][:2]), np.mean(data_json['5136']['ith'][:2])),
        #     1: (np.mean(data_json['5136']['fs'][2:4]), np.mean(data_json['5136']['ith'][2:4])),
        #     2: (np.mean(data_json['5136']['fs'][5:7]), np.mean(data_json['5136']['ith'][5:7])),
        #     3: (np.mean(data_json['5136']['fs'][11:13]), np.mean(data_json['5136']['ith'][11:13]))}
        #
        # annotation_pos_text_alpha = {
        #     0: (-35, 40),
        #     1: (90, -20),
        #     2: (40, 40),
        #     3: (60, 40)}

        # ax = plt.gca()
        # for l in range(4):
        #     print(annotation_text_alpha[l], annotation_pos_alpha[l])
        #     ax.annotate(r'%.1f$\mathregular{\cdot 10^{-4}}$' % (annotation_text_alpha[l]*10000), xy=annotation_pos_alpha[l], xycoords='data',
        #                 xytext=annotation_pos_text_alpha[l], textcoords='offset points',
        #                 arrowprops=dict(arrowstyle="->"),
        #                 horizontalalignment='right', verticalalignment='top', fontsize=15)

        for i, vrf in enumerate(vrf_list):
            ax = plt.gca()
            marker = ['^', 'o', 's']
            color = ['k', col[i], col[i]]
            for j, key in enumerate(['5110', '5136', '5606']):
                index = np.where(data_json[key]['vrf'] == vrf)

                plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index],
                             xerr=fs_error(data_json[key]['fs'][index]), yerr=0.0094, fmt='o', elinewidth=1,
                             color=color[j], ecolor=col[i], marker=marker[j])

        if True:
            ax = plt.gca()
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            for i, vrf in enumerate(vrf_list):
                index = np.where(data_json['5136']['vrf'] == vrf)

                ##cbs_alpha für spannungs kalibrierung genutzt:
                # plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                #          label=r'4$\mathdefault{\times}$ %i kV' % (vrf * ankacalc.voltage_factor().nominal_value))
                # FixMe: not 1 sigma error of all parameters, only vrf considered!
                # plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value - ankacalc.voltage_factor().std_dev)) * 1000,
                #          color=col[i], linestyle=':')
                # plt.plot(fs, ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev)) * 1000,
                #          color=col[i], linestyle=':')
                print(ankacalc.ith_bunchbeam_cai(fs, vrf * (ankacalc.voltage_factor().nominal_value + ankacalc.voltage_factor().std_dev )) -
                      ankacalc.ith_bunchbeam_cai(fs, vrf * ankacalc.voltage_factor().nominal_value)* 1000)
                # plt.plot(fs, ankacalc.ith_bunchbeam_sbb_lower(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                #          linestyle='--')

                #annotation at x (fs=8,11,13,12) and calculate y (i_th) corresponding to this x for each vrf
                annotation_pos = {150: (8, ankacalc.ith_bunchbeam_cai(8., vrf * ankacalc.voltage_factor().nominal_value) * 1000),
                                  300: (11, ankacalc.ith_bunchbeam_cai(11., vrf * ankacalc.voltage_factor().nominal_value) * 1000),
                                  330: (13, ankacalc.ith_bunchbeam_cai(13., vrf * ankacalc.voltage_factor().nominal_value) * 1000),
                                  400: (9.5, ankacalc.ith_bunchbeam_cai(9.5, vrf * ankacalc.voltage_factor().nominal_value) * 1000)}

                annotation_pos_text = {150: (40, -20),
                                       300: (-20, 30),
                                       330: (40, -20),
                                       400: (40, -20)}

                # ax.annotate('%i kV' % (vrf * ankacalc.voltage_factor().nominal_value), xy=annotation_pos[vrf], xycoords='data',
                #             xytext=annotation_pos_text[vrf], textcoords='offset points',
                #             arrowprops=dict(arrowstyle="->", color=col[i]),
                #             horizontalalignment='right', verticalalignment='top', fontsize=15, color=col[i])
                ax.annotate('%i kV' % (vrf * ankacalc.voltage_factor().nominal_value), xy=annotation_pos[vrf], xycoords='data',
                            xytext=annotation_pos_text[vrf], textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", color='w'),
                            horizontalalignment='right', verticalalignment='top', fontsize=15, color=col[i])

            plt.ylim(ylim)
            plt.xlim(xlim)



        plt.xlim(xmin=4, xmax=14)
        plt.ylim(ymax=0.5, ymin=0.05)



        handles, labels = ax.get_legend_handles_labels()
        theoryArtist = plt.Line2D((0, 1), (0, 0), color='k')
        measurementArtist1 = plt.Line2D((0, 1), (0, 0), color='k', marker='^', linestyle='')
        measurementArtist2 = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='')
        measurementArtist3 = plt.Line2D((0, 1), (0, 0), color='k', marker='s', linestyle='')
        constOpticArtist = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')

        extra = matplotlib.patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)


        # l = plt.legend([measurementArtist1, measurementArtist2, measurementArtist3, constOpticArtist, theoryArtist],
        #                ['Magnet Sweep', 'Combined Sweep (1)', 'Combined Sweep (2)', 'Constant Magnet Optics', 'Theory'],
        #                loc='upper left', ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.5, handletextpad=0.5,
        #                handlelength=1.5, frameon=False, borderpad=0.5, fontsize=15)  # , title='RF voltage')

        # plt.xlabel('Synchrotron Frequency / kHz')
        plt.xlabel('Synchrotronfrequenz / kHz')
        # plt.ylabel('Threshold Current / mA')
        plt.ylabel('Schwellstrom / mA')
        plt.locator_params(axis='y', nbins=6)
        # plt.title('gleichungen von theorie')
        # plt.savefig('/home/miriam/Documents/plots_for_diss/Verteidigung/i_th_over_fs_both_talk-only-meas.png')
        plt.savefig('/home/miriam/Documents/plots_for_diss/Verteidigung/i_th_over_fs_both_talk-only-meas_german.png')
        # plt.savefig('/home/miriam/Documents/plots_for_diss/Verteidigung/i_th_over_fs_both_talk.png')
        # plt.savefig('/home/miriam/Documents/plots_for_diss/Verteidigung/i_th_over_fs_both_talk_german.png')

        if args.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')

    plt.show()
