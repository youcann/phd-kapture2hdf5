#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import matplotlib
from uncertainties import unumpy
import i_th_fit_ankacalc as ankacalc
import json
import yaml


# write sbb parameters to file for Peter K.:
def data_write_to_file():
    font = {'size': 19}
    matplotlib.rc('font', **font)
    import matplotlib.pyplot as plt

    fill=[]
    squeeze=np.array([])
    vrf_all=np.array([])
    vrf_all_error=np.array([])
    fs_all=np.array([])
    ith_all=np.array([])
    ith_all_error=np.array([])
    ith_upper_all=np.array([])
    ith_upper_all_error=np.array([])
    ith_lower_all=np.array([])
    ith_lower_all_error=np.array([])

    keys = data_json.keys()
    keys.sort()

    for key in keys:
        fill.extend([int(key.split('_')[0]), ] * len(data_json[key]['fs']))
        squeeze = np.hstack((squeeze,data_json[key]['steps'] * 1e3))
        vrf_all = np.hstack((vrf_all, data_json[key]['vrf'] * ankacalc.voltage_factor().nominal_value if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf']))
        vrf_all_error = np.hstack((vrf_all_error, data_json[key]['vrf'] * ankacalc.voltage_factor().std_dev if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf'] * 0.01))
        fs_all = np.hstack((fs_all, data_json[key]['fs']))
        ith_all = np.hstack((ith_all, data_json[key]['ith']))
        ith_all_error = np.hstack((ith_all_error, ith_error(data_json[key]['ith']) if 'ith_error' not in data_json[key] else data_json[key]['ith_error']))
        ith_upper_all = np.hstack((ith_upper_all,np.array([0, ]*len(data_json[key]['fs'])) if 'ith_upper' not in data_json[key] else data_json[key]['ith_upper']))
        ith_upper_all_error = np.hstack((ith_upper_all_error,np.array([0, ]*len(data_json[key]['fs'])) if 'ith_upper_error' not in data_json[key] else data_json[key]['ith_upper_error']))
        ith_lower_all = np.hstack((ith_lower_all,np.array([0, ]*len(data_json[key]['fs'])) if 'ith_lower' not in data_json[key] else data_json[key]['ith_lower']))
        ith_lower_all_error = np.hstack((ith_lower_all_error,np.array([0, ]*len(data_json[key]['fs'])) if 'ith_lower_error' not in data_json[key] else data_json[key]['ith_lower_error']))

    shielding_all = ankacalc.shielding_bane(fs_all, vrf_all)
    scsr_all = ankacalc.S_csr(ith_all / 1000., fs_all, vrf_all)
    scsr_upper_all = ankacalc.S_csr(ith_upper_all / 1000., fs_all, vrf_all)
    scsr_lower_all = ankacalc.S_csr(ith_lower_all / 1000., fs_all, vrf_all)

    shielding_all_error = unumpy.std_devs(ankacalc.shielding_bane(unumpy.uarray(fs_all,ankacalc.fs_error(fs_all)),unumpy.uarray(vrf_all, vrf_all_error)))
    scsr_all_error = unumpy.std_devs(ankacalc.S_csr(unumpy.uarray(ith_all, ith_all * np.sqrt((ith_all_error / ith_all) ** 2 + rel_current_error ** 2)) / 1000., unumpy.uarray(fs_all, ankacalc.fs_error(fs_all)), unumpy.uarray(vrf_all, vrf_all_error)))
    scsr_upper_all_error = unumpy.std_devs(ankacalc.S_csr(unumpy.uarray(ith_upper_all, ith_upper_all * np.sqrt((ith_upper_all_error / ith_upper_all) ** 2 + rel_current_error ** 2)) / 1000., unumpy.uarray(fs_all, ankacalc.fs_error(fs_all)), unumpy.uarray(vrf_all, vrf_all_error)))
    scsr_lower_all_error = unumpy.std_devs(ankacalc.S_csr(unumpy.uarray(ith_lower_all, ith_lower_all * np.sqrt((ith_lower_all_error / ith_lower_all) ** 2 + rel_current_error ** 2)) / 1000., unumpy.uarray(fs_all, ankacalc.fs_error(fs_all)), unumpy.uarray(vrf_all, vrf_all_error)))

    a = np.asarray([fill, squeeze, vrf_all, fs_all, ith_all, ith_upper_all, ith_lower_all, shielding_all, scsr_all, scsr_upper_all, scsr_lower_all, shielding_all_error,scsr_all_error,scsr_upper_all_error, scsr_lower_all_error])
    fmt = ",".join(["%s"] + ["%10.3e"] * (a.T.shape[1] - 1))
    np.savetxt("parameter_threshold_peterk_test.csv", a.T, delimiter=",",
               header='fillnumber,squeeze step,voltage/kV,fs/kHz,threshold/mA,upper threshold sbb/mA, lower threshold sbb/mA,shielding_all, scsr_all, scsr_upper_all, scsr_lower_all, shielding_all_error,scsr_all_error,scsr_upper_all_error, scsr_lower_all_error',
               fmt=fmt)


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
    parser.add_argument('--vrf', type=int, nargs='+')
    parser.add_argument('--loglog', action='store_true')
    parser.add_argument('--save-data', action='store_true')
    parser.add_argument('--plot1', action='store_true')
    parser.add_argument('--plot2', action='store_true')

    args = parser.parse_args()

    font = {'size': 19}
    matplotlib.rc('font', **font)
    # matplotlib.use('PDF')
    import matplotlib.pyplot as plt

    ## Data:
    # import von custom json is possible mit json and yaml!!
    with open("threshold_data_incl_sbb_new_json.json", 'r') as f:
        data_json = json.loads(f.read())
    data_json = parse_data_after_readin(data_json)

    # with open("threshold_data_incl_sbb_new_json.json", 'r') as stream:
    #     try:
    #         data_json = yaml.safe_load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)


    # drop last entry, which shows sbb
    # fs_5606 = np.delete(fs_5606, [19])
    data_json['5606']['fs'] = data_json['5606']['fs'][:-1]
    data_json['5606']['vrf'] = data_json['5606']['vrf'][:-1]
    data_json['5606']['ith'] = data_json['5606']['ith'][:-1]
    data_json['5606']['steps'] = data_json['5606']['steps'][:-1]


    if args.vrf:
        vrf_list = np.asarray(args.vrf)
    else:
        vrf_list = list(set([vrf for key in data_json for vrf in data_json[key]['vrf']]))
        print(vrf_list)

    fs = np.linspace(-0.5, 20, 1000)

    # rel_current_error = 0.001 # 1permille if fitted current
    rel_current_error = 0.01  # 1 percent if interpolated current

    def ith_error(i):
        return i * 0.032
        return 0.0094


    if args.save_data:
        data_write_to_file()
        exit()


    ### Plots

    if args.plot1:
        if not args.vrf:
            print('Please specify values for vrf to be displayed.')
            exit()
        ## I_th over fs
        # plt.figure(figsize=(12, 6), tight_layout=True)
        plt.figure(figsize=(10, 5), tight_layout=True)

        # plt.errorbar(sigma_m_d, ith_ab_5110,yerr=ith_ab_5110*0.032, fmt='o',elinewidth=2,color='k',)
        # plt.errorbar(fs_5110, ith_ab_5110, yerr=0.0094, fmt='o', elinewidth=2, color='k', label=r'4$\mathdefault{\times}$ %i kV' % (300*u_v_cal_factor.nominal_value))
        # plt.errorbar(fs_5110, ith_ab_5110, yerr=0.0094, fmt='o', elinewidth=0.1, color='k')

        col = matplotlib.cm.rainbow(np.linspace(0, 1, len(args.vrf)))


        # col = ['blueviolet', 'cyan', 'darkorange', 'red', 'yellow']
        # import itertools
        # marker = itertools.cycle(('v', 'o', 's', 'D'))
        #
        for i, vrf in enumerate(vrf_list):
            for key in ['5894', '5864', '5865']:
                index = np.where(data_json[key]['vrf'] == vrf)
                for th, marker in zip(['ith', 'ith_upper', 'ith_lower'],['^','s','o']):
                    if th in data_json[key]:
                        plt.errorbar(data_json[key]['fs'][index], data_json[key][th][index],
                                     xerr=ankacalc.fs_error(data_json[key]['fs'][index]),
                                     yerr=data_json[key][th][index] * np.sqrt((data_json[key]['ith_error'][index] /
                                                                                  data_json[key][th][
                                                                                      index]) ** 2 + rel_current_error ** 2),
                                     fmt='o',
                                     elinewidth=1,
                                     color=col[i], ecolor=col[i],
                                     marker=marker)

            for key in ['5894_no','5110','5136','5606']:
                index = np.where(data_json[key]['vrf'] == vrf)
                ith_error_tmp = data_json[key]['ith'][index] * \
                                np.sqrt(((ith_error(data_json[key]['ith'][index]) if 'ith_error' not in data_json[key] else data_json[key]['ith_error'][index])
                                         / data_json[key]['ith'][index]) ** 2 + rel_current_error ** 2)
                plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index],
                             xerr=ankacalc.fs_error(data_json[key]['fs'][index]),
                             yerr=ith_error_tmp,
                             fmt='o',
                             elinewidth=1,
                             color=col[i], ecolor=col[i],
                             marker='^')
            for key in ['5627','5629','5740']:
                index = np.where(data_json[key]['vrf'] == vrf)
                ith_error_tmp = data_json[key]['ith'][index] * \
                                np.sqrt(((ith_error(data_json[key]['ith'][index]) if 'ith_error' not in data_json[key] else data_json[key]['ith_error'][index])
                                         / data_json[key]['ith'][index]) ** 2 + rel_current_error ** 2)
                plt.errorbar(data_json[key]['fs'][index], data_json[key]['ith'][index],
                             xerr=ankacalc.fs_error(data_json[key]['fs'][index]),
                             yerr=ith_error_tmp,
                             fmt='o',
                             elinewidth=1,
                             color='k', ecolor=col[i],
                             marker='*')

        if True:
            ax = plt.gca()
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            for i, vrf in enumerate(vrf_list):
                if vrf in data_json['5894']['vrf'] or vrf in data_json['5864']['vrf']:
                    y = np.ma.masked_less(ankacalc.ith_bunchbeam_bane(fs, vrf) * 1000, 0.042)

                    plt.plot(fs, y, color=col[i], ls='-', label='%i' % vrf)
                    # plt.plot(fs, ith_bunchbeam_sbb_upper(fs, vrf) * 1000, color=col[i], ls='--')
                    # plt.plot(fs, ith_bunchbeam_sbb_lower(fs, vrf) * 1000, color=col[i], ls=':')
                elif vrf in data_json['5136']['vrf']:
                    plt.plot(fs, ankacalc.ith_bunchbeam_bane(fs, vrf * ankacalc.voltage_factor().nominal_value) * 1000, color=col[i],
                             label=r'4$\mathdefault{\times}$ %i kV' % (vrf * ankacalc.voltage_factor().nominal_value))

            plt.ylim(ylim)
            plt.xlim(xlim)

        plt.xlim(xmin=4.9, xmax=8.4)
        plt.ylim(ymax=0.083, ymin=0.005)
        from matplotlib.patches import Rectangle
        someX, someY = 0, 0.042
        width, hight = 9, 0.5
        currentAxis = plt.gca()
        # currentAxis.add_patch(Rectangle((someX, someY), width, hight, facecolor="none", edgecolor='none', hatch='/'))
        someX, someY = 0, 0.022+0.0066
        width, hight = 9, 0.02-0.0066
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((someX, someY), width, hight, facecolor="none", edgecolor='grey', hatch='/'))
        someX, someY = 0, 0
        width, hight = 9, 0.022
        currentAxis = plt.gca()
        currentAxis.add_patch(Rectangle((someX, someY), width, hight, facecolor="none", edgecolor='grey', hatch='\ '))

        plt.xlabel('Synchrotron frequency / kHz')
        plt.ylabel('Bursting threshold / mA')
        plt.locator_params(axis='y', nbins=6)
        plt.legend(loc='upper left', ncol=2, numpoints=1, scatterpoints=1, labelspacing=0.5, handletextpad=0.5,
                       handlelength=0.7, frameon=False, borderpad=0.5, fontsize=17, title='RF voltage / kV')
        # plt.title('gleichungen von theorie')
        # plt.savefig('i_th_over_fs.pdf')

        if args.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')



    if args.plot2:
        ## S_csr over shielding
        # col = ['blueviolet', 'cyan', 'darkorange', 'red', 'yellow']
        # col = matplotlib.cm.rainbow(np.linspace(0, 1, len(args.vrf)))
        # plt.figure(figsize=(15, 6), tight_layout=True)
        plt.figure(figsize=(12, 6), tight_layout=True)

        x2 = list()
        y2_upper = list()
        y2_lower = list()

        x3 = list()
        y3 = list()
        for i, vrf in enumerate(vrf_list):

            for key in ['5894', '5864', '5865']:
                index = np.where(data_json[key]['vrf'] == vrf)
                ax = plt.gca()

                plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], data_json[key]['vrf'][index]),
                             ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index],
                                            data_json[key]['vrf'][index]),
                             xerr=unumpy.std_devs(
                                 ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
                                                                       ankacalc.fs_error(data_json[key]['fs'][index])),
                                                         unumpy.uarray(data_json[key]['vrf'][index],
                                                                       data_json[key]['vrf'][index] * 0.01))),
                             yerr=unumpy.std_devs(ankacalc.S_csr(
                                 unumpy.uarray(data_json[key]['ith'][index], data_json[key]['ith'][index] * np.sqrt(
                                     (data_json[key]['ith_error'][index] / data_json[key]['ith'][
                                         index]) ** 2 + rel_current_error ** 2)) / 1000.,
                                 unumpy.uarray(data_json[key]['fs'][index],
                                               ankacalc.fs_error(data_json[key]['fs'][index])),
                                 unumpy.uarray(data_json[key]['vrf'][index], data_json[key]['vrf'][index] * 0.01))),
                             fmt='o', elinewidth=1,
                             color='r',  # color=col[i],
                             marker='s', markersize='6')  # , label='standard bursting threshold (with SBB)')
                x3.extend(ankacalc.shielding_bane(data_json[key]['fs'][index], data_json[key]['vrf'][index]))
                y3.extend(ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index],
                                         data_json[key]['vrf'][index]))
                print(key,vrf,data_json[key]['fs'][index], ankacalc.shielding_bane(data_json[key]['fs'][index], data_json[key]['vrf'][index]))

                plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], data_json[key]['vrf'][index]),
                             ankacalc.S_csr(data_json[key]['ith_upper'][index] / 1000., data_json[key]['fs'][index],
                                            data_json[key]['vrf'][index]),
                             xerr=unumpy.std_devs(
                                 ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
                                                                       ankacalc.fs_error(data_json[key]['fs'][index])),
                                                         unumpy.uarray(data_json[key]['vrf'][index],
                                                                       data_json[key]['vrf'][index] * 0.01))),
                             yerr=unumpy.std_devs(
                                 ankacalc.S_csr(unumpy.uarray(data_json[key]['ith_upper'][index],
                                                              data_json[key]['ith_upper'][index] * np.sqrt(
                                                                  (data_json[key]['ith_upper_error'][index] /
                                                                   data_json[key]['ith_upper'][
                                                                       index]) ** 2 + rel_current_error ** 2)) / 1000.,
                                                unumpy.uarray(data_json[key]['fs'][index],
                                                              ankacalc.fs_error(data_json[key]['fs'][index])),
                                                unumpy.uarray(data_json[key]['vrf'][index],
                                                              data_json[key]['vrf'][index] * 0.01))),
                             fmt='o', elinewidth=1,
                             color='b',  # color=col[i],
                             marker='^', markersize='6')  # , label = 'upper end of SBB')
                plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], data_json[key]['vrf'][index]),
                             ankacalc.S_csr(data_json[key]['ith_lower'][index] / 1000., data_json[key]['fs'][index],
                                            data_json[key]['vrf'][index]),
                             xerr=unumpy.std_devs(
                                 ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
                                                                       ankacalc.fs_error(data_json[key]['fs'][index])),
                                                         unumpy.uarray(data_json[key]['vrf'][index],
                                                                       data_json[key]['vrf'][index] * 0.01))),
                             yerr=unumpy.std_devs(
                                 ankacalc.S_csr(unumpy.uarray(data_json[key]['ith_lower'][index],
                                                              data_json[key]['ith_lower'][index] * np.sqrt(
                                                                  (data_json[key]['ith_lower_error'][index] /
                                                                   data_json[key]['ith_lower'][
                                                                       index]) ** 2 + rel_current_error ** 2)) / 1000.,
                                                unumpy.uarray(data_json[key]['fs'][index],
                                                              ankacalc.fs_error(data_json[key]['fs'][index])),
                                                unumpy.uarray(data_json[key]['vrf'][index],
                                                              data_json[key]['vrf'][index] * 0.01))),
                             fmt='o', elinewidth=1,
                             color='orange',  # color=col[i],
                             marker='o', markersize='6')  # , label = 'lower end of SBB')
                x2.extend(ankacalc.shielding_bane(data_json[key]['fs'][index], data_json[key]['vrf'][index]))
                y2_upper.extend(ankacalc.S_csr(data_json[key]['ith_upper'][index] / 1000., data_json[key]['fs'][index],
                                               data_json[key]['vrf'][index]))
                y2_lower.extend(ankacalc.S_csr(data_json[key]['ith_lower'][index] / 1000., data_json[key]['fs'][index],
                                               data_json[key]['vrf'][index]))

            for key in ['5894_no', '5110', '5136', '5606']:
                index = np.where(data_json[key]['vrf'] == vrf)
                vrf_tmp = data_json[key]['vrf'][index] * ankacalc.voltage_factor().nominal_value \
                    if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf'][index]
                vrf_tmp_error = data_json[key]['vrf'][index] * ankacalc.voltage_factor().std_dev \
                    if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf'][index] * 0.01
                ith_error_tmp = data_json[key]['ith'][index] * \
                                np.sqrt(((ith_error(data_json[key]['ith'][index]) if 'ith_error' not in data_json[key] else data_json[key]['ith_error'][index])
                                         / data_json[key]['ith'][index]) ** 2 + rel_current_error ** 2)

                plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp),
                             ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index], vrf_tmp),
                             xerr=unumpy.std_devs(
                                 ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
                                                                       ankacalc.fs_error(data_json[key]['fs'][index])),
                                                         unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             yerr=unumpy.std_devs(ankacalc.S_csr(
                                 unumpy.uarray(data_json[key]['ith'][index],ith_error_tmp) / 1000.,
                                 unumpy.uarray(data_json[key]['fs'][index],
                                               ankacalc.fs_error(data_json[key]['fs'][index])),
                                 unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             color='g',  # color=col[i],
                             fmt='o',
                             elinewidth=1, markersize='6', marker='d')
                x3.extend(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp))
                y3.extend(ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index], vrf_tmp))

                print(key,vrf,data_json[key]['fs'][index], ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp))


            #Single bunch decays
            for key in ['5627', '5740', '5744', '5781', '5789', '5796', '5896', '5904', '5905', '6053']: #5741 was with rf exitation! not used here atm, #'5629' MB decay same settings as 5627
                index = np.where(data_json[key]['vrf'] == vrf)
                vrf_tmp = data_json[key]['vrf'][index] * ankacalc.voltage_factor().nominal_value \
                    if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf'][index]
                vrf_tmp_error = data_json[key]['vrf'][index] * ankacalc.voltage_factor().std_dev \
                    if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf'][index] * 0.01
                ith_error_tmp = data_json[key]['ith'][index] * \
                                np.sqrt(((ith_error(data_json[key]['ith'][index]) if 'ith_error' not in data_json[key] else data_json[key]['ith_error'][index])
                                         / data_json[key]['ith'][index]) ** 2 + rel_current_error ** 2)

                plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp),
                             ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index], vrf_tmp),
                             xerr=unumpy.std_devs(
                                 ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
                                                                       ankacalc.fs_error(data_json[key]['fs'][index])),
                                                         unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             yerr=unumpy.std_devs(ankacalc.S_csr(
                                 unumpy.uarray(data_json[key]['ith'][index],ith_error_tmp) / 1000.,
                                 unumpy.uarray(data_json[key]['fs'][index],
                                               ankacalc.fs_error(data_json[key]['fs'][index])),
                                 unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             color='m',  # color=col[i],
                             fmt='o',
                             elinewidth=1, markersize='10', marker='*')
                x3.extend(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp))
                y3.extend(ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index], vrf_tmp))

                print(key,vrf,data_json[key]['fs'][index], ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp))


            for key in ['5905','6053']: #upper limit SBB single bunch # lower not measured
                index = np.where(data_json[key]['vrf'] == vrf)
                vrf_tmp = data_json[key]['vrf'][index]
                vrf_tmp_error = data_json[key]['vrf'][index] * 0.01
                ith_error_tmp = data_json[key]['ith_upper'][index] * \
                                np.sqrt(((ith_error(data_json[key]['ith_upper'][index]))
                                         / data_json[key]['ith_upper'][index]) ** 2 + rel_current_error ** 2)
                plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp),
                             ankacalc.S_csr(data_json[key]['ith_upper'][index] / 1000., data_json[key]['fs'][index], vrf_tmp),
                             xerr=unumpy.std_devs(
                                 ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
                                                                       ankacalc.fs_error(data_json[key]['fs'][index])),
                                                         unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             yerr=unumpy.std_devs(ankacalc.S_csr(
                                 unumpy.uarray(data_json[key]['ith_upper'][index],ith_error_tmp) / 1000.,
                                 unumpy.uarray(data_json[key]['fs'][index],
                                               ankacalc.fs_error(data_json[key]['fs'][index])),
                                 unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             color='m',  # color=col[i],
                             fmt='o',
                             elinewidth=1, markersize='10', marker='*')


            for key in ['6053']: #upper limit SBB single bunch # lower not measured
                index = np.where(data_json[key]['vrf'] == vrf)
                vrf_tmp = data_json[key]['vrf'][index]
                vrf_tmp_error = data_json[key]['vrf'][index] * 0.01
                ith_error_tmp = data_json[key]['ith_lower'][index] * \
                                np.sqrt(((ith_error(data_json[key]['ith_lower'][index]))
                                         / data_json[key]['ith_lower'][index]) ** 2 + rel_current_error ** 2)
                plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp),
                             ankacalc.S_csr(data_json[key]['ith_lower'][index] / 1000., data_json[key]['fs'][index], vrf_tmp),
                             xerr=unumpy.std_devs(
                                 ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
                                                                       ankacalc.fs_error(data_json[key]['fs'][index])),
                                                         unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             yerr=unumpy.std_devs(ankacalc.S_csr(
                                 unumpy.uarray(data_json[key]['ith_lower'][index],ith_error_tmp) / 1000.,
                                 unumpy.uarray(data_json[key]['fs'][index],
                                               ankacalc.fs_error(data_json[key]['fs'][index])),
                                 unumpy.uarray(vrf_tmp, vrf_tmp_error))),
                             color='m',  # color=col[i],
                             fmt='o',
                             elinewidth=1, markersize='10', marker='*')


            # for key in ['5731','5732']:
            #     index = np.where(data_json[key]['vrf'] == vrf)
            #     vrf_tmp = data_json[key]['vrf'][index] * ankacalc.voltage_factor().nominal_value \
            #         if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf'][index]
            #     vrf_tmp_error = data_json[key]['vrf'][index] * ankacalc.voltage_factor().std_dev \
            #         if data_json[key]['voltage_correction_necessary'] == 'True' else data_json[key]['vrf'][index] * 0.01
            #     ith_error_tmp = data_json[key]['ith'][index] * \
            #                     np.sqrt(((ith_error(data_json[key]['ith'][index]) if 'ith_error' not in data_json[key] else data_json[key]['ith_error'][index])
            #                              / data_json[key]['ith'][index]) ** 2 + rel_current_error ** 2)
            #
            #     plt.errorbar(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp),
            #                  ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index], vrf_tmp),
            #                  xerr=unumpy.std_devs(
            #                      ankacalc.shielding_bane(unumpy.uarray(data_json[key]['fs'][index],
            #                                                            ankacalc.fs_error(data_json[key]['fs'][index])),
            #                                              unumpy.uarray(vrf_tmp, vrf_tmp_error))),
            #                  yerr=unumpy.std_devs(ankacalc.S_csr(
            #                      unumpy.uarray(data_json[key]['ith'][index],ith_error_tmp) / 1000.,
            #                      unumpy.uarray(data_json[key]['fs'][index],
            #                                    ankacalc.fs_error(data_json[key]['fs'][index])),
            #                      unumpy.uarray(vrf_tmp, vrf_tmp_error))),
            #                  color='k',  # color=col[i],
            #                  fmt='o',
            #                  elinewidth=1, markersize='6', marker='<')
            #     x3.extend(ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp))
            #     y3.extend(ankacalc.S_csr(data_json[key]['ith'][index] / 1000., data_json[key]['fs'][index], vrf_tmp))
            #
            #     print(key,vrf,data_json[key]['fs'][index], ankacalc.shielding_bane(data_json[key]['fs'][index], vrf_tmp))




        if True:
            plt.plot(ankacalc.shielding_bane(fs, min(vrf_list)), ankacalc.S_csr(ankacalc.ith_bunchbeam_bane(fs, min(vrf_list)), fs, min(vrf_list)),
                     color='k', ls='-')
            # plt.plot(ankacalc.shielding_bane(fs, vrf), ankacalc.S_csr(ankacalc.ith_bunchbeam_sbb_lower(fs, vrf), fs, vrf),
            #          color='k', ls=':')
            # plt.plot(shielding_bane(fs, vrf), S_csr(0.035/1000., fs, vrf), color='r', ls='-')
            # plt.plot(shielding_bane(fs, vrf), S_csr(0.016/1000., fs, vrf), color='r', ls='-')


            # x_index1 = np.where(shielding_bane(fs, vrf) > 0.63)[0][0]
            # x_index2 = np.where(shielding_bane(fs, vrf) > 0.88)[0][0]
            # x_index3 = np.where(shielding_bane(fs, vrf) > 2.6)[0][0]
            # x = np.array([0.66, 0.755, 0.883, 0.885, 2.6])
            # y = np.array([0.17, 0.05, 0.17, S_csr(ith_bunchbeam_bane(fs, vrf), fs, vrf)[x_index2], S_csr(ith_bunchbeam_bane(fs, vrf), fs, vrf)[x_index3]])
            #
            # # plt.plot(x, y, 'o', color='r')
            #
            # from matplotlib.path import Path
            # verts = [
            #     (x[0], y[0]),  # P0
            #     (x[1], y[1]), # P1
            #     (x[2], y[2]), # P2
            #     (x[3], y[3]),
            #     (x[4], y[4]), # P3
            #     ]
            #
            # codes = [Path.MOVETO,
            #          Path.CURVE4,
            #          Path.CURVE4,
            #          Path.CURVE4,
            #          Path.LINETO,
            #          ]
            #
            # path = Path(verts, codes)
            # patch = matplotlib.patches.PathPatch(path, edgecolor='violet', facecolor='none', lw=10, alpha=0.4)
            # ax=plt.gca()
            # ax.add_patch(patch)

            x2 = np.asarray(x2)
            y2_lower = np.asarray(y2_lower)
            y2_upper = np.asarray(y2_upper)
            x2_sort_index = x2.argsort()
            x2_sorted = x2[x2_sort_index]
            y2_upper_sorted = y2_upper[x2_sort_index]
            y2_lower_sorted = y2_lower[x2_sort_index]
            # plt.plot(x2_sorted,y2_upper_sorted, '-')
            # plt.plot(x2_sorted,y2_lower_sorted, '-')
            plt.gca().fill_between(x2_sorted, y2_lower_sorted, y2_upper_sorted, facecolor="k", alpha=0.1, linewidth=0)
            plt.axvline(x=0.8350, ymin=0.05, ymax=0.35, linewidth=4, color='w', alpha=1, zorder=1)

            x3.append(0)
            y3.append(0.5)
            x3.append(3)
            y3.append(0.5 + 0.12 * 3)
            x3 = np.asarray(x3)
            y3 = np.asarray(y3)
            x3_sort_index = x3.argsort()
            x3_sorted = x3[x3_sort_index]
            y3_sorted = y3[x3_sort_index]
            plt.gca().fill_between(x3_sorted, y3_sorted, 1, facecolor='k', alpha=0.1, linewidth=0)
            # plt.gca().fill_between(shielding_bane(fs, vrf), S_csr(ith_bunchbeam_bane(fs, vrf), fs, vrf), 1, facecolor='k', alpha=0.25, linewidth=0)


        # from matplotlib.patches import Rectangle
        # someX, someY = 0, -0.05
        # hight, width = 1, 0.63
        # currentAxis = plt.gca()
        # currentAxis.add_patch(Rectangle((someX, someY), width, hight, facecolor="lightgrey", edgecolor='none'))


        # plt.xlim(xmin=0, xmax=2.4)
        plt.gca().locator_params(axis='x', nbins=17)  # ????warum ist das nicht die zahl die ich eingebe

        plt.xlim(xmin=0.6, xmax=2.4)
        plt.ylim(ymax=0.85, ymin=0.075)
        plt.xlabel('Shielding $\Pi$')
        plt.ylabel('CSR Strength $S_{\mathrm{CSR}}$')

        handles, labels = ax.get_legend_handles_labels()

        measurementArtist1 = plt.Line2D((0, 1), (0, 0), color='r', marker='s', markersize='6', linestyle='')
        measurementArtist4 = plt.Line2D((0, 1), (0, 0), color='g', marker='d', markersize='6', linestyle='')
        measurementArtist2 = plt.Line2D((0, 1), (0, 0), color='b', marker='^', markersize='6', linestyle='')
        measurementArtist3 = plt.Line2D((0, 1), (0, 0), color='orange', marker='o', markersize='6', linestyle='')
        measurementArtist5 = plt.Line2D((0, 1), (0, 0), color='m', marker='*', markersize='10', linestyle='')
        extra = matplotlib.patches.Rectangle((0, 0), 1, 1, facecolor="k", alpha=0.1, linewidth=0)
        extra2 = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-')

        l = plt.legend([measurementArtist1, measurementArtist4, measurementArtist5, measurementArtist2, measurementArtist3, extra, extra2],
                       ['Main Instability Threshold (with SBB)', 'Main instability Threshold (without SBB)', 'Thresholds (single bunch)',
                        'Upper Bound of SBB', 'Lower Bound of SBB', 'Regions of Instability', 'Theoretical Prediction'],
                       # 'sketch of threshold with dip'],
                       loc=(0.45, 0.1), ncol=1, numpoints=1, scatterpoints=1, labelspacing=0.5, handletextpad=0.5,
                       handlelength=1.5, frameon=False, borderpad=0.5, fontsize=17)  # , title='RF voltage')
        # plt.ylabel('CSR strength $\mathregular{S}_{CSR}$')

        # plt.savefig('/home/miriam/Documents/plots_for_diss/Scsr_over_Pi_including_Single-bunch-decays.pdf')


    plt.show()
