#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function, division

import csv
import numpy as np
import scipy.signal
import scipy.optimize
import scipy.constants
from uncertainties import unumpy
import constants

## Konstanten
R = 5.559
h = 32e-3
# h=16e-3
# h=184
e_spread = 0.47e-3
# e_spread = 0.59e-3
# e_spread = 0.47e-3 *1.1
f_rf = 499.705e6
# vorfaktor = 1.25
vorfaktor = 1.
# vorfaktor = 0.85
energy = 1.3e9

# energy = 1.285e9
# energy = 1.287e9


# TODO: check if sigma in m or mm! -> m 99%
def sigma(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    vrf = vrf * 1e3
    return scipy.constants.c * e_spread * 1e3 * energy * constants.ANKA.h / (
        (f_rf) ** 2 * vrf) * fs


def sigma_exact(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    vrf = vrf * 1e3
    return scipy.constants.c * e_spread * 1e3 * energy * constants.ANKA.h / (
        f_rf ** 2 * np.sqrt(vrf ** 2 - (8.8575e-5 * (1e-9 ** 3) / 5.559 * energy ** 4) ** 2)) * fs


def alpha(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    #return sigma(fs, vrf, e_spread, f_rf, energy) * 2 * np.pi * fs * 1e3 / (scipy.constants.c * e_spread)
    vrf = vrf * 1e3
    return  2 * np.pi * fs * fs * 1e3 * 1e3 * energy * constants.ANKA.h / ((f_rf) ** 2 * vrf) 


def vrf(fs, alpha, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return (fs * 1e3) ** 2 * energy * constants.ANKA.h * 2 * np.pi / ((f_rf) ** 2 * alpha) / (1e3)


def fs(alpha, vrf, energy=energy):
    vrf = vrf * 1e3
    return np.sqrt(alpha * f_rf ** 2 / constants.ANKA.h * vrf / (energy * 2 * np.pi)) / 1000.


def fs_from_sigma(sigma, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):  # in kHz
    vrf = vrf * 1e3
    return sigma * f_rf ** 2 * np.sqrt(vrf ** 2 - (8.8575e-5 * (1e-9 ** 3) / 5.559 * energy ** 4) ** 2) / (
    scipy.constants.c * e_spread * energy * constants.ANKA.h) * 1 / 1e3


def ith_bunchbeam_freeCSR(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return vorfaktor * (energy / 511e3) * alpha(fs, vrf, e_spread, f_rf, energy) * (e_spread) ** 2 * 17045 * 0.5 * np.power(R,
                                                                                                    -1.0 / 3.0) * np.power(
        sigma(fs, vrf, e_spread, f_rf, energy), 1.0 / 3.0)


# def ith_bunchbeam(fs, vrf):
#     return vorfaktor * (energy / 511e3) * alpha(fs, vrf) * (e_spread) ** 2 * 17045 * (
#                0.5 * np.power(R, -1.0 / 3.0) * np.power(sigma(fs, vrf), 1.0 / 3.0)
#                + 0.34 * np.power(R, 1.0 / 6.0) * np.power(sigma(fs, vrf), 4.0 / 3.0)
#                * np.power(h, -3.0 / 2.0))


def ith_bunchbeam_cai(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return vorfaktor * (energy / 511e3) * alpha(fs, vrf, e_spread, f_rf, energy) * (e_spread) ** 2 * 17045 * np.power(R,
                                                                                                           -1.0 / 3.0) * \
           np.power(sigma(fs, vrf, e_spread, f_rf, energy=energy), 1.0 / 3.0) * \
           (0.5 + 0.34 * np.power(R, 1.0 / 2.0) * sigma(fs, vrf, e_spread, f_rf, energy) * np.power(h, -3.0 / 2.0))


def ith_bunchbeam_bane(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return vorfaktor * (energy / 511e3) * alpha(fs, vrf, e_spread, f_rf, energy) * (e_spread) ** 2 * 17045 * np.power(R, -1.0 / 3.0) * \
           np.power(sigma(fs, vrf, e_spread, f_rf, energy), 1.0 / 3.0) * \
           (0.5 + 0.12 * np.power(R, 1.0 / 2.0) * sigma(fs, vrf, e_spread, f_rf, energy) * np.power(h / 2., -3.0 / 2.0))


# def ith_bunchbeam_sbb_approx(fs, vrf):
#     return vorfaktor * (energy / 511e3) * alpha(fs, vrf) * (e_spread) ** 2 * 17045 * (
#                0.15 * np.power(R, -1.0 / 3.0) * np.power(sigma(fs, vrf), 1.0 / 3.0))


def ith_bunchbeam_sbb_lower(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return vorfaktor * (energy / 511e3) * alpha(fs, vrf, e_spread, f_rf, energy) * (e_spread) ** 2 * 17045 * np.power(R, -1.0 / 3.0) * np.power(
        sigma(fs, vrf, e_spread, f_rf, energy), 1.0 / 3.0) * 0.17


def ith_bunchbeam_sbb_upper(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return vorfaktor * (energy / 511e3) * alpha(fs, vrf, e_spread, f_rf, energy) * (e_spread) ** 2 * 17045 * np.power(R, -1.0 / 3.0) * np.power(
        sigma(fs, vrf, e_spread, f_rf, energy), 1.0 / 3.0) * 0.54


def shielding_cai(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return sigma(fs, vrf, e_spread, f_rf, energy) * np.power(R, 0.5) / np.power(h, 3. / 2.)


def shielding_bane(fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return sigma(fs, vrf, e_spread, f_rf, energy) * np.power(R, 0.5) / np.power(h / 2., 3. / 2.)


def I_norm(i_th, fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return i_th * sigma(fs, vrf, e_spread, f_rf, energy) / ((energy / 511e3) * alpha(fs, vrf, e_spread, f_rf, energy) * e_spread ** 2 * 17045)


def S_csr(i_th, fs, vrf, e_spread=e_spread, f_rf=f_rf, energy=energy):
    return I_norm(i_th, fs, vrf, e_spread, f_rf, energy) * np.power(R, 1.0 / 3.0) * np.power(sigma(fs, vrf, e_spread, f_rf, energy), -4.0 / 3.0)


def cbs_voltage_calibration():
    cbs_alpha = np.array([0.00018, 0.00047, 0.00068, 0.0082])
    cbs_alpha_error = np.array([0.00005, 0.00002, 0.00002, 0.0001])
    cbs_steps = np.array([27, 25, 23, 11])
    cbs_fs = np.array([4.52, 6.289, 7.44, 25.78])
    cbs_fs_error = cbs_fs * 0.02
    cbs_vrf = np.array([140, 140, 140, 140])

    ## rückwärts spannung kalibriert:
    cbs_energy = 1.285e9

    u_cbs_alpha = unumpy.uarray(cbs_alpha, cbs_alpha_error)
    u_cbs_fs = unumpy.uarray(cbs_fs, cbs_fs_error)

    u_cbs_v_cal_wfs = (u_cbs_fs * 1e3) ** 2 * cbs_energy * constants.ANKA.h * 2 * np.pi / ((f_rf) ** 2 * u_cbs_alpha) / (4e3)
    u_v_cal_factor_array = (u_cbs_v_cal_wfs / cbs_vrf)
    u_v_cal_factor = (u_v_cal_factor_array[1:]).mean()

    import matplotlib.pyplot as plt
    plt.errorbar([1,2,3,4], unumpy.nominal_values(u_cbs_v_cal_wfs), yerr=unumpy.std_devs(u_cbs_v_cal_wfs), fmt='o')
    plt.plot([1,2,3,4], [140,140,140,140])
    plt.xlabel('Measurments')
    plt.ylabel('Voltage per cavity / kV')
    # plt.show()

    # print('back calc voltage:', u_cbs_v_cal_wfs)
    # print('calibration factor per cavity:', u_v_cal_factor_array)
    # print('mean calibration factor:', u_v_cal_factor)

    return u_v_cal_factor


def voltage_factor():
    return cbs_voltage_calibration() * 4


def fs_error(fs):
    fs_err_fac = 0.016
    fs_err = 0.5
    fs_err = 0.1  # 100Hz
    # fs_err = 0.01  # 10Hz
    # return fs*fs_err_fac
    return fs_err


# def ith_fv(fs, vrf):
#    fs = fs * 1e3
#    vrf = 4 * vrf * 1e3
#    return vorfaktor * (energy / 511e3 + 1) * 2 * np.pi * 17045 * (
#        0.5 * (scipy.constants.c * (e_spread) ** 7 * fs ** 7 * (energy) ** 4 * h ** 4 /
#               (R * f_rf ** 8 * vrf ** 4)) ** (1 / 3) +
#        0.34 * R ** (1 / 6) * h ** (5 / 6) * (scipy.constants.c ** 4 * e_spread ** 10 * fs ** 10 * (energy) ** 7 /
#                                              (f_rf ** 14 * vrf ** 7)) ** (1 / 3))
#
#    #
#
# def ith_test(fs, vrf):
#    vrf = np.sqrt(vrf ** 2 - (8.8575e-5 * (1e-9 ** 3) / 5.559 * (energy) ** 4) ** 2)
#    return vorfaktor * (energy / 511e3 + 1) * ((scipy.constants.c * e_spread * 1e3 * energy * constants.ANKA.h / (
#        (f_rf) ** 2 * 4 * vrf * 1e3) * fs) * 2 * np.pi * fs * 1e3 / (scipy.constants.c * e_spread)) * (
#               e_spread) ** 2 * 17045 * (0.5 * np.power(R, -1 / 3) * np.power(
#        (scipy.constants.c * e_spread * 1e3 * energy * constants.ANKA.h / ((f_rf) ** 2 * 4 * vrf * 1e3) * fs), 1 / 3)
#                                         + 0.34 * np.power(R, 1 / 6) * np.power(
#        (scipy.constants.c * e_spread * 1e3 * energy * constants.ANKA.h / ((f_rf) ** 2 * 4 * vrf * 1e3) * fs), 4 / 3)
#                                         * np.power(h, -3 / 2))


def load_csv(filename, skipp=None):
    tmp = list()
    for i, row in enumerate(csv.reader(open(filename, 'r'))):
        if i in skipp:
            continue
        tmp.append(row)
    return tmp


if __name__ == '__main__':
    print('main')
    # print(400*voltage_factor().nominal_value)
    # # print(voltage_factor())
    # print(shielding_bane(7.19,400*voltage_factor().nominal_value))
    # print(shielding_bane(6.55,1400))
    # print(shielding_bane(8.1,1500))
    # print(shielding_bane(4.5,650))

    # print('23k: a_0: %f' %(0.00067), 'a_1: %.3f' %(-0.023),'a_1*de: %.7f' %(-0.023*e_spread))
    # print('23k: (a_1*de)/a_0: %f Prozent' %((-0.023*e_spread)/0.00067 *100))
    # print('25k: a_0: %f' %(0.00047), 'a_1: %.3f' %(-0.016),'a_1*de: %.7f' %(-0.016*e_spread),'a_2: %.1f' %(-2.),'a_2*de^2: %.7f' %(-2. * e_spread**2))
    # print('25k: (a_1*de)/a_0: %f Prozent' %((-0.016*e_spread)/0.00047 *100))
    # print('25k: (a_2*de^2)/a_0: %f Prozent' %((-2.0*e_spread**2)/0.00047 *100))
    # print('27k: a_0: %f' %(0.00019), 'a_1: %.3f' %(-0.014),'a_1*de: %.7f' %(-0.014*e_spread),'a_2: %.1f' %(14.),'a_2*de^2: %.7f' %(14. * e_spread**2))
    # print('27k: (a_1*de)/a_0: %f Prozent' %((-0.014*e_spread)/0.00019 *100))
    # print('27k: (a_2*de^2)/a_0: %f Prozent' %((14.0*e_spread**2)/0.00019 *100))
    # #
    # print(alpha(7.2,400*voltage_factor().nominal_value))
    # # print(ith_bunchbeam_bane(9.8, 325*voltage_factor().nominal_value)*1000)
    # print(300*voltage_factor().nominal_value)
    # b=ith_bunchbeam_bane(7.2,300*voltage_factor().nominal_value-10)*1000
    # a=ith_bunchbeam_bane(7.2,300*voltage_factor().nominal_value)*1000
    # c=ith_bunchbeam_bane(7.2,300*voltage_factor().nominal_value+10)*1000
    # print(a,b,c)
    # print(a-b)
    # print(a-c)
    # print(abs(a-b)-abs(a-c))
    #

    # print(sigma(11.4,1100)/scipy.constants.c *1e12)
    # print(ith_bunchbeam_cai(8.24, 799))
    print(alpha(8.5, 800))
    print(alpha(7.2,300*voltage_factor().nominal_value))
    print(alpha(4.8,800))
    print(alpha(3.9,770))
    print(alpha(3.7,770))
    print(sigma(7.3,1500)/scipy.constants.c *1e12)
    print(ith_bunchbeam_cai(7.5, 800)*1000.)

    #print(sigma(np.array([37.95,21.28,20.14,18.94,16.1,14.73,10.84,9.3,7.49,5.39]),np.array([770,]*10))/scipy.constants.c *1e12)
    #print(ith_bunchbeam_cai(np.array([37.95,21.28,20.14,18.94,16.1,14.73,10.84,9.3,7.49,5.39]),np.array([770,]*10))/scipy.constants.c *1e12)

    #print(alpha(7.3,600),alpha(8.25,800),fs(alpha(8.25,800),770),ith_bunchbeam_cai(8.25,800)*1000.,ith_bunchbeam_cai(8.0,770)*1000.)


    #
    # print(ith_bunchbeam_cai(8.25,800)*1000)
    # print(ith_bunchbeam_cai(8.34,800)*1000)
    #print(voltage_factor())
    #print(sigma(7.5,771)/scipy.constants.c *1e12)

    #print(ith_bunchbeam_cai(7.95, 771)*1000)
    ##print(ith_bunchbeam_cai(7.6, np.arange(730,800,10))*1000)

    #print(ith_bunchbeam_bane(33,311,e_spread=0.47e-3/1.3*0.5, energy=0.5e9))
    #print(ith_bunchbeam_cai(33,311,e_spread=0.47e-3/1.3*0.5, energy=0.5e9))
    #print(ith_bunchbeam_cai(33,311,e_spread=0.47e-3, energy=1.3e9))
    #print(alpha(33,311,e_spread=0.47e-3/1.3*0.5, energy=0.5e9), alpha(33,311,e_spread=0.47e-3, energy=1.3e9))

    #print(ith_bunchbeam_bane(30,600,e_spread=0.47e-3/1.3*0.5, energy=0.5e9)*1000)



    #print(alpha(13.1,50,e_spread=0.47e-3/1.3*0.5, energy=0.5e9))

    #print('f05731-2')
    #print(shielding_bane(fs(2.81e-4,1100),1100), shielding_bane(fs(2.89e-4,1100),1100))
    #print(shielding_bane(fs(4.85e-4, 1100), 1100), shielding_bane(fs(4.98e-4, 1100), 1100))
    #print(shielding_bane(fs(6.73e-4, 1100), 1100), shielding_bane(fs(6.88e-4, 1100), 1100))


    # print(alpha(np.array([8.065,6.318,5.722,5.25,5.722,6,7.68,6.17,6.25,6.55,6.84,5.84]), np.array([1500,1500,1300,1100,1200,1300,1500,1200,1300,1400,1500,1100])))
    # print(ith_bunchbeam_cai(8.34,800)*1000)

    print(sigma(33,311,e_spread=1.8e-4, energy=0.5e9)/scipy.constants.c *1e12)

    print('pos (311kV): ', ith_bunchbeam_cai(33,311,e_spread=1.8e-4, energy=0.5e9)*1000)
    print('neg (50kV): ', ith_bunchbeam_cai(13,50,e_spread=1.8e-4, energy=0.5e9)*1000)

    print(fs(alpha(13,50,e_spread=1.8e-4, energy=0.5e9), 100, energy=0.5e9))

    print('neg (100kV): ', ith_bunchbeam_cai(18.3,100,e_spread=1.8e-4, energy=0.5e9)*1000)

    print(fs(alpha(33,311,e_spread=1.8e-4, energy=0.5e9), 50, energy=0.5e9))

    print('pos (50kV): ', ith_bunchbeam_cai(fs(alpha(33,311,e_spread=1.8e-4, energy=0.5e9), 50, energy=0.5e9),50,e_spread=1.8e-4, energy=0.5e9)*1000)

    print('pos (30kV): ', ith_bunchbeam_cai(fs(alpha(33,311,e_spread=1.8e-4, energy=0.5e9), 30, energy=0.5e9),30,e_spread=1.8e-4, energy=0.5e9)*1000)


    print(alpha(7.5,771))

    print('low neg alpha:')
    print(ith_bunchbeam_cai(fs(alpha(6, 50, e_spread=1.8e-4, energy=0.5e9),300, e_spread=1.8e-4, energy=0.5e9),300, e_spread=1.8e-4, energy=0.5e9)*1000)
    print(ith_bunchbeam_cai(6, 50, e_spread=1.8e-4, energy=0.5e9)*1000)
    print(alpha(6, 50, e_spread=1.8e-4, energy=0.5e9))


    if True:
        import matplotlib.pyplot as plt
        import matplotlib

        font = {'size': 19}
        matplotlib.rc('font', **font)
        # plt.figure()
        # plt.plot(range(50,500, 20), [ith_bunchbeam_cai(fs(alpha(6, 50, e_spread=1.8e-4, energy=0.5e9),i, e_spread=1.8e-4, energy=0.5e9),i, e_spread=1.8e-4, energy=0.5e9)*1000 for i in range(50, 500, 20)])
        # plt.xlabel(r'$V_{\mathrm{RF}}$ / V')
        # plt.ylabel(r'$I_{\mathrm{th}}$ / mA')
        #
        # plt.figure()
        # plt.plot(np.linspace(1, 8, 15) * 1e-3, [
        #     ith_bunchbeam_cai(fs(i, 50, e_spread=1.8e-4, energy=0.5e9), 50,
        #                       e_spread=1.8e-4, energy=0.5e9) * 1000 for i in np.linspace(1, 8, 15) * 1e-3])
        # plt.xlabel(r'$\alpha_{\mathrm{c}}$')
        # plt.ylabel(r'$I_{\mathrm{th}}$ / mA')
        #
        # plt.figure()
        # plt.plot(np.log10(np.linspace(1, 8, 15)*1e-3), np.log10([
        #     ith_bunchbeam_cai(fs(i, 50, e_spread=1.8e-4, energy=0.5e9), 50,
        #                       e_spread=1.8e-4, energy=0.5e9) * 1000 for i in np.linspace(1, 8, 15)*1e-3]))
        #
        # tmpy=np.log10([
        #     ith_bunchbeam_cai(fs(i, 50, e_spread=1.8e-4, energy=0.5e9), 50,
        #                       e_spread=1.8e-4, energy=0.5e9) * 1000 for i in np.linspace(1, 8, 15)*1e-3])
        # tmpx=np.log10(np.linspace(1, 8, 15)*1e-3)
        #
        # print((tmpy[-1]-tmpy[0])/(tmpx[-1]-tmpx[0]))
        # plt.show()

        # for fill 5789 at 1.3GeV 27k steps 1500kV fs= 8.56kHz -> 2.94e-4 alpha
        print(alpha(8.56, 1500, e_spread=e_spread, energy=energy), e_spread, energy)
        # plt.figure(figsize=(12,6), tight_layout=True)
        plt.figure(figsize=(7,4), tight_layout=True)
        plt.plot(range(100, 1500, 20), [
            ith_bunchbeam_cai(fs(alpha(8.56, 1500, e_spread=e_spread, energy=energy), i, e_spread=e_spread, energy=energy), i,
                              e_spread=e_spread, energy=energy) * 1000 for i in range(100, 1500, 20)], 'b')
        plt.xlabel(r'$V_{\mathrm{RF}}$ / V')
        plt.ylabel(r'$I_{\mathrm{th}}$ / mA')
        ax=plt.gca()
        plt.text(0.5, 0.9, r'$\alpha_{\mathrm{c}} = 2.94 \cdot 10^{-4}$',horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)

        plt.savefig('/Users/miriam/Documents/sync_server/plots_for_diss/ith_over_alpha_f5789-as-start-value.pdf')

        plt.figure(figsize=(7,4), tight_layout=True)
        plt.plot(np.linspace(0.1, 1.8, 18)*10, [
            ith_bunchbeam_cai(fs(i, 1500, e_spread=e_spread, energy=energy), 1500,
                              e_spread=e_spread, energy=energy) * 1000 for i in np.linspace(0.1, 1.8, 18) * 1e-3], 'b')
        plt.xlabel(r'$\alpha_{\mathrm{c}}$ / $10^{-4}$')
        plt.ylabel(r'$I_{\mathrm{th}}$ / mA')
        ax = plt.gca()
        plt.text(0.5, 0.9, r'$V_{\mathrm{RF}} = 1500$ kV',horizontalalignment='center',
                 verticalalignment='center', transform=ax.transAxes)
        plt.savefig('/Users/miriam/Documents/sync_server/plots_for_diss/ith_over_voltage_f5789-as-start-value.pdf')


        # plt.figure()
        # plt.plot(np.log10(np.linspace(1, 8, 15) * 1e-3), np.log10([
        #     ith_bunchbeam_cai(fs(i, 50, e_spread=e_spread, energy=energy), 50,
        #                       e_spread=e_spread, energy=energy) * 1000 for i in np.linspace(1, 8, 15) * 1e-3]))
        #
        # tmpy = np.log10([
        #     ith_bunchbeam_cai(fs(i, 50, e_spread=e_spread, energy=energy), 50,
        #                       e_spread=e_spread, energy=energy) * 1000 for i in np.linspace(1, 8, 15) * 1e-3])
        # tmpx = np.log10(np.linspace(1, 8, 15) * 1e-3)
        #
        # print((tmpy[-1] - tmpy[0]) / (tmpx[-1] - tmpx[0]))
        # plt.show()

        print(sigma(fs(alpha(6, 50, e_spread=1.8e-4, energy=0.5e9),300, e_spread=1.8e-4, energy=0.5e9),300, e_spread=1.8e-4, energy=0.5e9))

        print(shielding_bane(fs(alpha(6, 50, e_spread=1.8e-4, energy=0.5e9),300, e_spread=1.8e-4, energy=0.5e9),300))


    print('0.5GeV:')
    # print(ith_bunchbeam_cai(fs(alpha(6, 50, e_spread=1.8e-4, energy=0.5e9),300, e_spread=1.8e-4, energy=0.5e9),300, e_spread=1.8e-4, energy=0.5e9)*1000)
    print(ith_bunchbeam_cai(46, 600, e_spread=1.8e-4, energy=0.5e9)*1000)
    print(shielding_bane(46, 600, e_spread=1.8e-4, energy=0.5e9))


    # print('thesis_ino_fs_scan:')
    # print(shielding_bane(np.array([ 6.98992, 7.98848,  8.98704,  9.9856,  10.98416, 11.98272, 12.98128, 13.97984]), np.array([300*voltage_factor().nominal_value,]*8)))

    print('for maxfreq as function of pi')
    print(shielding_bane(np.array([8.2,9.8,9.4,7.2,7.7,7.2,10.3,7.5,6.32,8.7,9.62,11.4,10.75,8.1]),np.array([800,325*voltage_factor().nominal_value,300*voltage_factor().nominal_value,400*voltage_factor().nominal_value,733,300*voltage_factor().nominal_value,350*voltage_factor().nominal_value,400*voltage_factor().nominal_value,300*voltage_factor().nominal_value,300*voltage_factor().nominal_value,300*voltage_factor().nominal_value,1100,1500,1500])))
    print(shielding_bane(np.array([7.7,9.8,8.9,7.2,7.7,7.2,10.3,7.5,6.32,8.7,9.62,11.4,10.75,7.8]),np.array([800,325*voltage_factor().nominal_value,300*voltage_factor().nominal_value,400*voltage_factor().nominal_value,733,300*voltage_factor().nominal_value,350*voltage_factor().nominal_value,400*voltage_factor().nominal_value,300*voltage_factor().nominal_value,300*voltage_factor().nominal_value,300*voltage_factor().nominal_value,1100,1500,1500])))
    print(shielding_bane(7.9,771))
    print(400*voltage_factor().nominal_value)

    print(constants.ANKA.trev)

    print(alpha(11.2,1100))
    print(alpha(10.8,1500))

    print('for clic')
    print(ith_bunchbeam_bane(7.7,771))
    print(ith_bunchbeam_bane(7.5,771))
    print(ith_bunchbeam_bane(7.95,771))
    print(ith_bunchbeam_bane(fs(3.3e-5,771),771))
    print(ith_bunchbeam_bane(fs(3.3e-5,771),771))

    print('niels decay')
    print(ith_bunchbeam_bane(4.2, 1000)*1000)
    print(alpha(4.2, 1000))
    print(shielding_bane(4.2, 1000))
    print(sigma(fs(alpha(4.2, 1000),1500),1500)/scipy.constants.c*1e12)

    print('diss:')
    print(alpha(8.9, 300*voltage_factor().nominal_value))
    print(alpha(9.8, 325*voltage_factor().nominal_value))
    print(alpha(10.3, 350*voltage_factor().nominal_value))
    print(alpha(7.2, 300*voltage_factor().nominal_value))

    print(shielding_bane(7.2,400*voltage_factor().nominal_value), 400*voltage_factor().nominal_value)

    # print(shielding_bane(np.array([7.7,6.11,6.31,6.51,6.77]),np.array([1500,1200,1300,1400,1500])))
    # print(alpha(np.array([7.7,6.11,6.31,6.51,6.77]),np.array([1500,1200,1300,1400,1500])))


    print(alpha(np.array([13.15,12.56,11.96,11.17,10.21,9.14,8.27,7.11]),300*voltage_factor().nominal_value))


    print(alpha(np.array([8.2,9.8,9.4,7.2,7.7,7.2,10.3,7.5,6.32,8.7,9.62,11.4,10.75,8.1,7.7,7.9]), np.array([800, 325 * voltage_factor().nominal_value, 300 * voltage_factor().nominal_value,400 * voltage_factor().nominal_value, 733, 300 * voltage_factor().nominal_value,350 * voltage_factor().nominal_value, 400 * voltage_factor().nominal_value, 300 * voltage_factor().nominal_value,300 * voltage_factor().nominal_value, 300 * voltage_factor().nominal_value, 1100, 1500, 1500,733,771])))

    fs_list=np.array([8.2,9.8,9.4,7.2,7.7,7.2,10.3,7.5,6.32,8.7,9.62,11.4,10.75,8.1,7.7,7.9])
    vrf_list=np.array([800, 325 * voltage_factor().nominal_value, 300 * voltage_factor().nominal_value,400 * voltage_factor().nominal_value, 733, 300 * voltage_factor().nominal_value,350 * voltage_factor().nominal_value, 400 * voltage_factor().nominal_value, 300 * voltage_factor().nominal_value,300 * voltage_factor().nominal_value, 300 * voltage_factor().nominal_value, 1100, 1500, 1500,733,771])
    print(unumpy.std_devs(shielding_bane(unumpy.uarray(fs_list,fs_error(fs_list)),
                                     unumpy.uarray(vrf_list,vrf_list * 0.01))))



    print(ith_bunchbeam_cai(14, 300, e_spread=1.8e-4, energy=0.5e9)*1000)
    print(ith_bunchbeam_cai(20, 600, e_spread=1.8e-4, energy=0.5e9)*1000)
