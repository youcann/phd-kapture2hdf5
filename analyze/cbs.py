#! /usr/bin/python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal
import scipy.optimize
import os
import scipy.constants
import i_th_fit_ankacalc as ankacalc
import uncertainties
from uncertainties import unumpy

## cbs alpha:

# data

cbs_alpha = np.array([0.00018, 0.00047, 0.00068, 0.0082])
# cbs_alpha = np.array([0.00018, 0.00047, 0.00067, 0.0082])
# cbs_alpha = np.array([0.00026, 0.00047, 0.00068, 0.0082])
cbs_alpha_error = np.array([0.00005, 0.00002, 0.00002, 0.0001])
# cbs_alpha_error = np.array([0.00005, 0.00002, 0.00001, 0.0001])
cbs_steps = np.array([27, 25, 23, 11])
cbs_fs = np.array([4.52, 6.289, 7.44, 25.78])
cbs_fs_error = cbs_fs * 0.02
cbs_vrf = np.array([140, 140, 140, 140])

# plt.figure()
# plt.plot(cbs_fs, cbs_alpha, 'o')
# cbs_fs_alpha_corrcoef = np.corrcoef(cbs_fs, cbs_alpha)
# print(cbs_fs_alpha_corrcoef)
# from scipy.optimize import curve_fit
#
# def func(x, a, b):
#    return a*x**2 + b
#
# popt, pcov = curve_fit(func, cbs_fs, cbs_alpha)
# print (popt,pcov)
# cbs_fs_linspace = np.linspace(min(cbs_fs)-2, max(cbs_fs)+2, 100)
# plt.plot(cbs_fs_linspace, func(cbs_fs_linspace,*popt))
#


# ???

parameters = ankacalc.load_csv('/mnt/linux_data/2015_CBS_alpha/alpha_measurement_CBS_april2015_settings.csv',
                      skipp=(0, 19, 20, 21, 64))
# print(parameters[0])

fs_cbs_scan = [i[3].replace(',', '.') for i in parameters]
fs_cbs_scan_new = np.array(fs_cbs_scan)
fs_cbs_scan_new = [float(i.split('/')[0]) for i in fs_cbs_scan_new]
# print(fs_cbs_scan_new)

v_cbs_scan = np.array([float((i[8].replace(',', '.')).split('/')[0]) for i in parameters])
# print(v_cbs_scan)

steps_cbs_scan = np.array([int(i[2]) for i in parameters]) / 1000
# print(steps_cbs_scan)

plt.figure()
f_rf = 499.7206e6
# matplotlib.rcParams['lines.markersize'] = 10
# plt.scatter(steps_5606, ankacalc.alpha(fs_5606, vrf_5606))
plt.errorbar(cbs_steps, cbs_alpha, yerr=cbs_alpha_error, color='r', fmt='o')
plt.scatter(steps_cbs_scan, ankacalc.alpha(fs_cbs_scan_new, v_cbs_scan), color='k')
plt.scatter(cbs_steps, ankacalc.alpha(cbs_fs, cbs_vrf), s=50, marker='s', color='g')
plt.scatter(cbs_steps, ankacalc.alpha(cbs_fs, np.ones(4) * 130), s=50, marker='*', color='g')
plt.scatter(cbs_steps, ankacalc.alpha(cbs_fs, np.ones(4) * 150), s=50, marker='<', color='g')
# plt.ylim(ymin=0, ymax=10e-4)
# plt.show()



## rückwärts spannung kalibriert:

# without uncertainties

cbs_energy = 1.285e9
# cbs_sigma = cbs_alpha * scipy.constants.c * e_spread / (2 * np.pi * cbs_fs * 1e3)
# delta_cbs_sigma = cbs_sigma / cbs_alpha * cbs_alpha_error
# print(cbs_alpha, cbs_alpha_error)
# print(cbs_sigma, delta_cbs_sigma)

cbs_v_cal_direkt = (cbs_fs * 1e3) ** 2 * cbs_energy * 184 * 2 * np.pi / ((f_rf) ** 2 * cbs_alpha) / (4e3)
delta_cbs_v_cal = np.abs(cbs_v_cal_direkt / cbs_alpha) * cbs_alpha_error
v_cal_factor_array = (cbs_v_cal_direkt / cbs_vrf)
delta_v_cal_factor_array = np.abs(v_cal_factor_array / cbs_alpha) * cbs_alpha_error
delta_v_cal_factor_array2 = np.sqrt(
    (v_cal_factor_array / cbs_alpha * cbs_alpha_error) ** 2 + (2 * v_cal_factor_array / cbs_fs * cbs_fs_error) ** 2)
# print(v_cal_factor_array, delta_v_cal_factor_array, delta_v_cal_factor_array2)
v_cal_factor = np.mean(v_cal_factor_array[1:])
delta_v_cal_factor = np.sqrt(np.sum((delta_v_cal_factor_array2 ** 2)[1:])) / len(v_cal_factor_array[1:])
# delta_v_cal_factor_grfe = np.mean(delta_v_cal_factor_array[1:])
# print(v_cal_factor, delta_v_cal_factor, delta_v_cal_factor_grfe)
print(v_cal_factor, delta_v_cal_factor)


# with uncertainties package

u_cbs_alpha = unumpy.uarray(cbs_alpha, cbs_alpha_error)
u_cbs_fs = unumpy.uarray(cbs_fs, cbs_fs_error)

# u_cbs_alpha2 = np.empty_like(u_cbs_alpha)
# u_cbs_fs2 = np.empty_like(u_cbs_fs)
# for i in range(len(cbs_alpha)):
#    (u_cbs_alpha2[i], u_cbs_fs2[i]) = uncertainties.correlated_values_norm([(cbs_alpha[i], cbs_alpha_error[i]), (cbs_fs[i], cbs_fs_error[i])], cbs_fs_alpha_corrcoef)
# print(uncertainties.correlation_matrix([u_cbs_alpha2[i], u_cbs_fs2[i]]))

u_cbs_v_cal_wfs = (u_cbs_fs * 1e3) ** 2 * cbs_energy * 184 * 2 * np.pi / ((f_rf) ** 2 * u_cbs_alpha) / (4e3)
u_v_cal_factor_array = (u_cbs_v_cal_wfs / cbs_vrf)
u_v_cal_factor = (u_v_cal_factor_array[1:]).mean()
print(u_v_cal_factor_array)
print('mean cal factor', u_v_cal_factor)

# plt.show()
