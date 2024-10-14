#! /usr/bin/python

import numpy as np
import scipy.signal


def movingaverage(interval, window_size):
    window = scipy.signal.hamming(window_size)
    window = window / window.sum()
    # window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
    # return scipy.signal.medfilt(interval,window_size)
