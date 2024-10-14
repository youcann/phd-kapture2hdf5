

import math
import numpy as np
from scipy import signal

import scipy.fftpack
import analyze.constants as const

import logging
import time

_ofloor = math.floor
math.floor = lambda x: int(_ofloor(x))


# try:
#     import reikna.cluda
#     import reikna.fft
#
#     _plans = {}
#     _in_buffers = {}
#     _out_buffers = {}
#
#     _api = reikna.cluda.ocl_api()
#     _thr = _api.Thread.create()
#
#
#     def fft(array):
#         start = time.time()
#         padded = array.astype(np.complex64)
#         height, width = padded.shape
#
#         if width in _plans:
#             fft = _plans[width]
#             in_dev = _in_buffers[width]
#             out_dev = _out_buffers[width]
#         else:
#             fft = reikna.fft.FFT(padded, axes=(1,)).compile(_thr)
#             in_dev = _thr.to_device(padded)
#             out_dev = _thr.empty_like(in_dev)
#             _plans[width] = fft
#             _in_buffers[width] = in_dev
#             _out_buffers[width] = out_dev
#
#         fft(out_dev, in_dev)
#         results = out_dev.get()[:, :width // 2 + 1]
#
#         print("GPU fft: {} s".format(time.time() - start))
#         return results
#
#     print('Using GPU based FFT!')
#
# except ImportError:
#     logging.info("Failed to import reikna package. Falling back to Numpy FFT.")
#     print('Using Numpy FFT!')


def fft(array):
    start = time.time()
    freqs = np.fft.rfft(array)
    print("np fft: {} s".format(time.time() - start))
    return freqs


def rfftfreq(n, d=1.0):
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = np.arange(0, N, dtype=int)
    return results * val


def doFFT(data, powerOfTwo=True, bucket=None, phase=False, skippedTurns=1):
    # set n to a power of 2 to get a higher symmetry and therefore a faster fft
    # TODO: May move this cut in its own function
    if powerOfTwo:
        n = 2 ** math.floor(math.log(data.shape[1], 2))
        data = data[:, :n]

    if bucket == None:
        fft = fft(data)
    else:
        fft = fft(data[bucket, :])

    if not phase:
        fft = np.abs(fft)

    fftfreq = rfftfreq(data.shape[1], const.ANKA.trev * skippedTurns)
    # print(fft[:,1].shape)
    # print(fftfreq)
    freqstep = fftfreq[1] - fftfreq[0]
    freqmax = fftfreq[-1]

    return {'fft': fft, 'fftfreq': fftfreq, 'freqstep': freqstep, 'freqmax': freqmax}


    # When the input a is a time-domain signal and A = fft(a),
    # np.abs(A) is its amplitude spectrum and
    # np.abs(A)**2 is its power spectrum.
    # The phase spectrum is obtained by np.angle(A).
