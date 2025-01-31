#! /usr/bin/python



import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.optimize
import scipy.interpolate
import argparse


class PeakReconstruct(object):
    def __init__(self, numAdc):
        self.values = np.zeros(numAdc)
        self.times = np.zeros(numAdc)
        self.fit = None
        self.params = None

    def calc_gauss(self):
        # problem with maximum
        # print(self.times,self.values)
        mu = sum(self.times * self.values) / sum(self.values)  # mu
        width = np.sqrt(abs(sum((self.times - mu) ** 2 * self.values) / sum(self.values)))  # sigma

        max = self.values.max() * 1.3  # problem with maximum

        self.fit = lambda t: max * np.exp(-(t - mu) ** 2 / (2 * width ** 2))
        self.params = np.array([max, mu, width])
        #print(self.params)
        return self.params

    def calc_fit(self, intrument, offsetfit=False, plot=None):

        offset = 0 if intrument=='raw_zero' else 2048

        if offsetfit:
            assert offset == 0, 'to fit offset please use instrument = raw_zero, to ensure offset is 0' # TODO: offsetfit is slower and works only when offset is 0 not for 2048

            def gaussian(x, max, mu, sig, c):
                return max * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c  # / (sig * np.sqrt(2 * np.pi))

            p0 = max(self.values) - offset, np.mean(self.times), np.std(self.times), offset
            bounds = ([-np.inf, -np.inf, -np.inf, 0], np.inf)
            # p0 = np.append(self.calc_gauss(),2048)
        else:

            def gaussian(x, max, mu, sig):
                return max * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))+offset  # / (sig * np.sqrt(2 * np.pi))

            p0 = max(self.values) - offset, np.mean(self.times), np.std(self.times)
            bounds = None

        if bounds != None:
            curvefit, _ = scipy.optimize.curve_fit(gaussian, self.times, self.values, p0=p0, maxfev=1000,bounds=bounds)#, ftol=1e-6)
        else:
            curvefit, _ = scipy.optimize.curve_fit(gaussian, self.times, self.values, p0=p0, maxfev=1000)#, ftol=1e-6)
        curvefit[2] = abs(curvefit[2])

        self.params = curvefit

        self.fit = lambda t, curvefit=curvefit: gaussian(t, *curvefit)
        # TODO: why problem with limits?
        limit = [4 * np.max(self.values-offset), min(self.times), max(self.times), 0]
        if curvefit[0] > limit[0] or curvefit[1] < limit[1] or curvefit[1] > limit[2]:# or curvefit[2] < limit[3]:
            # self.plot()
            # plt.show()
            raise ValueError('Fit results out of range! Maximum=%.2f should be < %.2f and mu=%.2f should be between %.2f and %.2f' %(curvefit[0], limit[0], curvefit[1], limit[1], limit[2]))
        # else:
        #     print(curvefit)
        if plot:
            self.plot()
            plt.show()
        # self.chi_square_red()

    def chi_square_red(self, sigma=1):
        FitModel = self.fit(self.times)
        DoF = len(self.times) - self.params.size

        # Chi_square = np.sum((self.values-FitModel)**2/(rel_sigma*self.values)**2)
        Chi_square = np.sum((self.values-FitModel)**2/(sigma)**2)
        # Chi_square = np.sum((self.values-FitModel)**2/(FitModel))
        Chi_square_red = Chi_square / DoF
        # print(Chi_square_red)
        return Chi_square_red

    def fit_log_gauss(self, optimize=False):

        def gaussian(x, max, mu, sig):
            return max * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))  # / (sig * np.sqrt(2 * np.pi))

        def log_gaussian(x, max, mu, sig):
            return - np.power(x - mu, 2.) / (2 * np.power(sig, 2.)) + np.log(max)  # - np.log(sig * np.sqrt(2 * np.pi))

        p0 = max(self.values), np.mean(self.times), np.std(self.times)

        if optimize:
            curvefit, _ = scipy.optimize.curve_fit(log_gaussian, self.times, np.log(self.values), p0=p0, maxfev=100)
            print(curvefit)
        else:
            curvefit = np.polyfit(self.times, np.log(self.values), 2)
            fit_mu = -curvefit[1] * 1. / (2 * curvefit[0])
            fit_sig = np.sqrt(-1. / (2 * curvefit[0]))
            fit_max = np.exp(-np.power(curvefit[1], 2) / (4 * curvefit[0]) + curvefit[2])
            curvefit = [fit_max, fit_mu, fit_sig]
            print(curvefit)

        # if curvefit[0] > 4*np.max(self.values) or curvefit[1] < min(self.times) or curvefit[1] > max(self.times) or curvefit[2] < 0:
        #     print('fallback_test')
        #     raise RuntimeError()

        # print(curvefit)
        self.params = curvefit

        self.fit = lambda t, curvefit=curvefit: gaussian(t, *curvefit)
        # self.fit = f

    def sinx(self, i, x):
        X = (x - self.times[i]) / ((self.times[-1] - self.times[0]) / 3.)
        return self.values[i] * np.sinc(X)

    def fit_sinx(self):
        points = np.arange(self.times.min(), self.times.max(), 1 / 100.)
        # res = np.zeros_like(points)
        # fig = plt.figure(figsize=(12,6), tight_layout=True)
        # ax = fig.add_subplot(111)

        # for i, adc in enumerate(self.times):
        #     for j, point in enumerate(points):
        #         res[j] = self.sinx(i, point)
        #     # plt.plot(points, res, ':', label='fit_fine')

        res2 = np.zeros_like(points)
        for i, adc in enumerate(self.times):
            res2 += np.array([self.sinx(i, point) for j, point in enumerate(points)])
            # plt.plot(points, res2, label='fit_fine')
            # plt.plot(self.times, self.values, 'o', label='data')

            # print(res)

    def ipe_fit_modified(self):

        c = np.polynomial.polynomial.polyfit(self.times, self.values, 6)

        smooth_xs = np.linspace(self.times[0], self.times[-1], 16)
        fitted_ys = np.polynomial.polynomial.polyval(smooth_xs, c, tensor=True)

        max = np.max(fitted_ys)
        xs = smooth_xs[np.argmax(fitted_ys)]

        self.params = np.array([max, xs, None])

        self.fit = lambda t: np.polynomial.polynomial.polyval(t, c, tensor=True)

    def plot(self, numPoints=100):

        # set  default fontsize
        # standard figure size is (8,6) (width, height): if bigger or smaller picture is needed, change figure size instead of scaling the resulting image
        # for 100% text width in lyx use (12,6) (width,height)
        font = {'size': 19}
        matplotlib.rc('font', **font)

        fig = plt.figure(figsize=(12, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        # plt.plot(self.times, self.fit(self.times), label='fit')
        points = np.arange(self.times.min(), self.times.max(), 1. / numPoints)
        points = np.arange(self.params[1]-3*self.params[2], self.params[1]+3*self.params[2], 1. / numPoints)
        plt.plot(points, self.fit(points), label='fit_fine')
        plt.plot(self.times, self.values, 'o', label='data')
        plt.title('gaussfit: ' + str(self.params))
        plt.xlabel('Zeit / ps')
        plt.ylabel('Detektorsignal / a.u.')

        plt.legend(loc=8)
        # plt.xlim(xmin=0)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fit gaussian shape to 4 adc sampling points')
    parser.add_argument('action', choices=['class', 'calc', 'normpdf', 'gauss'])
    args = parser.parse_args()

    if args.action == 'class':

        peak_reconstruction = PeakReconstruct(numAdc=4)
        # peak_reconstruction.values = np.array([24, 40, 24, 5]) +2048 -2048
        # peak_reconstruction.times = (np.array([-1, 0, 1, 2]) + 20) * 3

        # peak_reconstruction.values = np.array([0.3, 0.38, 0.38, 0.3])
        # peak_reconstruction.times = (np.array([-0.77, -0.30, 0.30, 0.77]))

        peak_reconstruction.values = np.array([3, 38, 38, 3])
        peak_reconstruction.times = (np.array([-0.77, -0.30, 0.30, 0.77]) + 20) * 3
        # def gaussian(x, max, mu, sig):
        #     return max * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) # / (sig * np.sqrt(2 * np.pi))
        # peak_reconstruction.times = (np.array([-1.5, -0.5, 0.5, 1.5]))
        #
        # peak_reconstruction.values = np.array([gaussian(-1.5, 2, 1, 1), gaussian(-0.5, 2, 1, 1), gaussian(0.5, 2, 1, 1), gaussian(1.5, 2, 1, 1)])

        # points = np.arange(peak_reconstruction.times.min(), peak_reconstruction.times.max(), 1/100.)


        # peak_reconstruction.calc_gauss()
        # peak_reconstruction.plot(100)
        # peak_reconstruction.calc_fit()


        for i in range(1000):
            # peak_reconstruction.fit_log_gauss(optimize=True)
            # peak_reconstruction.fit_log_gauss()
            peak_reconstruction.calc_fit()
        # print(peak_reconstruction.params)
        # peak_reconstruction.plot(100)


        # for i in range(1000):
        # peak_reconstruction.fit_sinx()
        # plt.plot(points, gaussian(points,2, 1, 1), label='gaus')
        # peak_reconstruction.plot(100)

        # peak_reconstruction.ipe_fit_modified()

        plt.show()

    if args.action == 'normpdf':
        mean = 0
        variance = 1
        sigma = np.sqrt(variance)
        x = np.linspace(-3, 3, 100)
        plt.plot(x, mlab.normpdf(x, mean, sigma))
        plt.show()

        mlab.normpdf(x, mean, sigma)

        import scipy as sp
        import scipy.stats

        sp.stats.norm.pdf(x, mean, sigma)

        from matplotlib import pyplot as mp
        import numpy as np


        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))  # / (sig * np.sqrt(2 * np.pi))


        for mu, sig in [(-1, 1), (0, 2), (2, 3)]:
            mp.plot(gaussian(np.linspace(-3, 3, 120), mu, sig))

        mp.show()

        # from python import random        for mu, sig in [(-1, 1), (0, 2), (2, 3)]:

        # random.gauss(mu, sigma)
