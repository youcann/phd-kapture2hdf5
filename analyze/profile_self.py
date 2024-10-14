from __future__ import print_function, division

import numpy as np


# import matplotlib.pyplot as plt


def Profile_rewrite(x, y, nbins, xmin, xmax, ax, col):
    # print(x)
    # print('x', x)
    print(((xmax - xmin) / float(nbins)))
    binedges = xmin + ((xmax - xmin) / nbins) * np.arange(nbins + 1)
    bind_index = np.digitize(x, binedges)
    print()

    bincenters = xmin + ((xmax - xmin) / nbins) * np.arange(nbins) + ((xmax - xmin) / (2 * nbins))
    # ProfileFrame = DataFrame({'bincenters' : bincenters, 'N' : df['bin'].value_counts(sort=False)},index=range(1,nbins+1))
    # number = len(bind_index)
    # print('edges', binedges, 'centers', bincenters)
    # ax.plot(bincenters, np.ones(len(bincenters))*(max(y)-min(y))/2, 'g+')
    # ax.plot(binedges, np.ones(len(binedges))*(max(y)-min(y))/2, 'y|')


    bins = np.arange(1, nbins + 1)
    # bins = np.nonzero(np.bincount(bind_index))[0]

    # print('bincount',np.bincount(bind_index), 'bins',bins)
    counts = np.bincount(bind_index, minlength=max(bins)+1)[bins]
    ymean = np.empty_like(bins, dtype=np.float32)
    ystd = np.empty_like(bins, dtype=np.float32)
    ymeanerror = np.empty_like(bins, dtype=np.float32)

    for i, bin in enumerate(bins):
        ymean[i] = np.nanmean(np.where(bind_index == bin, y, np.nan))
        ystd[i] = np.nanstd(np.where(bind_index == bin, y, np.nan))
        ymeanerror[i] = ystd[i] / np.sqrt(float(counts[i]))

    ax.errorbar(bincenters, ymean, yerr=ymeanerror, xerr=(xmax - xmin) / nbins / np.sqrt(12.), c=col)
    # ax_new = ax.twinx()
    # ax_new.plot(bincenters, ystd)
    # ax_new.set_ylim(ymax=10)
    # ax.errorbar(bincenters, ymean, yerr=ystd, xerr=(xmax-xmin)/(2*nbins)/np.sqrt(12.), fmt=None, color='r')
    # print(ymean, ystd)
    # return ax,ax_new
    return ax


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    ax = plt.gca()

    x1=np.arange(1000)**2#, dtype='float')
    y1=x1**2
    x=np.arange(1000,step=10)**2#, dtype='float')
    y=x**2

    ax.plot(x1,y1,'r*')
    max_delta=max(max([x1[i+1]-x1[i] for i in range(len(x1)-2)]), max([x[i+1]-x[i] for i in range(len(x)-2)]))
    nbins = int(np.rint((max(max(x1),max(x))-min(min(x1),min(x)))/(max_delta*5)))
    print(max_delta, nbins)

    ax = Profile_rewrite(x1,y1, nbins, xmin=min(x1), xmax=max(x1), ax=ax, col='b')
    plt.figure()
    ax = plt.gca()



    ax.plot(x,y,'r*')

    ax = Profile_rewrite(x,y, nbins, xmin=min(x), xmax=max(x), ax=ax, col='b')

    # x = np.load('../x_all.npy')
    # y = np.load('../data_all.npy')
    # nbins = len(x) / 10
    # ax = Profile_rewrite(x, y, nbins, xmin=min(x), xmax=max(x), ax=ax, col='b')
    #
    # x = np.load('../x_all_red.npy')
    # y = np.load('../data_all_red.npy')
    # nbins = len(x) / 10
    # ax = Profile_rewrite(x, y, nbins, xmin=min(x), xmax=max(x), ax=ax, col='r')
    #
    # plt.sca(ax)
    # plt.xlim([0.163, 0.183])
    # plt.ylim([9, 41])

    # print(x,y)
    plt.show()
