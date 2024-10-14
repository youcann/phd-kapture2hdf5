import scipy.io
import h5py
import math
import numpy as np


def readCurrent(currentPath):
    if currentPath.split('.')[-1] == 'h5':
        # Test for big accumulated h5 illingpattern File
        if 'FillingPattern_PH' in currentPath.split('/')[-1]:
            currentFile = h5py.File(currentPath, 'r')
            timestamps = currentFile['TimeStamps'][...]
            fillingPattern = currentFile['FillingPattern_PH'][...]
            beamCurrents = currentFile['BeamCurrents'][...]
            beamCurrents = beamCurrents[:,np.newaxis]
            bunchCurrents = np.multiply(beamCurrents,fillingPattern)
            attr = None
            return bunchCurrents, timestamps, attr
        else:
            currentFile = h5py.File(currentPath, 'r')

            current = currentFile['BeamData']['BeamCurrent'][...]
            fillingPattern = currentFile['FillingPattern']['FillingPattern_PH'][...]
            curr = current.mean() * fillingPattern
            # TODO: better using fillingpattern*current or bunchCurrents?
            # return curr
            if current[0] ==1:
                bunchCurrent=current[1]*fillingPattern
            elif current[1]==1:
                bunchCurrent=current[0]*fillingPattern
            else:
                bunchCurrent = currentFile['FillingPattern']['BunchCurrents_PH'][...]

    else:
        currentFile = scipy.io.loadmat(currentPath, squeeze_me=True)

        current = currentFile['Results']['Current'][()]
        fillingPattern = currentFile['Results']['FillingPatternPicoHarp'][()]
        curr = current.mean() * fillingPattern
        # TODO: better using fillingpattern*current or bunchCurrents?
        # return curr
        bunchCurrent = currentFile['Results']['BunchCurrents'][()]

    return bunchCurrent


def readinfo(currentPath, info='MissalignedBucket'):
    try:
        if currentPath.split('.')[-1] == 'h5':
            currentFile = h5py.File(currentPath, 'r')
            info = currentFile['FillingPattern'][info][...]
        else:
            currentFile = scipy.io.loadmat(currentPath, squeeze_me=True)
            info = currentFile['Results'][info]
    except:
        info = None

    return info


if __name__ == '__main__':
    import argparse
   
    parser = argparse.ArgumentParser(description='Processing ANKA filling pattern files')
    parser.add_argument('infile', type=str)
    
    args = parser.parse_args()
    infile= args.infile

    # infiles=['/mnt/internal4t/raw/f05107_2014-04-15/f05107_2014-04-15T21h45m08s_PH.mat',
    #          '/mnt/internal4t/raw/f05107_2014-04-15/f05107_2014-04-15T21h46m41s_PH.mat',
    #          '/mnt/internal4t/raw/f05107_2014-04-15/f05107_2014-04-15T21h48m15s_PH.mat',
    #          '/mnt/internal4t/raw/f05107_2014-04-15/f05107_2014-04-15T21h49m48s_PH.mat',
    #          '/mnt/internal4t/raw/f05107_2014-04-15/f05107_2014-04-15T21h51m23s_PH.mat']

    for f in infiles:
        currentFile = scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)
        print(currentFile['Results']._fieldnames)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(f.split('/')[-1] + '  ' + str(currentFile['Results'].Current))
        plt.plot(currentFile['Results'].BunchCurrents)
        # plt.ylim(ymax=1e5)
        print(currentFile['Results'].Energy)

    plt.show()
