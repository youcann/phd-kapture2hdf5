import numpy as np
import analyze.constants as const


def binaere_suche(data):
    maxindex = data.shape[1] - 1
    index = 1
    # print('start')
    first_turn = data[:, 0]
    while index <= maxindex:
        mitte = index + ((maxindex - index) / 2)
        comp = data[:, mitte]
        conv2 = np.array([np.sum((np.roll(first_turn, shift) * comp)) for shift in range(const.ANKA.h)])
        # print(' ')
        # print('mitte:')
        # print(mitte)
        # print(conv2.argmax())

        if conv2.argmax() in (182, 183, 0, 1, 2):  # (183,0,1):
            index = mitte + 1
            # print('obere haelfte')
            # print(index)
        else:
            maxindex = mitte - 1
            # print('untere haelfte')
            # print(maxindex)
            # print "Die Suche ist am Ende"
    # print(mitte)
    # print(conv2.argmax())
    return mitte


def corjump(data):
    # test = data[:,0]*1.0
    # comp = data[:,87499]*1.0

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.arange(h),test)
    # plt.plot(np.arange(h),comp)

    # conv2 = np.array([np.sum((np.roll(test, shift) * comp)) for shift in range(h)])

    # print('maxpos:')
    # print(conv2.argmax())

    # plt.figure()
    # plt.plot(np.arange((h)),conv2)

    turnnumberjump = binaere_suche(data)
    if not turnnumberjump == data.shape[1] - 1:
        # print(data.shape[1])
        print('jump! at ' + str(turnnumberjump))
        # correcteddata = correct(data, turnnumberjump)

    return turnnumberjump
