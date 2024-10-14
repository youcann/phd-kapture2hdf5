import numpy as np


def rollData(data, referenceIntensity, roll_method):
    h = data.shape[0]  # equals harmonic number (184)
    roll_method = {
        'mean': lambda x: x.mean(axis=1, dtype=np.float64),
        'std': lambda x: x.std(axis=1, dtype=np.float64),
        'heb_inverted': lambda x: (((x - 2048) * -1.) + 2048).mean(1, dtype=np.float64)
    }[roll_method]

    avgData = roll_method(data)

    chi2 = np.array([np.sum((np.roll(avgData, shift) - referenceIntensity) ** 2) for shift in range(h)])

    shift = chi2.argmin()
    # print(shift)
    # Do not roll, if we do not need to
    if shift:
        # print('rolling')
        rolledData = np.roll(data, shift, axis=0)
    else:
        rolledData = data

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(rolledData.mean(1))

    # plt.figure()
    # x = range(h)
    # rolledAvg = rolledData.mean(1)
    # plt.plot(x, (rolledAvg / np.sum(rolledAvg)), x, (currentSqr/np.sum(currentSqr)))

    # print(chi2.argmin())
    return rolledData
