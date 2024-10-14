import scipy.constants


class ANKA:
    circumference = 110.4  # Meter
    h = 184
    frev = scipy.constants.c / circumference
    # frev = 2715995. # Calculated from some rf frequency
    trev = 1 / frev

    frf = frev * h
    trf = 1 / frf
