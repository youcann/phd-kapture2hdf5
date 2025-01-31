import numpy as np
import warnings
import math
from . import constants

FILLING = (0x01234567, 0x89ABCDEF, 0xDEADDEAD)
MAX_ADCS = 4
HEADER_SIZE_WORDS = 32 // 4

_ofloor = math.floor
math.floor = lambda x: int(_ofloor(x))


def detect_data_chunks_by_filling(data):
    # Find data filling
    data_filling = (data == FILLING[0]) | (data == FILLING[1]) | (data == FILLING[2])
    # Find indexes of where filling starts and ends
    filling_changes = np.where(np.diff(data_filling))
    # Prefix changes with begin and end of array
    filling_changes = np.append([-1, ], np.append(filling_changes, data.size - 1))

    data_chunks = []
    for i in range(filling_changes.size - 1):
        boundary_pair = filling_changes[[i, i + 1]] + [1, 0]
        inJunk = data_filling[boundary_pair]

        # Detect data junk "transitions" and remember them
        if (inJunk == [False, False]).all() or (inJunk == [True, False]).all():
            data_length = boundary_pair[1] - boundary_pair[0]
            data_chunks.append((data_length, boundary_pair))

    return data_chunks


def parse_header(header):
    parsed = dict()
    print('parsing header')
    # print([format(b, '#08X') for b in header])

    assert header[0] == 0xf8888888 or header[7] == 0xf8888888, 'Field 8 is saved for data'
    assert header[1] == 0xf7777777 or header[6] == 0xf7777777, 'Field 7 is saved for data'
    assert header[2] == 0xf6666666 or header[5] == 0xf6666666, 'Field 6 is saved for data'

    if header[0] == 0xf8888888 and header[1] == 0xf7777777 and header[2] == 0xf6666666:
        header = header[::-1]

    # TODO: what kind of timestamp is this? counts with appr. 25MHz up?!
    parsed['timestamp'] = header[4]

    assert header[3] >> 8 == 0, 'Highest 6 x 4 bits of field 3 is supposed to be zero'
    parsed['delay_adc4'] = header[3] & 0xf
    parsed['delay_adc3'] = header[3] >> 4 & 0xf

    assert header[2] >> 16 == 0, 'Highest 4 x 4 bits of field 2 is supposed to be zero'
    parsed['delay_adc2'] = header[2] & 0xf
    parsed['delay_adc1'] = header[2] >> 4 & 0xf
    parsed['delay_th'] = header[2] >> 8 & 0xf
    parsed['delay_fpga'] = header[2] >> 12 & 0xf

    parsed['fine_delay_adc1'] = header[1] & 0xff
    parsed['fine_delay_adc2'] = header[1] >> 8 & 0xff
    parsed['fine_delay_adc3'] = header[1] >> 16 & 0xff
    parsed['fine_delay_adc4'] = header[1] >> 24 & 0xff

    assert header[0] >> 28 == 0xF, 'Highest 4 bits of field 0 is supposed to be 0xF'
    parsed['skip_turns'] = header[0] & 0xfffffff

    print(parsed)
    return parsed


def read_raw_board_file(file_name, num_adcs=4, read_adc=None, samples=None, detect_filling=True):
    """
    Read in raw board files

    :param file_name: File name to open and read the raw data from
    :param read_adc: List of ADC numbers to read in, all available ADCs if None
    :param detect_filling: Try to detect any filling patterns. Slow and not recommended for new files.
    """
    print(num_adcs)
    # fp = np.memmap(file_name, dtype='<I4')
    fp = np.memmap(file_name, dtype='uint32')

    # Check if the first entry is filling
    first_is_filling = fp[0] in FILLING
    last_is_filling = fp[-1] in FILLING

    if last_is_filling and not detect_filling:
        raise RuntimeError('Last entry contains a filling pattern, but you did not ask for its detection!')

    if first_is_filling:
        raise RuntimeError('Filling pattern at start of file detected. This is not supported!')

    if detect_filling:
        data_chunks = detect_data_chunks_by_filling(fp)

        if len(data_chunks) > 1:
            warnings.warn('Multiple chunks of data detected! Just using the first one.')

        # We can only use the first chunk, since otherwise ADC detection mixes up.
        chunk = data_chunks[0]

        data_start, data_stop = chunk[1]
        assert data_start == 0, 'Detected chunk does not start at 0!'
    else:
        data_start, data_stop = 0, fp.shape[0]

    # print('Data read from row %i to row %i.' % (data_start, data_stop))

    # Define conversion functions
    extract_counter = lambda x: np.right_shift(x, 24) & 0xfff
    extract_bunch_1 = lambda x: np.right_shift(x, 12) & 0xfff
    extract_bunch_0 = lambda x: x & 0xfff

    counter = extract_counter(fp[data_start:data_start + MAX_ADCS + HEADER_SIZE_WORDS + 1])
    # Test if counter for all ADCs start with zero (only problematic if not always all ADCs are saved in out file
    # TODO: improve test by doing it step by step (undependet of number of saved ADCs)
    print('test for header')
    has_header = (counter[0:MAX_ADCS - 1].any() != 0)
    if num_adcs==8:
        has_header = False
    print(has_header)

    tmp_start = HEADER_SIZE_WORDS if has_header else 0

    #nr_adcs = 0
    #for x in counter[tmp_start:]:
    #    if x != 0:
    #        break
    #    nr_adcs += 1
    nr_adcs = num_adcs

    assert nr_adcs > 0, 'No ADC format found due to malformed counter'
    # FIXME: Add safety check for ADC detection

    if has_header:
        header = fp[data_start:data_start + HEADER_SIZE_WORDS]
        data_start += HEADER_SIZE_WORDS

    # big hack to skip reading header when 8 ADCS....TODO: parse header correctly for KAPTURE2
    if num_adcs==8:
        data_start += HEADER_SIZE_WORDS

    # Define the ADCs we want to read if none were defined.
    read_adc = list(range(nr_adcs)) if read_adc is None else read_adc

    results = dict()
    for adc in read_adc:
        assert 0 <= adc < nr_adcs, 'ADC %i does not exist in this file!' % adc

        stop = data_stop if not samples else min(data_stop, (samples // 2 + 1) * nr_adcs)
        single_adc_data = fp[data_start + adc:stop:nr_adcs]
        bunch_0 = extract_bunch_0(single_adc_data)
        bunch_1 = extract_bunch_1(single_adc_data)
        bunch_counter = extract_counter(single_adc_data)

        # The counter should count up in steps of two and drop back to 0 once it reaches the harmonic number.
        # Let's check for this (more or less...)
        # assert np.all(np.unique(np.diff(np.asarray(bunch_counter, dtype=np.int))) == [-182, 2]), \
        #     'Counter does not count up.'

        # Interleave values
        bunches = np.empty((bunch_0.size + bunch_1.size,), dtype=np.int16)
        bunches[0::2] = bunch_0
        bunches[1::2] = bunch_1

        if samples:
            results[adc] = bunches[:samples]
        else:
            results[adc] = bunches

    return {'samples': results, 'header': parse_header(header) if has_header else None, 'nr_adcs': nr_adcs}
    # return {'samples': results, 'header': None, 'nr_adcs': nr_adcs}


def reshape_data_to_bunches(adcs_samples):
    result = dict()
    for adc, samples in list(adcs_samples.items()):
        turns = math.floor(samples.size // constants.ANKA.h)

        cutData = samples[:constants.ANKA.h * turns]
        cutData = cutData.reshape((turns, constants.ANKA.h)).transpose()

        result[adc] = cutData
    return result


def apply_normalization(adcs_samples, scaling):
    def convert_raw(x):
        return x

    def convert_raw_zero(x):
        x -= 2 ** 11
        return x

    def convert_inverted(x):
        tmp = np.empty_like(x, dtype=np.float32)
        tmp[...] = x
        tmp -= 2**11
        tmp *= -1.
        return tmp

    def convert_float_example(x):
        # Example on how to scale efficiently with floats
        tmp = np.empty_like(x, dtype=np.float32)
        tmp[...] = x
        tmp /= 10.
        return tmp

    scale_dict = {'raw': convert_raw,
                  'raw_zero': convert_raw_zero,
                  'inverted': convert_inverted,
                  'float_example': convert_float_example}

    for adc in list(adcs_samples.keys()):
        assert scaling[adc] in scale_dict, 'ADC %i scaling function must be one of %s' %(adc, list(scale_dict.keys()))

    result = dict()
    for adc, samples in list(adcs_samples.items()):
        scale_fun = scale_dict[scaling[adc]]
        result[adc] = scale_fun(samples)
    return result


if __name__ == '__main__':
    # path = '/mnt/network/volatile/Temp/peakreconstruct_testdata/1399481194.823.out'
    path = '/mnt/network/volatile/Temp/1399651267.676.out'
    # path = '/mnt/network/volatile/Temp/thz_group_data/ipeboard/data/bench_multi_0000.out'

    raw_sample_data = read_raw_board_file(path, samples=1000, read_adc=(0, 3), detect_filling=False)
    reshaped_data = reshape_data_to_bunches(raw_sample_data['samples'])
    normalized_data = apply_normalization(reshaped_data, 'raw')

    print(raw_sample_data)
