#! /usr/bin/python

from __future__ import print_function, division
import argparse
import math
import numpy as np
import h5py
import os
import os.path
import itertools
import time
import sys

from analyze import filereader
from analyze import current
from analyze import roll
from analyze import constants as const
from analyze import current2
from analyze import fft
from analyze import gaus
from tqdm import *

from analyze.helpers import description_to_ranges, ranges_to_numbers, find_ranges, \
    describe_ranges, user_yes_no_query


_ofloor = math.floor
math.floor = lambda x: int(_ofloor(x))


def floor_power_two(x):
    return 2 ** math.floor(math.log(x, 2))


# 'ADC:0-3:waveform,fft_abs,fft_phase:unit=Volt'
# python2 hdf5_readin2.py raw 1399651267.676.out test.hdf --channel ADC:0-3:waveform MEAN_ADC:0-3:waveform:name=Test

def convert_channel_adc(data, input_channels, channel_types, options, header, skipped_turns=0, *args):
    channels = dict()

    do_waveform = False
    do_fft_abs = False
    do_fft_phase = False

    for output_channel in channel_types:
        if output_channel == 'waveform':
            do_waveform = True
        elif output_channel == 'fft_abs':
            do_fft_abs = True
        elif output_channel == 'fft_phase':
            do_fft_phase = True
        elif output_channel == 'fft':
            do_fft_abs = True
            do_fft_phase = True
        else:
            raise RuntimeError('Unknown output channel description "%s"' % output_channel)

    for in_channel in input_channels:
        channel = dict()

        channel['unit'] = options.get('unit', 'ADC value')
        channel['short_unit'] = options.get('short_unit', 'ADC value')
        channel['name'] = options.get('name', 'ADC Channel %i' % in_channel)
        channel['short_name'] = options.get('short_name', 'ADC %i' % in_channel)

        if do_waveform:
            channel['waveform'] = data[in_channel]

        if do_fft_abs or do_fft_phase:
            n_samples_fft = floor_power_two(data[in_channel].shape[1])
            fft_comp = fft.fft(data[in_channel][:, :n_samples_fft])

            if do_fft_abs:
                channel['fft_abs'] = np.abs(fft_comp)
            if do_fft_phase:
                channel['fft_phase'] = np.angle(fft_comp)

            if do_fft_abs or do_fft_phase:
                channel['fft_freq'] = fft.rfftfreq(n_samples_fft, const.ANKA.trev * (skipped_turns + 1))

        channels['ADC_%i' % in_channel] = channel
    return channels


def convert_channel_mean_adc(data, input_channels, channel_types, options, header, skipped_turns=0, *args):
    do_waveform = False
    do_fft_abs = False
    do_fft_phase = False

    for output_channel in channel_types:
        if output_channel == 'waveform':
            do_waveform = True
        elif output_channel == 'fft_abs':
            do_fft_abs = True
        elif output_channel == 'fft_phase':
            do_fft_phase = True
        elif output_channel == 'fft':
            do_fft_abs = True
            do_fft_phase = True
        else:
            raise RuntimeError('Unknown output channel description "%s"' % output_channel)

    assert len(input_channels) >= 2, 'At least two ADC channels expected to generate a mean value'
    mean = np.zeros_like(data[input_channels[0]], dtype=np.float32)

    for in_channel in input_channels:
        mean += data[in_channel]
    mean /= len(input_channels)

    channel = dict()

    channel['unit'] = options.get('unit', 'Average ADC value')
    channel['short_unit'] = options.get('short_unit', 'Avg. ADC value')
    channel['name'] = options.get('name', 'Average of channels %s' % describe_ranges(find_ranges(input_channels)))
    channel['short_name'] = options.get('short_name', 'Avg. ADC %s' % describe_ranges(find_ranges(input_channels)))

    if do_waveform:
        channel['waveform'] = mean

    if do_fft_abs or do_fft_phase:
        n_samples_fft = floor_power_two(mean.shape[1])
        fft_comp = fft.fft(mean[:, :n_samples_fft])

        if do_fft_abs:
            channel['fft_abs'] = np.abs(fft_comp)
        if do_fft_phase:
            channel['fft_phase'] = np.angle(fft_comp)

        channel['fft_freq'] = fft.rfftfreq(n_samples_fft, const.ANKA.trev * (skipped_turns + 1))

    return {'MEAN_ADC_%s' % '_'.join((str(x) for x in input_channels)): channel}


def convert_channel_peak(data, input_channels, channel_types, options, header, skipped_turns=0, dropped_buckets=[], dropped_buckets_value=0, instrument='raw'):

    assert all(value == instrument[0] for value in instrument.values()), 'For peakreconstruction all channels need to be treated the same, please choose one instrument for all channels!'
    assert all(value == dropped_buckets_value[0] for value in dropped_buckets_value.values()), 'For peakreconstruction all channels need to be treated the same, please choose one drop-buckets-value for all channels!'

    channels = dict()

    do_waveform = False
    do_fft_abs = False
    do_fft_phase = False

    for output_channel in channel_types:
        if output_channel == 'waveform':
            do_waveform = True
        elif output_channel == 'fft_abs':
            do_fft_abs = True
        elif output_channel == 'fft_phase':
            do_fft_phase = True
        elif output_channel == 'fft':
            do_fft_abs = True
            do_fft_phase = True
        else:
            raise RuntimeError('Unknown output channel description "%s"' % output_channel)

    assert len(input_channels) == 4, 'All four ADC channels expected to reconstruct the peak'

    # print(input_channels)

    # time point of each sampling channel out of options:
    # times = np.array([options[key] for key in options.keys()])
    times = np.array([int(options[key]) for key in ('ta', 'tb', 'tc', 'td')])
    assert all([header['fine_delay_adc'+str(i+1)]*3 == times[i] for i in range(4)]), 'delays don\'t fit to values in header'
    coarse_delay = header['delay_th']

    # Merge by gaussian fit
    recon_data = np.zeros(data[input_channels[0]].shape)
    recon_time = np.zeros(data[input_channels[0]].shape)
    recon_sigma = np.zeros(data[input_channels[0]].shape)
    recon_true = np.zeros(data[input_channels[0]].shape, dtype=np.uint16)
    peak_recon = gaus.PeakReconstruct(numAdc=len(input_channels))

    # get background time scan data
    f = file('/mnt/heb1/2017-07-05_alp_peak_tests-25Gev/50ohm/timescan/timescan_1499264066.443.out')
    #TODO: fix readin of 50Ohm offset timescan!!!!
    print('\n\n Hard coded file path for 50Ohm reference file, please fix me!\n')
    c_adc = -1  # -1 because first adc will have #ADC0 and this will increment c_adc
    zero_correction_values = np.zeros((4, 32, 16))  # we know there are 4 adcs, 16 coarse delays and 32 fine delays
    all_check = np.zeros((4, 32, 16), dtype=np.bool)
    for line in f:
        line = line.strip()
        if line == "": continue  # skip empty lines
        if line.startswith("#"):
            c_adc += 1  # c_adc = int(line.split("_")[-1])
            continue
        coarse, fine, value = line.split(";")
        zero_correction_values[c_adc, int(fine), int(coarse)] = value
        all_check[c_adc, int(fine), int(coarse)] = True
    if False in all_check:
        import warnings
        warnings.warn("Read 50Ohm timescan for offset correction does not include all delay values")
    channel_offsets = np.array([zero_correction_values[in_channel,times[in_channel]//3,coarse_delay] for in_channel in input_channels])-2048

    # parse through data and fit gaussian
    for bucket in range(const.ANKA.h):
        if bucket in dropped_buckets:
            recon_data[bucket, :] = dropped_buckets_value
            recon_time[bucket, :] = 0
            recon_sigma[bucket, :] = 0
            recon_true[bucket, :] = 0
        else:
            print('bucket %i' % bucket)
            for turn in tqdm(range(data[input_channels[0]].shape[1]), desc='bucket %i' % bucket):
                # maxi = None
                # mu = None
                # sigma = None

                # print('test')
                peak_recon.values = np.asarray([data[in_channel][bucket, turn] for in_channel in input_channels])-channel_offsets
                peak_recon.times = times - np.array([0, 1.9, 8.4, 4.2])
                try:
                    peak_recon.calc_fit(instrument, offsetfit=False)#, plot=True if turn==147 else None) # TODO: offsetfit is slower and works only when offset is 0 not for 2048
                    # peak_recon.calc_gauss()
                    # peak_recon.plot()
                    maxi = peak_recon.params[0] + peak_recon.params[3] if len(peak_recon.params) == 4 else \
                        peak_recon.params[0]
                    mu = peak_recon.params[1]
                    sigma = peak_recon.params[2]
                    worked = peak_recon.chi_square_red()
                except (RuntimeError, ValueError) as e:
                    # TODO: what to do if fit doesn't converge?
                    maxi = np.max(peak_recon.values)
                    # maxi = 5000
                    mu = peak_recon.times[np.argmax(peak_recon.values)]
                    # mu = -2000.
                    # sigma = np.sqrt(abs(sum((peak_recon.times - mu) ** 2 * peak_recon.values) / sum(peak_recon.values))) #sigma
                    sigma = np.std(peak_recon.times)
                    # sigma = 0.
                    worked = 0
                    # e = sys.exc_info()[0]

                    print('Error at bucket %i and turn %i (%s)' % (bucket, turn, e))

                recon_data[bucket, turn] = maxi
                recon_time[bucket, turn] = mu
                recon_sigma[bucket, turn] = sigma
                recon_true[bucket, turn] = worked
                # f.close()
    # return (recon_data, recon_time, recon_sigma)

    for i, recon in enumerate((recon_data, recon_time, recon_sigma, recon_true)):
        channel = dict()

        if i == 0:
            channel['unit'] = options.get('unit', 'THz-Signal')
            channel['short_unit'] = options.get('short_unit', 'THz-Signal')
            channel['name'] = options.get('name', 'Reconstructed Signal Maximum')
            channel['short_name'] = options.get('short_name', 'Reconstructed Signal')
        elif i == 1:
            channel['unit'] = options.get('unit', 'Time')
            channel['short_unit'] = options.get('short_unit', 'Time')
            channel['name'] = options.get('name', 'Reconstructed Signal Arrival Time')
            channel['short_name'] = options.get('short_name', 'Arrival Time')
        elif i == 2:
            channel['unit'] = options.get('unit', 'Time')
            channel['short_unit'] = options.get('short_unit', 'Time')
            channel['name'] = options.get('name', 'Reconstructed Signal Width')
            channel['short_name'] = options.get('short_name', 'Signal Width')
        else:
            channel['unit'] = options.get('unit', 'Binary')
            channel['short_unit'] = options.get('short_unit', 'Binary')
            channel['name'] = options.get('name', 'Reconstruction Flag')
            channel['short_name'] = options.get('short_name', 'Flag')

        if do_waveform:
            channel['waveform'] = recon

        if do_fft_abs or do_fft_phase:
            n_samples_fft = floor_power_two(recon.shape[1])
            fft_comp = fft.fft(recon[:, :n_samples_fft])

            if do_fft_abs:
                channel['fft_abs'] = np.abs(fft_comp)
            if do_fft_phase:
                channel['fft_phase'] = np.angle(fft_comp)

            channel['fft_freq'] = fft.rfftfreq(n_samples_fft, const.ANKA.trev * (skipped_turns + 1))

        channels['_'.join(channel['short_name'].upper().split(' '))] = channel

    return channels


def parse_channel_specification(channel_spec):
    results = list()
    for channel in channel_spec:
        fields = [x.strip() for x in channel.split(':')]

        if not (3 <= len(fields) <= 4):
            raise RuntimeError('Channel specification not valid, expected 3 or 4 columns.')

        converter_name = fields[0].lower()
        in_channels = ranges_to_numbers(description_to_ranges(fields[1]))
        out_channels = [x.strip().lower() for x in fields[2].split(',')]

        # We call the converter function, which has the name convert_channel_{converter_name}
        # Get it via global defined variables, which behaves like a normal dictionary.
        converter_fun_name = 'convert_channel_%s' % converter_name
        assert converter_fun_name in globals(), 'Channel converter not known.'
        converter_fun = globals()[converter_fun_name]

        if len(fields) == 4:
            # FIXME: REALLY stupid parser. Never use , or =  or : in values or keys!
            options = {key.strip(): value.strip() for key, value in (line.split('=') for line in fields[3].split(','))}
        else:
            options = dict()

        results.append({'converter': converter_name, 'in_channels': in_channels, 'out_channels': out_channels,
                        'options': options, 'fun': converter_fun})

    return results


def main():
    parser = argparse.ArgumentParser(description='Processing ANKA Bursting files into HDF5 files version 1')

    parser.add_argument('instrument', type=str, nargs='+', help='Normalization applied to the data file. if in doubt, use raw')
    parser.add_argument('--infile', nargs='+', type=str, required=True, help='Data files transferred into HDF5 file')
    parser.add_argument('--outfile', type=str, required=True, help='HDF5 output file.')
    parser.add_argument('--channel', '-c', nargs='+', required=True,
                        help='Input channel description.')

    input_group = parser.add_argument_group('Raw file import options')
    input_group.add_argument('--number-adcs', type=int, default=4)
    input_group.add_argument('--skipped-turns', type=int, default=None)
    input_group.add_argument('--detect-filling', action='store_true',
                             help='Try to detect data chunks in data files. '
                                  'Should not be needed for new files anymore.')
    input_drop_group = input_group.add_mutually_exclusive_group()
    input_drop_group.add_argument('--drop-buckets', nargs='+', type=str, default=[],
                                  help='One or multiple bucket range specifications. '
                                       'All buckets will be set to DROP_BUCKETS_VALUE')
    input_drop_group.add_argument('--keep-buckets', nargs='+', type=str, default=None,
                                  help='One or multiple bucket range specifications. '
                                       'All other buckets will be set to DROP_BUCKETS_VALUE')
    input_group.add_argument('--drop-buckets-value', type=int, nargs='+', default=[0,])
    input_group.add_argument('--time-source', choices=['filename', 'creation_timestamp', 'counter', 'none'],
                             default='filename')
    input_group.add_argument('--no-cut-power-two', action='store_true',
                             help='Do not cut the input files to contain only power of two samples.')
    input_group.add_argument('--max-samples', type=int, help='Maximum number of samples to read.', default=None)

    current_group = parser.add_argument_group('Bunch current parameters.')
    current_group.add_argument('--current-csv', type=str, help='CSV file containing bunch current over time.')
    current_group.add_argument('--current-roll-reference-file', type=str,
                               help='File to rotate the current to. Should fit to your other read in files.')
    current_group.add_argument('--current-roll-reference-file-adc', type=int, default=None,
                               help='ADC number of the CURRENT_ROLL_REFERENCE_FILE to take as reference. '
                                    'Defaults to one of the read in ADCs.')
    current_group.add_argument('--current-roll-method', choices=('mean', 'std', 'heb_inverted'), default='mean')

    hdf5_group = parser.add_argument_group('HDF5 specific parameters')
    hdf5_group.add_argument('--compression', choices=['gzip', 'lzf', 'szip'], default=None)
    hdf5_group.add_argument('--overwrite', action='store_true', help='Overwrite an existing HDF5 file without asking.')

    meta_group = parser.add_argument_group('Measurement metadata and descriptions')
    meta_group.add_argument('--description', type=str, help='Description of the measurement', default='')
    meta_group.add_argument('--short-description', type=str, help='Short description of the measurement', default='')
    meta_group.add_argument('--fill', type=int, help='Fill number', default=0)

    time_group = parser.add_argument_group('Time offset correction')
    time_group.add_argument('--timeoffset-board', type=float, default=0)
    time_group.add_argument('--timeoffset-ph', type=float, default=0)

    roll_group = parser.add_argument_group('Roll all data files to a reference')
    roll_group.add_argument('--reference-adc', type=int,
                            help='ADC of input file used to compare against reference')
    roll_group.add_argument('--roll-method', choices=('mean', 'std', 'heb_inverted'), default='mean')

    roll_ref_group = roll_group.add_mutually_exclusive_group()
    roll_ref_group.add_argument('--roll-reference-current-mat', type=str,
                                help='MATLAB file containg a filling pattern to.')
    roll_ref_group.add_argument('--roll-reference-raw', type=str,
                                help='A reference data file to roll to.')

    args = parser.parse_args()

    # Catch unimplemented rolling function
    # TODO: Implent rolling of data files to a current (*mat) reference
    if args.roll_reference_current_mat:
        raise NotImplementedError('Rolling of data files with a mat files as reference is not yet implemented!')

    rolling = False if (args.roll_reference_raw is None and args.reference_adc is None) else True
    print('Rolling is', rolling)

    # First try to parse the channel specification
    channel_spec = parse_channel_specification(args.channel)

    # Figure out which ADC channels are actually needed.
    used_adc_channels = tuple(sorted(set(itertools.chain(*[x['in_channels'] for x in channel_spec]))))
    print('The specified channel descriptions will require the %s ADC channels.' % (tuple(used_adc_channels),))

    # Figure out which channel will be normalized with which function
    assert len(args.instrument)==1 or len(args.instrument)==len(used_adc_channels), 'Please specify one instrument per used channel or one for all!'
    instrument = {adc: args.instrument[0] for i, adc in enumerate(used_adc_channels)} if len(args.instrument) == 1 else {adc: args.instrument[i] for i, adc in enumerate(used_adc_channels)}

    if args.drop_buckets or args.keep_buckets:
        if args.drop_buckets:
            dropped_buckets = sorted(set(itertools.chain(*[ranges_to_numbers(description_to_ranges(x)) for
                                                           x in args.drop_buckets])))
        else:
            dropped_buckets = sorted(
                set(range(const.ANKA.h)) - set(itertools.chain(*[ranges_to_numbers(description_to_ranges(x)) for
                                                        x in args.keep_buckets])))
        print('Buckets %s will be dropped.' % (describe_ranges(find_ranges(dropped_buckets)),))
    else:
        dropped_buckets = []
        print('All buckets will be imported.')

    assert len(args.drop_buckets_value)==1 or len(args.drop_buckets_value)==len(used_adc_channels), 'Please specify one drop-buckets-value per used channel or one for all!'
    drop_buckets_value = {adc: args.drop_buckets_value[0] for i, adc in enumerate(used_adc_channels)} if len(args.drop_buckets_value) == 1 else {adc: args.drop_buckets_value[i] for i, adc in enumerate(used_adc_channels)}

    # Check beforehand if all files do exists
    for file_path in args.infile:
        if not os.path.isfile(file_path):
            raise RuntimeError('Input file path "%s" does not exists!' % file_path)

    if os.path.isfile(args.outfile):
        if args.overwrite or user_yes_no_query('HDF output file exists, do you want to delete it?'):
            os.remove(args.outfile)
            print('Deleted existing HDF5 file.')
        else:
            raise RuntimeError('HDF output file exists!')

    # Warn of missing measurement metadata
    if args.fill == parser.get_default('fill'):
        print('WARNING: Fill number was not specified.')
    if args.description == parser.get_default('description'):
        print('WARNING: No measurement description given.')
    if args.short_description == parser.get_default('short_description'):
        print('WARNING: No short measurement description given.')

    # Read in the timestamps (catch errors early)
    timestamps = list()
    for i, file_path in enumerate(args.infile):
        if args.time_source == 'filename':
            try:
                timestamp = float(os.path.basename(file_path).split('_')[-1].split('.')[0])
            except IndexError or TypeError:
                raise RuntimeError
        elif args.time_source == 'creation_timestamp':
            timestamp = os.path.getctime(file_path)
        elif args.time_source == 'counter':
            timestamp = i
        else:
            timestamp = 0
        timestamps.append(timestamp + args.timeoffset_board)

    # Create HDF output file
    hdf_f = h5py.File(args.outfile, 'w-')
    hdf_f['/'].attrs['version'] = 1
    hdf_f['/'].attrs['description'] = args.description
    hdf_f['/'].attrs['short_description'] = args.short_description
    hdf_f['/'].attrs['fill_number'] = args.fill
    hdf_f.create_group('channels/')
    hdf_f['/dataset_timestamps'] = np.asarray(timestamps, dtype=float)
    hdf_shuffle = bool(args.compression)
    hdf_chunks = True if bool(args.compression) else None

    print(hdf_shuffle, hdf_chunks)

    # Remember all input options inside the HDF file
    cmd_param_group = hdf_f.create_group('import_options/cmd_parameters')
    for option, value in vars(args).items():
        if isinstance(value, (list, tuple)):
            # HDF5 attributes may also be arrays, but must be a numpy array to be saved correctly
            # cmd_param_group.attrs[option] = np.array(value)
            if len(value) > 500:
                print('filelist longer than 500 files, not saving filenames in attributes')
                cmd_param_group.attrs[option] = 'None'
            else:
                pass
                # TODO: (marvin) not sure whats going on here
                #cmd_param_group.attrs[option] = np.array(value)
                # cmd_param_group.attrs[option] = np.array(value)
        elif value is None:
            # None is not a valid HDF5 parameter value, so save it as a string
            cmd_param_group.attrs[option] = 'None'
        else:
            # All other case can be saved directly
            cmd_param_group.attrs[option] = value

    # Read in current if supplied
    if args.current_csv:
        print('Reading CSV current file...')
        current_readings = current2.CurrentReadings()
        current_readings.load(args.current_csv)

        if args.timeoffset_ph != 0:
            current_readings.timeoffset_PH(args.timeoffset_ph)

        # check if timestamps span same range as timestamps from datafiles
        sorted_timestamps = sorted(current_readings.readings.keys())
        assert min(sorted_timestamps) < min(timestamps) and max(timestamps) < max(sorted_timestamps), \
            'Data for bunch current does not cover complete time range of measurement! (%i - %i instead of %i - %i)' \
            % (min(sorted_timestamps), max(sorted_timestamps), min(timestamps), max(timestamps))

        print('Reading reference intensity file...')
        ref_file = args.current_roll_reference_file or args.infile[0]
        ref_adc = args.current_roll_reference_file_adc if args.current_roll_reference_file_adc is not None else used_adc_channels[0]
	

        raw_sample_data = filereader.read_raw_board_file(ref_file, num_adcs=args.number_adcs, read_adc=(ref_adc,),
                                                         detect_filling=args.detect_filling)
        reshaped_data = filereader.reshape_data_to_bunches(raw_sample_data['samples'])
        data = filereader.apply_normalization(reshaped_data, instrument)[ref_adc]

        roll_method = {
            'mean': lambda x: x.mean(axis=1, dtype=np.float64),
            'std': lambda x: x.std(axis=1, dtype=np.float64),
            'heb_inverted': lambda x: (((x - 2048) * -1.) + 2048).mean(1, dtype=np.float64)
        }[args.current_roll_method]

        print('Rolling current to reference...')
        current_readings.rotate_to_reference(roll_method(data), timestamps[0])

        current_data = hdf_f.create_dataset('beam/current', (len(current_readings.readings), const.ANKA.h + 1),
                                            dtype=np.float64, compression=args.compression, shuffle=hdf_shuffle,
                                            chunks=(True if bool(args.compression) else None))
        for i, timestamp in enumerate(sorted_timestamps):
            current_data[i, 0] = timestamp
            current_data[i, 1:] = current_readings.readings[timestamp]
        # TODO: nicer way to save attributes indicating if singlebunch, offset current and if subtracted
        current_data.attrs['singlebunch'] = current_readings.singlebunch
        current_data.attrs['offsetcurrent'] = current_readings.offsetcurrent
        current_data.attrs['offset_subtracted'] = current_readings.offset_subtracted

        print('Saved to HDF5 file.')

    if rolling:
        print('Reading reference file...')
        ref_file = args.roll_reference_raw or args.infile[0]
        ref_adc = args.reference_adc or 0

        ref_raw_sample_data = filereader.read_raw_board_file(ref_file, num_adcs=args.number_adcs, read_adc=(ref_adc,),
                                                             detect_filling=args.detect_filling)
        ref_reshaped_data = filereader.reshape_data_to_bunches(ref_raw_sample_data['samples'])
        ref_data = filereader.apply_normalization(ref_reshaped_data, instrument)[ref_adc]
        roll_method = {
            'mean': lambda x: x.mean(axis=1, dtype=np.float64),
            'std': lambda x: x.std(axis=1, dtype=np.float64),
            'heb_inverted': lambda x: (((x - 2048) * -1.) + 2048).mean(1, dtype=np.float64)
        }[args.roll_method]
        ref_profil = roll_method(ref_data)


    # Start reading the files
    nr_files = len(args.infile)
    start_time = time.time()
    for i, file_path in enumerate(args.infile):
        remaining_time = (time.time() - start_time) / i * (nr_files - i) if i > 0 else float('nan')
        print('Working on file "%s" (%0.2f%%, %0.2f seconds remaining)' % (os.path.basename(file_path),
                                                                           100. * i / nr_files, remaining_time))

        # Read in the data
        raw_sample_data = filereader.read_raw_board_file(file_path, num_adcs=args.number_adcs, read_adc=used_adc_channels,
                                                         detect_filling=args.detect_filling,
                                                         samples=args.max_samples * const.ANKA.h if args.max_samples else None)
        raw_header = raw_sample_data['header']
        reshaped_data = filereader.reshape_data_to_bunches(raw_sample_data['samples'])
        data = filereader.apply_normalization(reshaped_data, instrument)

        # Roll to reference:
        if rolling:
            print('rolling file')
            for adc_channel_data in data.values():
                adc_channel_data[...] = roll.rollData(adc_channel_data, ref_profil, args.roll_method)

        # Mask the buckets which we are not interested in.
        if dropped_buckets:
            for adc, adc_channel_data in data.items():
                adc_channel_data[dropped_buckets] = drop_buckets_value[adc]

        # Determine the number of skipped turns
        if args.skipped_turns is None and raw_header is None:
            raise RuntimeError('Number of skipped turns was not specified but there was also no header in the RAW file '
                               'to figure it out.')
        skipped_turns = args.skipped_turns
        if raw_header:
            if args.skipped_turns is not None and raw_header['skip_turns'] != args.skipped_turns:
                print('WARNING: Skipped turns specified in header (%i) '
                      'do not match the ones given on command line (%i)! '
                      'Using value from header.' % (raw_header['skip_turns'], args.skipped_turns))
            skipped_turns = raw_header['skip_turns']
            # TODO: find better way to change saved value of skipped_turns when used from header
            hdf_f['import_options/cmd_parameters'].attrs['skipped_turns'] = skipped_turns

        nr_samples = data[used_adc_channels[0]].shape[1]

        # Cut the channel data to be of power two
        if not args.no_cut_power_two:
            nr_samples = floor_power_two(nr_samples)
            for adc_channel, adc_channel_data in data.items():
                data[adc_channel] = adc_channel_data[:, :nr_samples]

        # Go through all the converters and save the resulting channel data
        for converter in channel_spec:
            print('Passing through converter "%s"...' % converter['converter'], end='')
            sys.stdout.flush()
            converted = converter['fun'](data, converter['in_channels'], converter['out_channels'],
                                         converter['options'], raw_header, skipped_turns, dropped_buckets, drop_buckets_value, instrument)
            print('CALCULATED! (Yields %s)...' % ', '.join(sorted(converted.keys())), end='')
            sys.stdout.flush()

            # On the first run, these channels do not yet exist, so they have to be created
            # Assume, that this data set is representative for all other data sets.
            for channel_name, channel_data in converted.items():
                if i == 0:
                    channel_group = hdf_f.create_group('channels/%s' % channel_name)

                    # Save channel data if available
                    if 'waveform' in channel_data:
                        dataset = channel_group.create_dataset('waveforms',
                                                               (nr_files, channel_data['waveform'].shape[0],
                                                                channel_data['waveform'].shape[1]),
                                                               dtype=channel_data['waveform'].dtype,
                                                               compression=args.compression,
                                                               compression_opts=1 if args.compression == 'gzip' else None,
                                                               shuffle=hdf_shuffle, chunks=(
                            (1, 1, channel_data['waveform'].shape[1]) if bool(args.compression) else None))
                        dataset[i] = channel_data['waveform']

                    if 'fft_abs' in channel_data:
                        dataset = channel_group.create_dataset('fft/abs',
                                                               (nr_files, channel_data['fft_abs'].shape[0],
                                                                channel_data['fft_abs'].shape[1]),
                                                               dtype=channel_data['fft_abs'].dtype,
                                                               compression=args.compression,
                                                               compression_opts=1 if args.compression == 'gzip' else None,
                                                               shuffle=hdf_shuffle, chunks=(
                            (1, 1, channel_data['fft_abs'].shape[1]) if bool(args.compression) else None))
                        dataset[i] = channel_data['fft_abs']
                        dataset.attrs['min'] = np.min(channel_data['fft_abs'])
                        dataset.attrs['max'] = np.max(channel_data['fft_abs'])

                    if 'fft_phase' in channel_data:
                        dataset = channel_group.create_dataset('fft/phase',
                                                               (nr_files, channel_data['fft_phase'].shape[0],
                                                                channel_data['fft_phase'].shape[1]),
                                                               dtype=channel_data['fft_phase'].dtype,
                                                               compression=args.compression,
                                                               compression_opts=1 if args.compression == 'gzip' else None,
                                                               shuffle=hdf_shuffle, chunks=(
                            (1, 1, channel_data['fft_phase'].shape[1]) if bool(args.compression) else None))
                        dataset[i] = channel_data['fft_phase']

                    if 'fft_freq' in channel_data:
                        dataset = channel_group.create_dataset('fft/freq',
                                                               (channel_data['fft_freq'].shape[0],),
                                                               dtype=channel_data['fft_freq'].dtype,
                                                               compression=args.compression,
                                                               compression_opts=1 if args.compression == 'gzip' else None,
                                                               shuffle=hdf_shuffle,
                                                               chunks=(True if bool(args.compression) else None))
                        dataset[...] = channel_data['fft_freq']

                    # Add channel meta information
                    channel_group.attrs['skipped_turns'] = skipped_turns
                    channel_group.attrs['unit'] = channel_data.get('unit', 'a.u.')
                    channel_group.attrs['short_unit'] = channel_data.get('short_unit', 'a.u.')
                    channel_group.attrs['name'] = channel_data.get('name', channel_name)
                    channel_group.attrs['short_name'] = channel_data.get('short_name', channel_name)
                else:
                    channel_group = hdf_f['channels/%s' % channel_name]
                    # The datasets already exist, so just save the data
                    for name, group_path in [('waveform', 'waveforms'),
                                             ('fft_abs', 'fft/abs'),
                                             ('fft_phase', 'fft/phase')]:
                        if name in channel_data:
                            channel_group[group_path][i] = channel_data[name]

                    if 'fft_abs' in channel_data:
                        dataset = channel_group['fft/abs']
                        dataset.attrs['min'] = min(dataset.attrs['min'], np.min(channel_data['fft_abs']))
                        dataset.attrs['max'] = max(dataset.attrs['max'], np.max(channel_data['fft_abs']))

            print('SAVED!')

    print('Finished - closing file.')
    hdf_f.flush()
    hdf_f.close()

    total_time = time.time() - start_time
    print('Whole import took %0.2f seconds. (%0.2f per file)' % (total_time, total_time / nr_files))


if __name__ == '__main__':
    main()
