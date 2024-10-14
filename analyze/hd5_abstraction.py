from __future__ import print_function, division

import h5py
import os.path
import numpy as np
import warnings
import datetime

from current2 import CurrentReadings
from helpers import cached_property
from analyze import constants

try:
    import tqdm

    trange = tqdm.trange
except ImportError:
    warnings.warn('No tqdm - no progress report for derived values!')
    trange = xrange


class Channel(object):
    def __init__(self, hdf5_group, overwrite_attributes=None, internal_name='UNKOWN'):
        self._hdf5_group = hdf5_group
        self.overwrite_attributes = overwrite_attributes or dict()
        self._turns = None
        self._internal_name = internal_name

    @property
    def internal_name(self):
        return self._internal_name

    @property
    def short_name(self):
        return self.overwrite_attributes.get('short_name') or self._hdf5_group.attrs['short_name']

    @property
    def name(self):
        return self.overwrite_attributes.get('name') or self._hdf5_group.attrs['name']

    @property
    def short_unit(self):
        return self.overwrite_attributes.get('short_unit') or self._hdf5_group.attrs['short_unit']

    @property
    def unit(self):
        return self.overwrite_attributes.get('unit') or self._hdf5_group.attrs['unit']

    @property
    def has_waveform(self):
        return 'waveforms' in self._hdf5_group

    @property
    def waveform(self):
        if not self.has_waveform:
            raise IndexError('Channel has no wave form data')
        return self._hdf5_group['waveforms']

    @staticmethod
    def apply_over_dataset(data, fun, dtype=np.float64, args=(), result_type='per_bucket'):
        assert result_type in ['per_bucket', 'per_sample']
        nr_columns = {'per_bucket': data.shape[1], 'per_sample': data.shape[2]}[result_type]
        reduced = np.zeros((data.shape[0], nr_columns), dtype=dtype)
        for i in trange(data.shape[0]):
            reduced[i] = fun(data[i, :, :], *args)
        return reduced

    @cached_property
    def waveform_mean(self):
        mean = self._derived_property('waveform_mean',
                                      lambda: self.apply_over_dataset(self.waveform,
                                                                      lambda x: np.mean(x, axis=1, dtype=np.float64)))
        return mean

    @cached_property
    def waveform_std(self):
        std = self._derived_property('waveform_std',
                                     lambda: self.apply_over_dataset(self.waveform,
                                                                     lambda x: np.std(x, axis=1, dtype=np.float64)))
        return std

    @property
    def has_fft_abs(self):
        return 'fft/abs' in self._hdf5_group

    @property
    def fft_abs(self):
        if not self.has_fft_abs:
            raise IndexError('Channel has no absolute fft data')
        return self._hdf5_group['fft/abs']

    def get_fft_abs_mean(self, min_freq, max_freq):
        min_idx, max_idx = (self.get_fft_bin(freq) for freq in (min_freq, max_freq))
        mean = self._derived_property('fft_abs_mean_%i_%i' % (min_idx, max_idx),
                                      lambda: self.apply_over_dataset(self.fft_abs,
                                                                      lambda x, imin, imax: np.mean(x[:, imin:imax],
                                                                                                    axis=1,
                                                                                                    dtype=np.float64),
                                                                      args=(min_idx, max_idx)))
        return mean

    def get_fft_abs_rms(self, min_freq, max_freq):
        min_idx, max_idx = (self.get_fft_bin(freq) for freq in (min_freq, max_freq))
        mean = self._derived_property('fft_abs_rms_%i_%i' % (min_idx, max_idx),
                                      lambda: self.apply_over_dataset(self.fft_abs,
                                                                      lambda x, imin, imax: np.sqrt(
                                                                          np.mean(x[:, imin:imax] ** 2,
                                                                                  axis=1,
                                                                                  dtype=np.float64)),
                                                                      args=(min_idx, max_idx)))
        return mean

    def get_fft_abs_sum(self, min_freq, max_freq):
        min_idx, max_idx = (self.get_fft_bin(freq) for freq in (min_freq, max_freq))
        sum = self._derived_property('fft_abs_sum_%i_%i' % (min_idx, max_idx),
                                     lambda: self.apply_over_dataset(self.fft_abs,
                                                                     lambda x, imin, imax: np.sum(x[:, imin:imax],
                                                                                                  axis=1,
                                                                                                  dtype=np.float64),
                                                                     args=(min_idx, max_idx)))
        return sum

    def get_fft_abs_std(self, min_freq, max_freq):
        min_idx, max_idx = (self.get_fft_bin(freq) for freq in (min_freq, max_freq))
        sum = self._derived_property('fft_abs_std_%i_%i' % (min_idx, max_idx),
                                     lambda: self.apply_over_dataset(self.fft_abs,
                                                                     lambda x, imin, imax: np.std(x[:, imin:imax],
                                                                                                  axis=1,
                                                                                                  dtype=np.float64),
                                                                     args=(min_idx, max_idx)))
        return sum

    @property
    def fft_abs_total_min_max(self):
        if not self.has_fft_abs:
            raise IndexError('Channel has no absolute fft data')

        return [self._hdf5_group['fft/abs'].attrs[x] for x in ('min', 'max')]

    @property
    def has_fft_phase(self):
        return 'fft/phase' in self._hdf5_group

    @property
    def fft_phase(self):
        if not self.has_fft_phase:
            raise IndexError('Channel has no phase fft data')
        return self._hdf5_group['fft/phase']

    def get_fft_phase_mean(self, min_freq, max_freq):
        # FIXME: We might need to unwrap the phase here....
        min_idx, max_idx = (self.get_fft_bin(freq) for freq in (min_freq, max_freq))
        mean = self._derived_property('fft_phase_mean_%i_%i' % (min_idx, max_idx),
                                      lambda: self.apply_over_dataset(self.fft_phase[:, :, min_idx:max_idx],
                                                                      lambda x: np.mean(x, axis=1, dtype=np.float64)))
        return mean

    @cached_property
    def fft_freq(self):
        if not (self.has_fft_abs or self.has_fft_phase):
            raise IndexError('Channel has no fft data')
        return np.asarray(self._hdf5_group['fft/freq'])

    def get_fft_bin(self, freq, search_type='closest'):
        """
        Find the FFT bin for a given frequency. The bin is either the
        ``closest``, the ``left`` or the ``right`` bin if there is no exact match.
        """

        if freq < self.fft_freq[0] or freq > self.fft_freq[-1]:
            raise IndexError('Frequency outside of valid range')

        lookup_funs = {
            'left': lambda: np.where(freq >= self.fft_freq)[0][-1],
            'right': lambda: np.where(self.fft_freq >= freq)[0][0],
            'closest': lambda: np.argmin(np.abs(self.fft_freq - freq))
        }

        return lookup_funs[search_type]()

    @property
    def nr_samples(self):
        return self.waveform.shape[2]

    @property
    def turns(self):
        if self._turns is None:
            self._turns = np.arange(self.nr_samples) * self.sample_every_nth_turn
        return self._turns

    @property
    def skipped_turns(self):
        return self._hdf5_group.attrs['skipped_turns']

    @property
    def sample_every_nth_turn(self):
        return self.skipped_turns + 1

    @property
    def metadata(self):
        return dict(self._hdf5_group.attrs)

    def __repr__(self):
        return '<Channel %s, unit="%s">' % (self.name, self.unit)

    def _derived_property(self, name, fun):
        # This function looks up calculated values derived from actual measurement data
        # if the value is not defined in the HDF file, it's calculated and saved if possible
        path = 'derived/%s' % name
        try:
            data = np.asarray(self._hdf5_group[path])
            return data

        except KeyError:
            data = fun()

            # Save it into the HDF5 file if possible
            if self._hdf5_group.file.mode == 'r+':
                self._hdf5_group[path] = data
            else:
                warnings.warn(
                    'Derived value "%s" was calculated on the fly but cannot be saved to readonly HDF5' % name)
            return data


class Measurement(object):
    def __init__(self, hdf5_path, writable=False):
        self._hdf_f = h5py.File(hdf5_path, 'r' if not writable else 'r+')
        self._current = None
        self._chosen_current = None

        if self.hdf5_version == 0:
            # HDF5 files of version 0 only have one channel

            sel_adcs = self._hdf_f['intensity'].attrs['sel_adcs'].split(',')
            if len(sel_adcs) == 1:
                channel_name = 'ADC_' + sel_adcs[0]

            elif len(sel_adcs) == 0:
                channel_name = 'UNKOWN'

            else:
                channel_name = 'MEAN_ADC_' + '_'.join(['ADC%s' % adc for adc in sel_adcs])

            channel = Channel(hdf5_group=self._hdf_f['intensity'],
                              overwrite_attributes={
                                  'name': channel_name,
                                  'short_name': channel_name,
                                  'unit': 'ADC value',
                                  'short_unit': 'ADC'
                              }, internal_name=channel_name)
            self._channels = {channel_name: channel}
        else:
            # New HDF5 file versions save each channel independently
            # in /channels/{CHANNEL_NAME}/

            self._channels = dict()
            for channel_name, channel_group in self._hdf_f['/channels/'].items():
                assert isinstance(channel_group, h5py.Group), 'Expected only Groups in /channels/'
                self._channels[channel_name] = Channel(channel_group, internal_name=channel_name)

    @property
    def hdf5_version(self):
        try:
            return self._hdf_f['/'].attrs['version']
        except KeyError:
            # The first version did not include a version number
            return 0

    @property
    def base_filename(self):
        return '-'.join(os.path.basename(self._hdf_f.filename).split('.')[:-1])

    @property
    def description(self):
        if self.hdf5_version == 0:
            return self.base_filename

        return self._hdf_f.attrs['description']

    @property
    def short_description(self):
        if self.hdf5_version == 0:
            return self.description

        return self._hdf_f.attrs['short_description']

    @property
    def fill_number(self):
        if self.hdf5_version == 0:
            desc = self.description

            fill_str = desc.split('_')[0]
            if fill_str[0] != 'f':
                return None

            try:
                return int(fill_str[1:])
            except ValueError:
                raise RuntimeError('Fill number is not an integer')

        return int(self._hdf_f.attrs['fill_number'])

    @property
    def available_channels(self):
        return self._channels.keys()

    def get_channel(self, channel_name):
        try:
            return self._channels[channel_name]
        except:
            raise KeyError('Channel "{0}" does not exist. Use one of {1}'.format(channel_name,
                                                                                 ', '.join(self.available_channels)))

    @property
    def has_current(self):
        return any([i.startswith("current") for i in self._hdf_f['beam'].keys()]) if 'beam' in self._hdf_f else False

    @property
    def current_options(self):
        assert 'beam' in self._hdf_f, 'no dataset with bunchcurrent in hdf5 file'
        return [i for i in self._hdf_f['beam'].keys() if i.startswith("current")]
    
    def choose_current(self, dataset_name):
        assert dataset_name in self.current_options, 'requested current dataset is not available; choose from ' + str(self.current_options)
        self._current = None
        self._chosen_current = dataset_name
        self.current
        return self._current

    @property
    def current(self):
        if not self.has_current:
            raise IndexError('Measurement does not have current readings')

        if not self._current:
            current_readings = CurrentReadings()
            
            current_group_list =[ i for i in self._hdf_f['beam'].keys() if i.startswith("current")]
            print('The following datasets for bunch currents are available in the hdf5 file: ' + str(current_group_list))
            #assert 'current' or 'current_deadtime_corrected' in current_group_list, 'No standard bunch current dataset is available in hdf5 file.'
            if self._chosen_current:
                current_group = self._hdf_f['beam/'+str(self._chosen_current)]
            elif 'current' in current_group_list:
                current_group = self._hdf_f['beam/current']
            elif 'current_deadtime-corrected' in current_group_list:
                current_group = self._hdf_f['beam/current_deadtime-corrected']
                print('Using current_deadtime-corrected')
            else: 
                raise IndexError('No standard bunch current dataset is available in hdf5 file. Please choose one from ' + str(current_group_list)) 
            
            assert current_group.shape[1] == constants.ANKA.h + 1, 'Not the correct number of columns'
            for row in range(current_group.shape[0]):
                current_readings.readings[current_group[row, 0]] = current_group[row, 1:]

            # TODO: nicer way to load attributes indicating if singlebunch, offset current and if subtracted
            for key, value in current_group.attrs.iteritems():
                setattr(current_readings, key, value)
            ## current_readings.singlebunch = current_group.attrs['singlebunch']
            ## current_readings.offsetcurrent = current_group.attrs['offsetcurrent']
            ## current_readings.offset_subtracted = current_group.attrs['offset_subtracted']
            ## TODO: there is no way of writing changes in the measurement.current object back to the hdf5 file!?!

            current_readings.set_fillnumber(self.fill_number)
            self._current = current_readings

        return self._current

    @cached_property
    def dataset_timestamps(self):
        path = '/dataset_timestamps' if self.hdf5_version > 0 else '/intensity/timestamp'
        return np.asarray(self._hdf_f[path])

    @staticmethod
    def unixToLocalTime(unixtimestamp, filename=False):
        time = datetime.datetime.fromtimestamp(int(unixtimestamp))
        return time.strftime('%Y-%m-%dT%Hh%Mm%Ss') if filename else time.strftime('%Y/%m/%d %H:%M:%S')

    @cached_property
    def timestrings(self):
        return np.array([self.unixToLocalTime(stamp) for stamp in self.dataset_timestamps])

    @cached_property
    def timestrings_for_filename(self):
        return np.array([self.unixToLocalTime(stamp, filename=True) for stamp in self.dataset_timestamps])

    @property
    def nr_datasets(self):
        # TODO: Find nicer ways for this
        return self.dataset_timestamps.shape[0]


def _test():
    path = '/mnt/usbraid/f05110_2014-04-16_SD_squeeze-scan_15_float32.hdf5'
    measurement = Measurement(hdf5_path=path, writable=False)

    print(measurement.hdf5_version)
    print(measurement.available_channels)
    print(measurement.description)
    print(measurement.fill_number)
    print(measurement.has_current)
    print(measurement.current)

    print(measurement.current_options)
    print(measurement.choose_current('current_uncorrected'))
    print(measurement._chosen_current)


    #channel = measurement.get_channel(measurement.available_channels[0])
    #print(channel)

    #from pprint import pprint
    #pprint(channel.metadata)
    #print(channel.metadata)
    # print(channel.fft_abs_total_min_max)

    #print(channel.internal_name)

    # print(channel.waveform_mean)
    # #print(channel.waveform_std.shape)
    # print(np.asarray(channel.fft_freq))
    # print(channel.get_fft_bin(3, 'left'))
    # print(channel.get_fft_bin(3, 'right'))
    # print(channel.get_fft_bin(5.3, 'closest'))
    #
    # print(channel.get_fft_abs_mean(0, 10))
    # print(channel.get_fft_abs_sum(0, 10))
    #
    # def list_prop(obj):
    #     for key in dir(obj):
    #         if key[0] == '_':
    #             continue
    #
    #         print(key, getattr(obj, key))
    #
    # list_prop(measurement)
    
    #print(measurement.dataset_timestamps)
    #print(measurement.timestrings)
    #print(measurement.timestrings_for_filename)


if __name__ == '__main__':
    _test()
