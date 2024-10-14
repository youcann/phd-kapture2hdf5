from __future__ import print_function, division

import sys
from distutils.util import strtobool

from PIL import Image
from PIL import PngImagePlugin
import numpy as np


def add_metadata(file_path, metadata):
    if file_path[-3:].lower() == 'png':
        im = Image.open(file_path)
        meta = PngImagePlugin.PngInfo()

        old_info = im.info
        try:
            for key, value in old_info.iteritems():
                meta.add_text(key, str(value))

            for key, value in metadata.iteritems():
                meta.add_text(key, str(value))

            im.save(file_path, "png", pnginfo=meta)
        except AttributeError as e:
            print("Command was not saved in image metadata!\nAttributeError: {0}".format(e))
    else:
        print('Command was not saved in image metadata! Because image is no png.')


def read_metadata(file_path):
    if file_path[-3:].lower() == 'png':
        im = Image.open(file_path)
        return im.info


def find_ranges(x):
    if len(x) < 2:
        return x
    diff1 = np.diff(x)
    assert (diff1 > 0).all(), 'Range must be continous and monotone rising'

    ranges = (diff1 == 1)

    found_ranges = list()
    in_range = ranges[0]
    range_start_idx = 0
    for i, new_in_range in enumerate(ranges):
        if not new_in_range and in_range:
            # Range ended
            found_ranges.append((x[range_start_idx], x[i]))

        if new_in_range and not in_range:
            # Range started
            range_start_idx = i

        if not new_in_range and not in_range:
            # Single freestandning value
            found_ranges.append(x[i])

        # Special handling of last value
        if i == len(ranges) - 1:
            if new_in_range:
                # Range ends here
                found_ranges.append((x[range_start_idx], x[i + 1]))
            else:
                # Single value
                found_ranges.append(x[i + 1])

        in_range = new_in_range

    return found_ranges


def describe_ranges(ranges, sep=', '):
    desc = []
    for r in ranges:
        if type(r) == tuple:
            desc.append('%i-%i' % r)
        else:
            desc.append('%s' % r)
    return sep.join(desc)


def description_to_ranges(description, sep=','):
    ranges = list()
    for range in description.split(sep):
        tmp = [x.strip() for x in range.split('-')]

        if len(tmp) == 1:
            ranges.append(int(tmp[0]))
        elif len(tmp) == 2:
            ranges.append((int(tmp[0]), int(tmp[1])))
        else:
            raise ValueError('Not a valid range description')
    return ranges


def ranges_to_numbers(ranges):
    results = list()
    for r in ranges:
        if isinstance(r, int):
            results.append(r)
        elif isinstance(r, (tuple, list)) and len(r) == 2:
            results.extend(range(r[0], r[1] + 1))
        else:
            raise ValueError('Not a valid range')
    return results


def user_yes_no_query(question):
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')


class cached_property(object):
    """
    A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::

        class Foo(object):

            @cached_property
            def foo(self):
                # calculate something important here
                return 42

    The class has to have a `__dict__` in order for this property to
    work.

    :copyright: (c) 2011 by the Werkzeug Team
    :license: BSD
    """

    # implementation detail: this property is implemented as non-data
    # descriptor.  non-data descriptors are only invoked if there is
    # no entry with the same name in the instance's __dict__.
    # this allows us to completely get rid of the access function call
    # overhead.  If one choses to invoke __get__ by hand the property
    # will still work as expected because the lookup logic is replicated
    # in __get__ for manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, '_missing')
        if value is '_missing':
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


if __name__ == '__main__':
    # Test ranges
    x = [1, 2, 3, 4, 7, 9, 10, 11, 12]
    ranges = find_ranges(x)
    described_ranges = describe_ranges(ranges)

    print(x)
    print(ranges, described_ranges)
    print(ranges_to_numbers(description_to_ranges(described_ranges)))

    metadata = {'command': ' '.join(['bla', 'Test3'])}
    add_metadata('/home/miriam/Analysis/bursting_analysis/test.png', metadata)
    im = Image.open('/home/miriam/Analysis/bursting_analysis/test.png')
    print(im.info)
