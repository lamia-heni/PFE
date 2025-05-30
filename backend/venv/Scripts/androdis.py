#!c:\users\lamia\desktop\backend\android-malware-detection-with-machine-learning\proposedapproach\venv\scripts\python.exe

# This file is part of Androguard.
#
# Copyright (C) 2012, Axelle Apvrille <aafortinet at gmail.com>
#                     Anthony Desnos <desnos at t0t0.fr>
# All rights reserved.
#
# Androguard is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Androguard is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Androguard.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
import sys

from optparse import OptionParser

from androguard.cli import androdis_main

option_0 = {
    'name': ('-i', '--input'),
    'help': 'file : use this filename (DEX/ODEX)',
    'nargs': 1
}
option_1 = {
    'name': ('-o', '--offset'),
    'help': 'offset to disassemble',
    'nargs': 1
}
option_2 = {'name': ('-s', '--size'), 'help': 'size', 'nargs': 1}

options = [option_0, option_1, option_2]


def main(options, arguments):
    if options.input and options.offset and options.size:
        offset = int(options.offset, 0)
        size = int(options.size, 0)
        androdis_main(offset, size, options.input)


if __name__ == "__main__":
    parser = OptionParser()
    for option in options:
        param = option['name']
        del option['name']
        parser.add_option(*param, **option)

    options, arguments = parser.parse_args()
    sys.argv[:] = arguments
    main(options, arguments)
