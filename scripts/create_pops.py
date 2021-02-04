'''
Pre-generate the synthpops population including school types. Takes ~15s per seed
if running with 50,000 people; or roughly 5 times longer for 223,000 people.

Warning: this script is quite memory intensive. It should pick the right degree
of parallelization so you do not run out of RAM, but be warned.

To run with a different location or size, you can specify these as follows, e.g.

    python create_pops.py --pop_size 35 --location Spokane_County
'''

import sys
import school_tools as sct


# This must be in a main block for parallelization to work on Windows
if __name__ == '__main__':

    sct.config.process_inputs(sys.argv)
    sct.create_pops()

