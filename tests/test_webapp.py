'''
Test the webapp.
'''

import school_tools as sct


def test_introductions():
    ''' Test introductions calculator '''
    icalc, ax = sct.plot_introductions(es=4, ms=3, hs=2)
    return icalc, ax


if __name__ == '__main__':

    icalc, ax = test_introductions()