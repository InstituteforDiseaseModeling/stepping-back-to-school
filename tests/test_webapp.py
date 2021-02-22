'''
Test the webapp.
'''

import school_tools as sct


def test_intro_calc():
    ''' Test introductions calculator '''
    icalc, ax = sct.plot_introductions(es=4, ms=3, hs=2)
    return icalc, ax

def test_load_trimmed_pop(**kwargs):
    pop = sct.load_trimmed_pop(pop_size=5e3, force=True, **kwargs)
    return pop


def test_outbreak_calc(**kwargs):
    ''' Test tree plotting '''
    ocalc, ax = sct.plot_outbreaks(**kwargs)
    return ocalc, ax

if __name__ == '__main__':

    # icalc, ax = test_intro_calc()
    # pop = test_load_trimmed_pop()
    ocalc, ax = test_outbreak_calc()