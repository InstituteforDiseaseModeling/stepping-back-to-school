'''
Test the webapp.
'''

import school_tools as sct


def test_intro_calc():
    ''' Test introductions calculator '''
    icalc, fig = sct.plot_introductions(es=4, ms=3, hs=2)
    return icalc, fig

def test_load_trimmed_pop(**kwargs):
    pop = sct.load_trimmed_pop(pop_size=5e3, force=True, **kwargs)
    return pop


def test_outbreak_calc(**kwargs):
    ''' Test tree plotting '''
    ocalc, fig = sct.plot_outbreaks(**kwargs)
    data = ocalc.to_dict()
    return ocalc, fig, data

if __name__ == '__main__':

    # icalc, fig = test_intro_calc()
    # pop = test_load_trimmed_pop()
    ocalc, fig, data = test_outbreak_calc()