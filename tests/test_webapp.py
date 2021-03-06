'''
Test the webapp.
'''

import sciris as sc
import school_tools as sct


def test_intro_calc():
    sc.heading('Testing introductions calculator...')
    icalc, fig = sct.multi_school_intro_api(es=4, ms=3, hs=2, prev=0.25, immunity=0.1, n_days=7, diagnostic='none', scheduling='none')
    data = icalc.to_dict()
    return icalc, fig, data


def test_load_trimmed_pop(**kwargs):
    sc.heading('Testing loading a trimmed population...')
    pop = sct.load_trimmed_pop(pop_size=5e3, force=True, **kwargs)
    return pop


def test_outbreak_calc():
    sc.heading('Testing outbreak calculator...')
    ocalc, fig = sct.outbreak_api(pop_size=10e3, prev=2.5, diagnostic='none', scheduling='none')
    data = ocalc.to_dict()
    return ocalc, fig, data


if __name__ == '__main__':
    # icalc, fig, data = test_intro_calc()
    # pop = test_load_trimmed_pop()
    ocalc, fig, data = test_outbreak_calc()