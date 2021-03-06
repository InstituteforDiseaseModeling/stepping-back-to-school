'''
Test the webapp.
'''

import sciris as sc
import school_tools as sct


def test_intro_calc():
    sc.heading('Testing introductions calculator...')
    fig, icalc = sct.multi_school_intro_api(web=False, es=4, ms=3, hs=2, prev=0.25, immunity=0.1, n_days=7, diagnostic='none', scheduling='none')
    data = icalc.to_json()
    return icalc, fig, data


def test_load_trimmed_pop(**kwargs):
    sc.heading('Testing loading a trimmed population...')
    pop = sct.load_trimmed_pop(pop_size=5e3, force=True, **kwargs)
    return pop


def test_outbreak_calc():
    sc.heading('Testing outbreak calculator...')
    fig, figs, ocalc = sct.outbreak_api(web=False, pop_size=10e3, prev=2.5, diagnostic='none', scheduling='none')
    data = ocalc.to_dict()
    return ocalc, fig, figs, data


if __name__ == '__main__':
    icalc, fig1, data1 = test_intro_calc()
    pop = test_load_trimmed_pop()
    ocalc, fig2, figs, data2 = test_outbreak_calc()