'''
Very simple test of sim creation and input argument handling (without inputs)
'''

import sys
import numpy as np
import sciris as sc
import covasim as cv
import school_tools as sct


def test_sims(do_plot=False):
    s1 = sct.create_sim(pop_size=3e3, load_pop=False, label='Default')
    s2 = sct.create_sim(pop_size=3e3, load_pop=False, rand_seed=23948, label='Different seed')
    s3 = sct.create_sim(pop_size=3e3, load_pop=False, beta=0.03, label='High beta')

    msim = cv.MultiSim([s1, s2, s3])
    msim.run()

    n_sims = len(msim.sims)
    infs = np.zeros(n_sims)
    for i in range(n_sims):
        infs[i] = msim.sims[i].summary['cum_infections']

    assert len(np.unique(infs)) == n_sims, 'Sims should not have the same results' # Ensure each is unique

    if do_plot:
        msim.plot()

    return msim


def test_inputs():
    sct.config.process_inputs(sys.argv)
    sct.config.set_full()
    p1 = sc.dcp(sct.config.sweep_pars)
    sct.config.set_debug()
    p2 = sc.dcp(sct.config.sweep_pars)
    assert p1.n_reps > p2.n_reps
    return


if __name__ == '__main__':
    msim = test_sims(do_plot=True)
    test_inputs()
