'''
Very simple test of sim creation -- not part of the main library
'''

import covasim as cv
import school_tools as sct

def test_sims(do_plot=False):
    s1 = sct.create_sim(pop_size=3e3, load_pop=False, label='Default')
    s2 = sct.create_sim(pop_size=3e3, load_pop=False, rand_seed=23948, label='Different seed')
    s3 = sct.create_sim(pop_size=3e3, load_pop=False, beta=0.03, label='High beta')

    msim = cv.MultiSim([s1, s2, s3])
    msim.run()

    if do_plot:
        msim.plot()

    return msim

if __name__ == '__main__':
    msim = test_sims(do_plot=True)
