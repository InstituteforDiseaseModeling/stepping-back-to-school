'''
Very simple test of population generation
'''

import sciris as sc
import covasim_schools as cvsch


def test_school_pop(pop_size=5e3, do_plot=False):
    ''' Test basic population creation '''

    pop = cvsch.make_population(pop_size=pop_size, rand_seed=1, do_save=False)

    if do_plot:
        pop.plot()
        pop.plot_schools()

    return pop


def test_pop_equality(pop_size=5e3):
    pops = sc.objdict()
    pops.seed1a = cvsch.make_population(pop_size=pop_size, rand_seed=1, do_save=False)
    pops.seed1b = cvsch.make_population(pop_size=pop_size, rand_seed=1, do_save=False)
    pops.seed2  = cvsch.make_population(pop_size=pop_size, rand_seed=2, do_save=False)

    n = sc.objdict()
    for k,pop in pops.items():
        n[k] = len(pop.contacts)
        print(f'Population {k} has {n[k]} contacts')

    assert (n.seed1a == n.seed1b) and (n.seed1a != n.seed2)

    return


if __name__ == '__main__':
    pop = test_school_pop(pop_size=20e3, do_plot=True)
    test_pop_equality()

