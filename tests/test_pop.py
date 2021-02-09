'''
Very simple test of population generation
'''

import covasim_schools as cvsch


def test_school_pop(pop_size=5e3, do_plot=False):
    ''' Test basic population creation '''

    pop = cvsch.make_population(pop_size=pop_size, rand_seed=1, do_save=False)

    if do_plot:
        pop.plot()
        pop.plot_schools()

    return pop


if __name__ == '__main__':
    pop = test_school_pop(pop_size=20e3, do_plot=True)

