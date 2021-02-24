'''
Copy of outbreak script to use for testing.
'''

import numpy as np
import school_tools as sct

sct.config.run_pars.parallel = False # Interferes with coverage calculation otherwise, even with pytest-cov and if concurrency = multiprocess is included

def test_analysis(pop_size=None):
    ''' Complete outbreak/API/plotting example '''
    sct.config.sweep_pars.n_reps = 2
    sct.config.sweep_pars.n_seeds = 2
    sct.config.sim_pars.pop_size = 5_000 if pop_size is None else pop_size

    sweep_pars = {
        'n_prev':  1, # Include controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-03-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 2)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = 'Prevalence Target'

    sct.create_pops(cfg=sct.config)
    mgr = sct.Manager(sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels, cfg=sct.config)
    mgr.run(force=True)
    analyzer = mgr.analyze()

    analyzer.outbreak_size_distribution(xvar=xvar)
    analyzer.outbreak_R0()
    analyzer.outbreak_reg(xvar=xvar)
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    analyzer.source_dow()
    analyzer.introductions_rate(xvar=xvar, huevar=huevar)
    analyzer.introductions_rate_by_stype(xvar=xvar)
    analyzer.outbreak_size_plot(xvar=xvar, loess=True, scatter=True)
    analyzer.exports_reg(xvar=xvar, huevar=huevar)

    mgr.tsplots()

    return mgr


def test_micro():
    ''' Simplest possible test'''
    sct.config.set_micro()
    mgr = sct.Manager(cfg=sct.config)
    mgr.run(force=True)
    mgr.analyze()
    sct.config.set_default() # Back to default
    return mgr


def test_scheduling():
    ''' Test classroom scheduling scenarios '''
    sct.config.set_micro()
    sweep_pars = dict(schcfg_keys = ['with_countermeasures', 'all_hybrid', 'k5'])
    mgr = sct.Manager(sweep_pars=sweep_pars, cfg=sct.config)
    mgr.run(force=True)
    sct.config.set_default() # Back to default
    return mgr


def test_testing():
    ''' Test COVID testing scenarios '''
    sct.config.set_micro()
    sweep_pars = dict(screen_keys =  ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'PCR every 1w'])
    mgr = sct.Manager(sweep_pars=sweep_pars, cfg=sct.config)
    mgr.run(force=True)
    sct.config.set_default() # Back to default
    return mgr


def test_alt_sus():
    ''' Test alternate symptomaticity '''
    sct.config.set_micro()
    sct.config.sweep_pars.alt_sus = True
    mgr = sct.Manager(cfg=sct.config)
    mgr.run(force=True)
    sct.config.set_default() # Back to default
    sct.config.sweep_pars.alt_sus = False
    return mgr


def test_trees():
    ''' Test tree plotting '''
    sct.config.set_micro()
    mgr = sct.Manager(cfg=sct.config)
    mgr.run(force=True)
    analyzer = mgr.analyze()
    analyzer.plot_tree()
    sct.config.set_default() # Back to default
    return mgr


if __name__ == '__main__':

    mgr1 = test_analysis(pop_size=10_000) # If run interactively, use a larger population size
    mgr2 = test_micro()
    mgr3 = test_scheduling()
    mgr4 = test_testing()
    mgr5 = test_alt_sus()
    mgr6 = test_trees()


