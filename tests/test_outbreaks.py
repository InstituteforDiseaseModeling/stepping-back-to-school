'''
Copy of outbreak script to use for testing.
'''

import numpy as np
import sciris as sc
import school_tools as sct


def test_micro():
    ''' Simplest possible test'''
    sct.config.set_micro()
    mgr = sct.Manager(cfg=sct.config)
    mgr.run(force=True)
    analyzer = mgr.analyze()
    return analyzer


def test_outbreaks():

    # Minimal example
    sct.config.sweep_pars.n_reps = 2
    sct.config.sweep_pars.n_seeds = 2
    sct.config.sim_pars.pop_size = 10_000
    sct.config.paths.inputs = sc.thisdir(None, 'inputs')
    sct.config.paths.outputs = sc.thisdir(None, 'outputs')
    sct.config.run_pars.parallel = False # Interferes with coverage calculation otherwise

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

    xvar='In-school transmission multiplier'
    huevar=None

    sct.create_pops(cfg=sct.config)
    mgr = sct.Manager(sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels, cfg=sct.config)
    mgr.run(force=True)
    analyzer = mgr.analyze()

    analyzer.outbreak_size_distribution(row=xvar, col=None)
    analyzer.outbreak_R0()
    analyzer.outbreak_reg(xvar, huevar)
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    mgr.tsplots()

    return mgr


if __name__ == '__main__':

    analyzer = test_micro()
    mgr = test_outbreaks()

