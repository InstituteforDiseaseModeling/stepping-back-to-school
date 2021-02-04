'''
Outbreak analysis to sweep in-school transmissibility and screening probability
'''

import sys
import numpy as np
import school_tools as sct

if __name__ == '__main__':

    # Settings

    args = sct.config.process_inputs(sys.argv)

    sweep_pars = {
        'n_prev': 0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    npi_scens = {x:{'beta_s': 1.5*x} for x in [0.5, 0.75, 1.0, 1.25, 1.5]}
    symp_screens = {p:{'screen_prob': p} for p in np.linspace(0, 1, 10)}
    levels = [
        {'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'},
        {'keyname':'Screen prob', 'level':symp_screens, 'func':'screenpars_func'},
    ]

    xvar = 'Screen prob'
    huevar = 'In-school transmission multiplier'

    # Create and run
    mgr = sct.Manager(name='OutbreakScreenProb', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    analyzer.outbreak_reg(xvar, huevar)
    analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
    analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
    analyzer.source_pie()
    mgr.tsplots()
