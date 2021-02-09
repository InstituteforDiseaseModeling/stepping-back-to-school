'''
Outbreak analysis to sweep in-school transmissibility
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

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 5)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = None

    # Create and run
    mgr = sct.Manager(name='OutbreakBetaSchool', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # g = analyzer.outbreak_size_stacked_distrib(xvar, rowvar=None, ext=None, height=6, aspect=2)

    # Plots
    g = analyzer.outbreak_multipanel(xvar, ext=None, jitter=0.2, values=None, legend=False, height=12, aspect=1.0) # height=10, aspect=0.7,

    # analyzer.exports_reg(xvar, huevar)
    # analyzer.outbreak_reg(xvar, ext='ppt')
    # analyzer.outbreak_reg(xvar, ext='ppt', by_stype=True)
    #analyzer.outbreak_reg_by_stype(xvar, height=6, aspect=1.4, ext='ppt', nboot=50, legend=True)
    #analyzer.outbreak_size_plot(xvar) #xvar, rowvar=None, ext=None, height=6, aspect=1.4, scatter=True, jitter=0.012
    # analyzer.cum_incidence(colvar=xvar)
    # analyzer.outbreak_size_over_time()
    # analyzer.source_pie()
    # mgr.tsplots()
