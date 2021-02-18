'''
Outbreak analysis with modified cohorting amongst students
'''

import sys
import numpy as np
import school_tools as sct

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import covasim as cv


if __name__ == '__main__':
    # Settings
    args = sct.config.process_inputs(sys.argv)

    rewire10  = sct.CohortRewiring(frac_edges_to_rewire = 0.10)
    rewire25  = sct.CohortRewiring(frac_edges_to_rewire = 0.25)
    #rewire50  = sct.CohortRewiring(frac_edges_to_rewire = 0.50)
    #rewire75  = sct.CohortRewiring(frac_edges_to_rewire = 0.75)
    #rewire100 = sct.CohortRewiring(frac_edges_to_rewire = 1.00)

    sweep_pars = {
        'n_prev': 0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        'cohort_rewiring': {
            'None': None,
            '10%':  [rewire10],
            '25%':  [rewire25],
        },
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 3)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = 'Cohort Rewiring'

    # Create and run
    mgr = sct.Manager(name='OutbreakCohorting', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    analyzer.outbreak_reg_facet(xvar, huevar, colvar='School Type', hue_order=['None', '10%', '25%'], height=6, aspect=1)
    exit()
    analyzer.outbreak_reg_facet(xvar, huevar, height=6, aspect=2.4)
    analyzer.outbreak_reg_facet(xvar, huevar, ext='ppt')

    g = analyzer.outbreak_size_stacked_distrib(xvar, rowvar=None, ext=None, height=6, aspect=2)

    # Plots
    g = analyzer.outbreak_multipanel(xvar, ext=None, jitter=0.2, values=None, legend=False, height=12, aspect=1.0) # height=10, aspect=0.7,

    analyzer.exports_reg(xvar, huevar)

    #analyzer.outbreak_reg_by_stype(xvar, height=6, aspect=1.4, ext='ppt', nboot=50, legend=True)
    #analyzer.outbreak_size_plot(xvar) #xvar, rowvar=None, ext=None, height=6, aspect=1.4, scatter=True, jitter=0.012
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()


    mgr.tsplots()
