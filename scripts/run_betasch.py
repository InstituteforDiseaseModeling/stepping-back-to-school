'''
Dense sweep of in-school transmissibility (beta_s) at a few fixed prevalence levels.
'''

import sys
import numpy as np
import school_tools as sct

if __name__ == '__main__':

    # Settings

    args = sct.config.process_inputs(sys.argv)

    sim_pars = dict(pop_size=223_000, end_day='2021-04-30')
    sweep_pars = dict(n_reps=10, prev=[0.002, 0.007, 0.014])

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = None

    # Create and run
    mgr = sct.Manager(name='BetaSchool', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    mgr.regplots(xvar=xvar, huevar=huevar)
    analyzer.outbreak_reg(xvar=xvar, huevar=huevar, height=5, aspect=2, ext='_wide')
    analyzer.cum_incidence(colvar=xvar)
    analyzer.introductions_rate_by_stype(xvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    mgr.tsplots()
