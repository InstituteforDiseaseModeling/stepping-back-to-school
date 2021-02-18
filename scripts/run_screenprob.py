'''
Dense sweep of screen_prob at a few fixed prevalence levels.
'''

import os
import sys
import numpy as np
import school_tools as sct
import covasim as cv

if __name__ == '__main__':

    # Settings
    outbreak = None # Set to True to force the outbreak version
    args = sct.config.process_inputs(sys.argv, outbreak=outbreak)

    symp_screens = {p:{'screen_prob': p} for p in np.linspace(0, 1, 10)}
    xvar = 'Screen prob'

    if not args.outbreak:

        #sweep_pars = dict(prev=[0.005, 0.01])
        sweep_pars = dict(n_prev=3)
        levels = [{'keyname':'Screen prob', 'level':symp_screens, 'func':'screenpars_func'}]
        huevar = 'Prevalence Target'

        # Create and run
        mgr = sct.Manager(name='ScreenProb', sweep_pars=sweep_pars, sim_pars=None, levels=levels)
        mgr.run(args.force)
        analyzer = mgr.analyze()

        # Plots
        mgr.regplots(xvar=xvar, huevar=huevar, height=6, aspect=2.4)
        analyzer.introductions_rate(xvar=xvar, huevar=huevar, height=6, aspect=1.4, ext='ppt')
        analyzer.cum_incidence(colvar=xvar)
        analyzer.introductions_rate_by_stype(xvar=xvar)
        analyzer.outbreak_size_over_time()
        analyzer.source_pie()
        mgr.tsplots()

    else:
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
        levels = [
            {'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'},
            {'keyname':'Screen prob', 'level':symp_screens, 'func':'screenpars_func'},
        ]

        huevar = 'In-school transmission multiplier'

        # Create and run
        mgr = sct.Manager(name='OutbreakScreenProb', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
        mgr.run(args.force)
        analyzer = mgr.analyze()

        # Plots
        g = analyzer.outbreak_reg_facet(xvar, huevar, aspect=2)
        for ax in g.axes.flat:
            ax.set_xlim([0,1])
            ax.set_yticks([1, 5, 10, 15])
        fn = 'OutbreakSizeRegression.png'

        cv.savefig(os.path.join(analyzer.imgdir, fn), dpi=300)

        g = analyzer.outbreak_reg_facet(xvar, huevar, aspect=1.4, ext='ppt')

        analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
        analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
        analyzer.source_pie()
        mgr.tsplots()
