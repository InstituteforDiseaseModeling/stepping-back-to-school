'''
Run school outbreak scenarios at a variety of locations around Washington State
'''

import sys
import numpy as np
import school_tools as sct

if __name__ == '__main__':

    # Settings
    outbreak = None # Set to True to force the outbreak version
    args = sct.config.process_inputs(sys.argv, outbreak=outbreak)

    if not args.outbreak:
        sweep_pars = dict(location =  ['seattle_metro', 'Spokane_County', 'Franklin_County', 'Island_County'])
        xvar = 'Prevalence Target'

        # Create and run
        mgr = sct.Manager(name='Location', sweep_pars=sweep_pars, sim_pars=None, levels=None)
        mgr.run(args.force)
        analyzer = mgr.analyze()

        # Plots
        analyzer.introductions_rate(xvar=xvar, huevar='Location', height=6, aspect=1.4, ext='_ppt')
        mgr.regplots(xvar=xvar, huevar='Location', height=6, aspect=2.4)

        analyzer.introductions_rate(xvar=xvar, huevar='Location', height=5, aspect=2, ext='_wide')
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
            #'screen_keys':  ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'Antigen every 2w', 'Antigen every 1w', 'PCR every 1w'],
            'location': ['seattle_metro', 'Spokane_County', 'Franklin_County', 'Island_County'],
        }

        pop_size = sct.config.sim_pars.pop_size
        sim_pars = {
            'pop_infected': 0, # Do not seed
            'pop_size': pop_size,
            'start_day': '2021-01-31',
            'end_day': '2021-08-31',
            'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
        }

        npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
        levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

        xvar = 'In-school transmission multiplier'
        huevar = 'Location'

        # Create and run
        mgr = sct.Manager(name='OutbreakLocations', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
        mgr.run(args.force)
        analyzer = mgr.analyze()

        # Plots
        #analyzer.outbreak_multipanel(row='Dx Screening', col='In-school transmission multiplier')
        g = analyzer.outbreak_reg_facet(xvar, huevar, aspect=2)
        g = analyzer.outbreak_reg_facet(xvar, huevar, aspect=1.4, ext='ppt')
        #for ax in g.axes.flat:
        #    ax.set_xlim([0,1])
        #    ax.set_yticks([1, 5, 10, 15])
        #fn = 'OutbreakSizeRegression.png'
        #import os, covasim as cv
        #cv.savefig(os.path.join(analyzer.imgdir, fn), dpi=300)
        # analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
        # analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
        # analyzer.source_pie()
        # mgr.tsplots()
