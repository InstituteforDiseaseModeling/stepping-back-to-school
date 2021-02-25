'''
Run a varitey of screening scenarios at a few prevalence levels
'''

import sys
import numpy as np
import school_tools as sct

if __name__ == '__main__':

    # Settings
    outbreak = None # Set to True to force the outbreak version
    args = sct.config.process_inputs(sys.argv, outbreak=outbreak)

    sweep_pars = dict(screen_keys =  ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'Antigen every 2w', 'Antigen every 1w', 'PCR every 1w'])
    huevar = 'Dx Screening'
    pop_size = sct.config.sim_pars.pop_size

    if not args.outbreak:
        xvar = 'Prevalence Target'

        # Create and run
        mgr = sct.Manager(name='Screening', sweep_pars=sweep_pars, sim_pars=None, levels=None)
        mgr.run(args.force)
        analyzer = mgr.analyze()

        # Plots
        mgr.regplots(xvar=xvar, huevar='Dx Screening', height=6, aspect=2.4)
        analyzer.introductions_rate(xvar=xvar, huevar='Dx Screening', height=6, aspect=1.4, ext='ppt')
        analyzer.cum_incidence(colvar=xvar)
        analyzer.introductions_rate_by_stype(xvar=xvar)
        analyzer.outbreak_size_over_time()
        analyzer.source_pie()
        mgr.tsplots()

    else:
        sweep_pars.update({
            'n_prev': 0, # No controller
            'school_start_date': '2021-02-01',
            'school_seed_date': '2021-02-01',
        })

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

        # Create and run
        mgr = sct.Manager(name='OutbreakScreening', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
        mgr.run(args.force)
        analyzer = mgr.analyze()

        # Plots
        analyzer.outbreak_reg_facet(xvar, huevar, aspect=2)
        analyzer.outbreak_reg_facet(xvar, huevar, aspect=1.4, ext='ppt')
        analyzer.cum_incidence(colvar=xvar, rowvar=huevar)
        analyzer.outbreak_size_over_time(colvar=xvar, rowvar=huevar)
        analyzer.source_pie()
        mgr.tsplots()

        args.handle_show()
