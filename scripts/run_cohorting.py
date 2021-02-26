'''
Explore modified cohorting amongst students
'''

import sys
import numpy as np
import school_tools as sct

if __name__ == '__main__':
    # Settings
    outbreak = None # Set to True to force the outbreak version
    args = sct.config.process_inputs(sys.argv, outbreak=outbreak)

    rewire10  = sct.CohortRewiring(frac_edges_to_rewire = 0.10)
    rewire25  = sct.CohortRewiring(frac_edges_to_rewire = 0.25)

    sweep_pars = {
        'cohort_rewiring': {
            'None': None,
            '10%':  [rewire10],
            '25%':  [rewire25],
        },
    }

    pop_size = sct.config.sim_pars.pop_size
    huevar = 'Cohort Mixing'

    if not args.outbreak:
        msg = 'This script is only intended for outbreak analysis; please set outbreak=True or use the --outbreak flag'
        raise NotImplementedError(msg)

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
        mgr = sct.Manager(name='OutbreakCohorting', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
        mgr.run(args.force)
        analyzer = mgr.analyze()

        # Plot by school type
        col_order = ['Elementary', 'Middle']# , 'High'] # Drop high as not cohorted in the first place
        analyzer.outbreak_reg_facet(xvar, huevar, colvar='School Type', col_order=col_order, hue_order=['None', '10%', '25%'], height=6, aspect=1.2, ext='by_stype')

        # Standardized plots
        analyzer.outbreak_size_distrib(xvar, rowvar=None, ext=None, height=6, aspect=2)
        analyzer.outbreak_multipanel(xvar, ext=None, jitter=0.2, values=None, legend=False, height=12, aspect=1.0) # height=10, aspect=0.7,
        analyzer.exports_reg(xvar, huevar)
        analyzer.outbreak_reg_facet(xvar, huevar, height=6, aspect=2.4)
        analyzer.outbreak_reg_facet(xvar, huevar, ext='ppt')
        #analyzer.outbreak_reg_by_stype(xvar, height=6, aspect=1.4, ext='ppt', nboot=50, legend=True)
        #analyzer.outbreak_size_plot(xvar) #xvar, rowvar=None, ext=None, height=6, aspect=1.4, scatter=True, jitter=0.012
        analyzer.cum_incidence(colvar=xvar)
        analyzer.outbreak_size_over_time()
        analyzer.source_pie()
        mgr.tsplots()

        args.handle_show()
