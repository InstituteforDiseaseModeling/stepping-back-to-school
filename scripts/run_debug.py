'''
Debugging
'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings

    args = sct.config.process_inputs(sys.argv)

    sweep_pars = {
        'n_reps':       1,
        'n_prev':       3,
        'screen_keys':  ['None'],
        'schcfg_keys':  ['k5']
    }

    sim_pars = dict(end_day='2021-04-30')

    npi_scens = {x:{'beta_s': 1.5*x} for x in [0.75, 2]}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = 'Prevalence'

    # Create and run
    mgr = sct.Manager(sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    # mgr.regplots(xvar=xvar, huevar=huevar)  # CK: doesn't work, "ValueError: Input contains NaN, infinity or a value too large for dtype('float64')."
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    mgr.tsplots()
