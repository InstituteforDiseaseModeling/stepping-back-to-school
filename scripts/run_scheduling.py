'''
Run a varitey of scheduling scenarios at a few prevalence levels
'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)
    sweep_pars = dict(schcfg_keys = ['with_countermeasures', 'all_hybrid', 'k5'])

    # Create and run
    mgr = sct.Manager(sweep_pars=sweep_pars, sim_pars=None, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    # mgr.regplots(xvar='Prevalence Target', huevar='Scenario') # CK: doesn't work
    # analyzer.introductions_rate(xvar='Prevalence Target', huevar='Scenario', height=5, aspect=2, ext='_wide') # CK: doesn't work
    # analyzer.cum_incidence(colvar='Prevalence Target') # CK: doesn't work
    # analyzer.introductions_rate_by_stype(xvar='Prevalence Target') # CK: doesn't work
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    mgr.tsplots()
