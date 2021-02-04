'''
Run a varitey of screening scenarios at a few prevalence levels
'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)
    sweep_pars = dict(screen_keys =  ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'Antigen every 2w', 'Antigen every 1w', 'PCR every 1w'])

    # Create and run
    mgr = sct.Manager(sweep_pars=sweep_pars, sim_pars=None, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    # mgr.regplots(xvar='Prevalence Target', huevar='Scenario') # CK: doesn't work
    # analyzer.introductions_rate(xvar='Prevalence Target', huevar='Dx Screening', height=5, aspect=2, ext='_wide') # CK: doesn't work
    # analyzer.cum_incidence(colvar='Prevalence Target') # CK: doesn't work
    # analyzer.introductions_rate_by_stype(xvar='Prevalence Target') # CK: doesn't work
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    mgr.tsplots()
