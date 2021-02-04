'''
Run a varitey of screening scenarios at a few prevalence levels
'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)
    sweep_pars = dict(screen_keys =  ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'Antigen every 2w', 'Antigen every 1w', 'PCR every 1w'])
    xvar = 'Prevalence Target'

    # Create and run
    mgr = sct.Manager(name='Screening', sweep_pars=sweep_pars, sim_pars=None, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    mgr.regplots(xvar=xvar, huevar='Scenario')
    analyzer.introductions_rate(xvar=xvar, huevar='Dx Screening', height=5, aspect=2, ext='_wide')
    analyzer.cum_incidence(colvar=xvar)
    analyzer.introductions_rate_by_stype(xvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    mgr.tsplots()
