'''
Dense sweep of screen_prob at a few fixed prevalence levels.
'''

import sys
import numpy as np
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)
    sweep_pars = dict(prev=[0.005, 0.01])

    symp_screens = {p:{'screen_prob': p} for p in np.linspace(0, 1, 10)}
    levels = [{'keyname':'Screen prob', 'level':symp_screens, 'func':'screenpars_func'}]

    xvar = 'Screen prob'
    huevar = 'Prevalence Target'

    # Create and run
    mgr = sct.Manager(ame='ScreenProb', sweep_pars=sweep_pars, sim_pars=None, levels=levels)
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
