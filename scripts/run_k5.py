'''
Introduction analysis sweeping over several prevalence levels.  Similar to run_baseline.py, but this is just for K-5.

Example usage, forcing new results and using a 4 different seeds:

    python run_k5.py --force --n_reps=4

'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)
    sweep_pars = dict(schcfg_keys = ['k5'])
    xvar = 'Prevalence Target'

    # Create and run
    mgr = sct.Manager(sweep_pars=sweep_pars, sim_pars=None, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    mgr.regplots(xvar=xvar, huevar='Dx Screening')
    analyzer.cum_incidence(colvar=xvar)
    analyzer.introductions_rate_by_stype(xvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    analyzer.source_dow(figsize=(6.5,5))
    mgr.tsplots()

