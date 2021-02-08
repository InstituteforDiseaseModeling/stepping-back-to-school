'''
Introduction analysis sweeping over several prevalence levels.

Example usage, forcing new results and using a 4 different seeds:

    python run_baseline.py --force --n_reps=4

'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)
    pop_size = sct.config.sim_pars.pop_size
    sim_pars = dict(pop_size=pop_size)
    xvar = 'Prevalence Target'

    # Create and run
    mgr = sct.Manager(name='Baseline', sweep_pars=None, sim_pars=sim_pars, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    mgr.regplots(xvar=xvar, huevar='Dx Screening', height=6, aspect=2.4)
    analyzer.source_pie()
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_dow(figsize=(10,5)) # 6.5 x 5
    mgr.tsplots()

