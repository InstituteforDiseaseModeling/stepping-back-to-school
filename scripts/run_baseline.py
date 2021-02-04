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

    # Create and run
    mgr = sct.Manager(sweep_pars=None, sim_pars=sim_pars, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    #mgr.regplots(xvar='Prevalence Target', huevar='Dx Screening') # CK: doesn't work
    #analyzer.cum_incidence(colvar='Prevalence Target') # CK: doesn't work
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    analyzer.source_dow(figsize=(8,5)) # 6.5 x 5
    mgr.tsplots()

