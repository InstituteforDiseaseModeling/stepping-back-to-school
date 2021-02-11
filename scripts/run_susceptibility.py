
'''
Sensitivity for susceptibility
'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)
    pop_size = sct.config.sim_pars.pop_size
    sim_pars = dict(pop_size=pop_size)
    sweep_pars = dict(alt_sus=[False, True])
    xvar = 'Prevalence Target'
    huevar = 'AltSus'

    # Create and run
    mgr = sct.Manager(name='Susceptibility', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    mgr.regplots(xvar=xvar, huevar=huevar, height=6, aspect=2.4)
    analyzer.introductions_rate(xvar, huevar=huevar, height=6, aspect=1.4, ext='ppt')

    analyzer.source_pie()
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_dow(figsize=(10,5)) # 6.5 x 5
    analyzer.source_dow(figsize=(6*1.4, 6), ext='ppt') # 6.5 x 5
    mgr.tsplots()
