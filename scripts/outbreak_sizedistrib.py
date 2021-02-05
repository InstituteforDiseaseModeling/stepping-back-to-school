'''
Outbreak analysis to sweep in-school transmissibility while also exploring several diagnostic screening scenarios.
'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings

    args = sct.config.process_inputs(sys.argv)

    sweep_pars = {
        'n_prev': 0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        #'screen_keys':  ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'Antigen every 2w', 'Antigen every 1w', 'PCR every 1w'],
        'screen_keys':  ['None', 'Antigen every 2w', 'Antigen every 1w'],
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    beta_s = [0.83333333, 1.61111111] # [0.83333333,1.22222222,1.61111111] # Just show one level
    npi_scens = {x:{'beta_s': 1.5*x} for x in beta_s}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    huevar = 'In-school transmission multiplier'
    xvar = 'Dx Screening'

    # Create and run
    mgr = sct.Manager(name='OutbreakScreeningSizeDistrib', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()


    #runner.regplots(xvar=xvar, huevar=huevar)

    #analyzer.outbreak_size_distribution(row='Dx Screening', col='In-school transmission multiplier', height=12, aspect=0.6)
    #g = analyzer.outbreak_multipanel(xvar, ext=None, jitter=0.15, values=None, legend=False, height=12, aspect=1.0) # height=10, aspect=0.7, 

    analyzer.outbreak_size_plot(xvar, rowvar='In-school transmission multiplier', ext=None, height=6, aspect=3, scatter=True, jitter=0.75)

    #analyzer.outbreak_size_plot(huevar, scatter=True, loess=False, landscape=False, ext='Dx', aspect=1.7)
