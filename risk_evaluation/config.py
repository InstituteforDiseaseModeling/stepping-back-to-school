'''
Set global configurations for the runs
'''

import sciris as sc

# Default settings are for debug runs
config = sc.objdict(
    inputs = 'inputs',
    results = 'results',
    n_reps = 3,
    n_seeds = 5,
)

sim_pars = sc.objdict(
    pop_size     = 50_000,
    verbose      = 0.1,
    pop_infected = 100,
    change_beta  = 1,
    symp_prob    = 0.08,
    asymp_factor = 0.8,
    start_day    = '2020-12-15', # First day of sim
    end_day      = '2021-02-26', #2021-04-30', # Last day of sim
    pop_scale    = 1,
    pop_type     = 'synthpops',
    rescale      = False, # True causes problems
)

sweep_pars = sc.objdict(
    schcfg_keys       = ['with_countermeasures'],
    screen_keys       = ['None'],
    school_start_date = '2021-02-01', # first day of school
    school_seed_date  = None,
    n_reps            = 3,
    n_seeds           = 5,
    prev              = [0.01],
)

run_pars = sc.objdict(
    n_cpus     = None, # Manually set the number of CPUs -- otherwise calculated automatically
    cpu_thresh = 0.95, # Don't use more than this amount of available CPUs, if number of CPUs is not set
    mem_thresh = 0.80, # Don't use more than this amount of available RAM, if number of CPUs is not set
    parallel   = True, # Only switch to False for debugging
    shrink     = True, # Whether to remove the people from the sim objects (makes for smaller files)
    verbose    = 0.1 # Print progress this fraction of simulated days
)

paths = sc.objdict(
    outputs = 'v2021-January', # Folder for outputs
    ei     = 'fit_EI.obj',
    ir     = 'fit_IR.obj',
    )


def process_inputs(argv):
    ''' Handle command-line input arguments '''
    if len(argv)>1:
        arg1 = argv[1]
        if arg1 == 'debug':
            return set_debug()
        elif arg1 == 'full':
            return set_full()
        else:
            sim_pars.pop_size = 1000*int(arg1)
    if len(argv)>2:
        sweep_pars.n_seeds = int(argv[2])
    return


def set_debug():
    ''' Reset the configuration for quick debugging runs '''
    sweep_pars.n_reps = 3
    sweep_pars.n_seeds = 5
    sim_pars.pop_size = 50_000
    return


def set_full():
    ''' Reset the configuration for the full run '''
    sweep_pars.n_reps = 5
    sweep_pars.n_seeds = 20
    sim_pars.pop_size = 223_000
    return