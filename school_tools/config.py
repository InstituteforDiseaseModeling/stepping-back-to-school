'''
Set global configurations for the runs
'''

import argparse
import numpy as np
import sciris as sc

# Default settings are for debug runs

def get_defaults():

    sim_pars = sc.objdict(
        pop_size     = 50_000,
        verbose      = 0.1,
        pop_infected = 100,
        change_beta  = 1,
        symp_prob    = 0.08,
        asymp_factor = 0.8,
        start_day    = '2020-12-15', # First day of sim
        end_day      = '2021-02-26', #2021-04-30', # Last day of sim -- usually overridden
        pop_scale    = 1,
        pop_type     = 'synthpops',
        rescale      = False, # True causes problems
    )

    pop_pars = sc.objdict(
        location = 'seattle_metro',
    )

    sweep_pars = sc.objdict(
        schcfg_keys       = ['with_countermeasures'],
        screen_keys       = ['None'],
        school_start_date = '2021-02-01', # first day of school
        school_seed_date  = None,
        n_reps            = 3,
        n_pops            = None,
        n_prev            = 3,
        prev              = None, # Computed in builder.py
        alt_symp          = [False],
        alt_sus           = [False],
        vaccine           = {'None':None},
        cohort_rewiring   = {'None':None},
    )

    run_pars = sc.objdict(
        n_cpus     = None, # Manually set the number of CPUs -- otherwise calculated automatically
        cpu_thresh = 0.95, # Don't use more than this amount of available CPUs, if number of CPUs is not set
        mem_thresh = 0.80, # Don't use more than this amount of available RAM, if number of CPUs is not set
        parallel   = True, # Only switch to False for debugging
        shrink     = True, # Whether to remove the people from the sim objects (makes for smaller files)
        verbose    = 0.1,  # Print progress this fraction of simulated days
        base_seed  = 0,    # Add this offset to all other random seeds
        outbreak   = False,# Whether to run a script as an outbreak instead of a simple run
    )

    paths = sc.objdict(
        inputs  = 'inputs', # Folder for population files
        outputs = 'results', # Folder for figures, sims, etc.
        ei      = sc.thisdir(None, 'fit_EI.obj'),
        ir      = sc.thisdir(None, 'fit_IR.obj'),
    )
    return sim_pars, pop_pars, sweep_pars, run_pars, paths


# Populate the global namespace
sim_pars, pop_pars, sweep_pars, run_pars, paths = get_defaults()
np.random.seed(run_pars.base_seed) # Reset the global seed on import


def process_inputs(argv, **kwargs): # pragma: no cover
    ''' Handle command-line input arguments -- used for most of the scripts. '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Regenerate files even if they exist')
    parser.add_argument('--full', action='store_true', help='Run with full population sizes and seeds (warning, takes hours!)')
    parser.add_argument('--debug', action='store_true', help='Run with moderate population sizes and seeds (default; takes a few minutes)')
    parser.add_argument('--micro', action='store_true', help='Run with minimal population sizes and seeds (warning, graphs will look strange)')
    parser.add_argument('--outbreak', action='store_true', help='If applicable, run as an outbreak scenario rather than a run scenario')
    parser.add_argument('--pop_size', type=int, default=0, help='Set the population size; if <1000, will automatically multiply, e.g. 25 and 25000 are the same')
    parser.add_argument('--n_reps', type=int, default=0, help='Set the number of repetitions (i.e. random seeds) for each run')
    parser.add_argument('--n_pops', type=int, default=0, help='Set the number of different populations generated/used (by default, n_reps)')
    parser.add_argument('--n_prev', type=int, default=0, help='Set the number of different prevalence levels used')
    parser.add_argument('--location', type=str, default='', help='Set the location (by default, Seattle)')
    args = parser.parse_args(argv[1:])

    # Handle any kwargs to override command-line options
    for k,v in kwargs.items():
        if v is not None:
            setattr(args, k, v)

    if args.full:
        set_full()
    elif args.debug:
        set_debug()
    elif args.micro:
        set_micro()

    if args.pop_size:
        if args.pop_size < 1000: # Allow e.g. 223 or 223000
            print('Automatically scaling population by 1000')
            args.pop_size *= 1000
        sim_pars.pop_size = args.pop_size
    if args.n_reps:
        sweep_pars.n_reps = args.n_reps
    if args.n_pops:
        sweep_pars.n_pops = args.n_pops
    if args.n_prev:
        sweep_pars.n_prev = args.n_prev
    if args.location:
        pop_pars.location = args.location

    return args


def print_pars(label):
    ''' Helper function to print the name '''
    print(f'Resetting parameters for a {label} run: n_reps={sweep_pars.n_reps}, n_pops={sweep_pars.n_pops}, pop_size={sim_pars.pop_size}')
    return


def set_default():
    ''' Reset to default settings -- currently debug '''
    return set_debug()


def set_full():
    ''' Reset the configuration for the full run '''
    sweep_pars.n_reps  = 20
    sweep_pars.n_pops  = None
    sweep_pars.n_prev  = 10
    sim_pars.pop_size  = 223_000
    print_pars('full')
    return


def set_debug():
    ''' Reset the configuration for quick debugging runs '''
    sweep_pars.n_reps  = 3
    sweep_pars.n_pops  = None
    sweep_pars.n_prev  = 3
    sim_pars.pop_size  = 50_000
    print_pars('debugging')
    return


def set_micro():
    ''' Reset the configuration to the smallest possible run '''
    sweep_pars.n_reps  = 1
    sweep_pars.n_pops  = None
    sweep_pars.n_prev  = 2
    sim_pars.pop_size = 10_000
    print_pars('micro')
    return
