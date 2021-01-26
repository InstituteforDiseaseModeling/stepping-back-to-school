'''
Set global configurations for the runs
'''

import sciris as sc

# Default settings are for debug runs
config = sc.objdict(
    inputs = 'inputs',
    results = 'results',
    n_seeds = 5,
    pop_size = 50_000,
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
            config.pop_size = 1000*int(arg1)
    if len(argv)>2:
        config.n_seeds = int(argv[2])
    return


def set_debug():
    ''' Reset the configuration for quick debugging runs '''
    config.n_seeds = 5
    config.pop_size = 50_000
    return


def set_full():
    ''' Reset the configuration for the full run '''
    config.n_seeds = 20
    config.pop_size = 223_000
    return