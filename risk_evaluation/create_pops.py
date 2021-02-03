'''
Pre-generate the synthpops population including school types. Takes ~15s per seed
if running with 50,000 people; or roughly 5 times longer for 223,000 people.

Warning: this script is quite memory intensive. It should pick the right degree
of parallelization so you do not run out of RAM, but be warned.

To run with a different location or size, you can specify these as follows, e.g.

    python create_pops.py --pop_size 35 --location Spokane_County
'''

import sys
import psutil
import numpy as np
import sciris as sc
import covasim_schools as cvsch
import config as cfg


def create_pops(config=None, parallelize=True):
    ''' Create the population files '''

    if config is not None:
        cfg = config # Replace with config, if supplied

    pop_size = cfg.sim_pars.pop_size
    n_seeds = cfg.sweep_pars.n_seeds
    location = cfg.pop_pars.location
    seeds = np.arange(n_seeds) + cfg.run_pars.base_seed

    print(f'Creating {n_seeds} populations of size {pop_size} for {location}...')
    kwargs = dict(pop_size=pop_size, location=location, folder=cfg.paths.inputs)

    if parallelize:
        ram = psutil.virtual_memory().available/1e9
        max_cpus = psutil.cpu_count()
        max_parallel = min(max_cpus, n_seeds)
        required = 1.5*pop_size/223e3 # 1.5 GB per 223e3 people
        max_required = max_parallel*required
        if max_required < ram:
            print(f'You have {ram:0.1f} GB of RAM, and this is estimated to require {max_required:0.1f} GB: you should be fine')
            ncpus = max_parallel
        else:
            ncpus = int(max_parallel*ram/max_required)
            print(f'You have {ram:0.1f} GB of RAM, but this is estimated to require {max_required:0.1f} GB -- changing from {max_cpus} CPUs to {ncpus}')
        sc.parallelize(cvsch.make_population, kwargs=kwargs, iterkwargs={'rand_seed':seeds}, ncpus=ncpus) # Run them in parallel
    else:
        for seed in seeds:
            cvsch.make_population(**kwargs, rand_seed=seed)


# This must be in a main block for parallelization to work on Windows
if __name__ == '__main__':

    cfg.process_inputs(sys.argv)
    parallelize = True
    create_pops(parallelize=parallelize)

