'''
Pre-generate the synthpops population including school types. Takes ~10s per seed
if running with 20,000 people; or roughly 10 times longer for 223,000 people.

After running this script, you will need to manually move the files generated in
the inputs folder into the main results folder (v20201009).

Warning: this script is quite memory intensive. If you do not have much RAM, you may
wish to turn off parallelization.
'''

import sys
import psutil
import sciris as sc
import covasim_schools as cvsch
import config as cfg


# This must be in a main block for parallelization to work on Windows
if __name__ == '__main__':

    cfg.process_inputs(sys.argv)

    pop_size = cfg.sim_pars.pop_size
    n_seeds = cfg.sweep_pars.n_seeds
    seeds = range(n_seeds)
    parallelize = True

    print(f'Creating {n_seeds} populations of size {pop_size}...')

    if parallelize:
        ram = psutil.virtual_memory().available/1e9
        max_cpus = psutil.cpu_count()
        max_parallel = min(max_cpus, n_seeds)
        required = 2.0*pop_size/225e3 # 1 GB per 225e3 people
        max_required = max_parallel*required
        if max_required < ram:
            print(f'You have {ram:0.1f} GB of RAM, and this is estimated to require {max_required:0.1f} GB: you should be fine')
            ncpus = max_parallel
        else:
            ncpus = int(max_parallel*ram/max_required)
            print(f'You have {ram:0.1f} GB of RAM, but this is estimated to require {max_required:0.1f} GB -- changing from {max_cpus} CPUs to {ncpus}')
        sc.parallelize(cvsch.make_population, kwargs={'pop_size':pop_size}, iterkwargs={'rand_seed':seeds}, ncpus=ncpus) # Run them in parallel
    else:
        for seed in seeds:
            cvsch.make_population(pop_size=pop_size, rand_seed=seed)
