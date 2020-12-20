import os
import psutil
import sciris as sc
import covasim as cv
import multiprocessing as mp
import create_sim as cs

#%% Running
def create_run_sim(sconf, n_sims, config):
    ''' Create and run the actual simulations '''
    print(f'Creating and running sim {sconf.count} of {n_sims}...')

    verbose = config['verbose'] if 'verbose' in config else 0
    alternate_symptomaticity = config['alternate_symptomaticity'] if 'alternate_symptomaticity' in config else False
    children_equally_sus = config['children_equally_sus'] if 'children_equally_sus' in config else False

    T = sc.tic()
    sim = cs.create_sim(sconf.sim_pars, pop_size=sconf.pop_size, folder=sconf.folder, alternate_symptomaticity=alternate_symptomaticity, children_equally_sus=children_equally_sus)

    sim['interventions'].append(sconf.sm)
    sim['interventions'].append(sconf.ctr)
    sconf.pop('sm')
    sconf.pop('ctr')

    for k,v in sconf.items():
        setattr(sim,k,v)
        '''
        sim.count = sconf.count
        sim.label = sconf.label
        sim.key1 = sconf.skey
        sim.key2 = sconf.tkey
        sim.key3 = sconf.prev
        sim.eidx = sconf.eidx
        sim.tscen = sconf.test
        sim.scen = sconf.this_scen
        sim.dynamic_par = sconf.sim_pars
        '''
    print(sim.prev)

    sim.run(verbose=verbose)
    if config['shrink']:
        if verbose > 0:
            print('Shrinking')
        sim.shrink() # Do not keep people after run
    sc.toc(T)
    return sim


def run_configs(sim_configs, stem, run_cfg, filename=None):
    n_cpus = run_cfg['n_cpus']
    pop_size = max([c.pop_size for c in sim_configs])

    sc.heading('Choosing correct number of CPUs...')
    if n_cpus is None:
        cpu_limit = int(mp.cpu_count()*run_cfg['cpu_thresh']) # Don't use more than 75% of available CPUs
        ram_available = psutil.virtual_memory().available/1e9
        ram_required = 1.5*pop_size/2.25e5 # Roughly 1.5 GB per 225e3 people
        ram_limit = int(ram_available/ram_required*run_cfg['mem_thresh'])
        n_cpus = min(cpu_limit, ram_limit)
        print(f'{n_cpus} CPUs are being used due to a CPU limit of {cpu_limit} and estimated RAM limit of {ram_limit}')
    else:
        print(f'Using user-specified {n_cpus} CPUs')

    sc.heading('Running sims...')
    TT = sc.tic()
    if run_cfg['parallel']:
        sims = sc.parallelize(create_run_sim, iterarg=sim_configs, kwargs=dict(n_sims=len(sim_configs), config=run_cfg), ncpus=n_cpus)
    else:
        sims = []
        for sconf in sim_configs:
            sim = create_run_sim(sconf, n_sims=len(sim_configs))
            sims.append(sim)

    if filename is not None:
        sc.heading('Saving all sims...')
        cv.save(filename, sims)
        print(f'Done, saved {filename}')

    sc.toc(TT)

    return sims
