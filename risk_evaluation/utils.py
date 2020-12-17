import os
import psutil
import sciris as sc
import create_sim as cs
import multiprocessing as mp

#%% Running
def create_run_sim(sconf, n_sims, verbose):
    ''' Create and run the actual simulations '''
    print(f'Creating and running sim {sconf.count} of {n_sims}...')
    T = sc.tic()
    sim = cs.create_sim(sconf.pars, pop_size=sconf.pop_size, folder=sconf.folder)
    sim.count = sconf.count
    sim.label = sconf.label
    sim.key1 = sconf.skey
    sim.key2 = sconf.tkey
    sim.key3 = sconf.prev
    sim.eidx = sconf.eidx
    sim.tscen = sconf.test
    sim.scen = sconf.this_scen # After modification with testing above
    sim.dynamic_par = sconf.pars
    sim['interventions'].append(sconf.sm)
    sim['interventions'].append(sconf.ctr)
    sim.run(verbose=verbose)
    sim.shrink() # Do not keep people after run
    sc.toc(T)
    return sim


def run_configs(sim_configs, stem, run_cfg):
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
        sims = sc.parallelize(create_run_sim, iterarg=sim_configs, kwargs=dict(n_sims=len(sim_configs), verbose=run_cfg['verbose']), ncpus=n_cpus)
    else:
        sims = []
        for sconf in sim_configs:
            sim = create_run_sim(sconf, n_sims=len(sim_configs))
            sims.append(sim)

    sc.heading('Saving all sims...')
    filename = os.path.join(folder, 'sims', f'{stem}.sims')
    cv.save(filename, sims)
    print(f'Done, saved {filename}')

    sc.toc(TT)
