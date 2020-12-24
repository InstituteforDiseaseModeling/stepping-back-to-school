import os
import psutil
import sciris as sc
import covasim as cv
import multiprocessing as mp
import create_sim as cs


def alternate_symptomaticity(config, key, value):
    print(f'Building alternate symptomaticity {key}={value}')
    if not value: # Only build if value is True
        return config
    if 'prognoses' in config.sim_pars:
        prog = config.sim_pars['prognoses']
    else:
        pars = cv.make_pars(set_prognoses=True, prog_by_age=True, **config.sim_pars)
        prog = pars['prognoses']

    ages = prog['age_cutoffs']
    symp_probs = prog['symp_probs']

    if False:
        # Source: table 1 from https://arxiv.org/pdf/2006.08471.pdf
        symp_probs[:] = 0.6456
        symp_probs[ages<80] = 0.3546
        symp_probs[ages<60] = 0.3054
        symp_probs[ages<40] = 0.2241
        symp_probs[ages<20] = 0.1809
        prog['symp_probs'] = symp_probs
    else:
        print('WARNING: DAN MADE THIS UP!!!')
        #prog['symp_probs'] = 0.10 + (0.9-0.15) * (ages - min(ages)) / (max(ages)-min(ages))
        symp_probs[:] = 0.8
        symp_probs[ages<20] = 0.20
        symp_probs[ages<10] = 0.15
        prog['symp_probs'] = symp_probs

    config.sim_pars['prognoses'] = sc.dcp(prog) # Ugh

    return config


def children_equally_sus(config, key, value):
    print(f'Building children equally susceptibility {key}={value}')
    if not value: # Only build if value is True
        return config
    prog = config.pars['prognoses']
    ages = prog['age_cutoffs']
    sus_ORs = prog['sus_ORs']
    sus_ORs[ages<=20] = 1
    prog['sus_ORs'] = sus_ORs
    return config


def p2f(x):
    return float(x.strip('%'))/100

#%% Running
def create_run_sim(sconf, n_sims, run_config):
    ''' Create and run the actual simulations '''
    print(f'Creating and running sim {sconf.count} of {n_sims}...')

    T = sc.tic()
    sim = cs.create_sim(sconf.sim_pars, folder=run_config['folder'])

    for intv in sconf.interventions:
        sim['interventions'].append(intv)

    sim.tags = sc.dcp(sconf.tags)

    sim.run()

    if run_config['shrink']:
        sim.shrink() # Do not keep people after run
    sc.toc(T)
    return sim


def run_configs(sim_configs, stem, run_cfg, filename=None):
    n_cpus = run_cfg['n_cpus']
    pop_size = max([c.sim_pars['pop_size'] for c in sim_configs])

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
        sims = sc.parallelize(create_run_sim, iterarg=sim_configs, kwargs=dict(n_sims=len(sim_configs), run_config=run_cfg), ncpus=n_cpus)
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
