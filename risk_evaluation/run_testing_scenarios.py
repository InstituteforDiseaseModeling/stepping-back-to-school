'''
This is the main script used for commissioning the different testing scenarios
(defined in testing_scenarios.py) for the paper results. Run this script first
(preferably on an HPC!), then run the different plotting scripts. Each sim takes
about 1 minute to run. With full settings, there are about 1300 scripts to run;
the test run uses 8.
'''

import os
import psutil
import numpy as np
import multiprocessing as mp
import covasim as cv
import create_sim as cs
import sciris as sc
import synthpops as sp
import covasim_schools as cvsch
import testing_scenarios as t_s # From the local folder
import covasim_controller as cvc


#%% Settings
sc.heading('Setting parameters...')

# Check that versions are correct
cv.check_save_version('1.7.6', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

test_run = False # Whether to do a small test run, or the full results: changes the number of runs and scenarios -- 1 for testing, or 30 for full results
parallel = True # Only switch to False for debugging

n_reps = 10
pop_size = 100_000 #2.25e5
skip_screening = False # Set True for the no-screening variant
save_each_sim = False # Save each sim separately instead of all together
n_cpus = None # Manually set the number of CPUs -- otherwise calculated automatically
cpu_thresh = 0.75 # Don't use more than this amount of available CPUs, if number of CPUs is not set
mem_thresh = 0.75 # Don't use more than this amount of available RAM, if number of CPUs is not set
verbose = 0.1 if test_run else 0.0 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)

folder = 'v2020-12-02'
#stem = f'k5_PCROptimistic_Sweep{n_reps}reps'
stem = f'optimistic_countermeasures_Antigen_{n_reps}reps'


# For a test run only use a subset of scenarios
start_day = '2020-11-30' # first day of school
scenarios = t_s.generate_scenarios(start_day) # Can potentially select a subset of scenarios
testing = t_s.generate_testing(start_day) # Potentially select a subset of testing
if test_run:
    scenarios = {k:v for k,v in scenarios.items() if k in ['k5']}#, 'with_countermeasures', 'k5', 'all_hybrid', 'all_remote']}
    testing = {k:v for k,v in testing.items() if k in ['None']}#, 'Antigen every 1w, PCR f/u']}
else:
    scenarios = {k:v for k,v in scenarios.items() if k in ['with_optimistic_countermeasures']}#, 'with_countermeasures', 'k5', 'all_hybrid', 'all_remote']}
    #testing = {k:v for k,v in testing.items() if k in ['None', 'Antigen every 4w, PCR f/u', 'Antigen every 2w, PCR f/u', 'Antigen every 1w, PCR f/u']} # ['None', 'Antigen every 4w, PCR f/u', 'PCR every 4w', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u', 'PCR every 1w']
    #testing = {k:v for k,v in testing.items() if k in ['None', 'PCR every 4w', 'PCR every 2w', 'PCR every 1w']} # ['None', 'Antigen every 4w, PCR f/u', 'PCR every 4w', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u', 'PCR every 1w']
    testing = {k:v for k,v in testing.items() if k in ['None', 'Antigen every 4w, PCR f/u', 'Antigen every 2w, PCR f/u', 'Antigen every 1w, PCR f/u']} # ['None', 'Antigen every 4w, PCR f/u', 'PCR every 4w', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u', 'PCR every 1w']


sc.heading('Choosing correct number of CPUs...')
if n_cpus is None:
    cpu_limit = int(mp.cpu_count()*cpu_thresh) # Don't use more than 75% of available CPUs
    ram_available = psutil.virtual_memory().available/1e9
    ram_required = 1.5*pop_size/2.25e5 # Roughly 1.5 GB per 225e3 people
    ram_limit = int(ram_available/ram_required*mem_thresh)
    n_cpus = min(cpu_limit, ram_limit)
    print(f'{n_cpus} CPUs are being used due to a CPU limit of {cpu_limit} and estimated RAM limit of {ram_limit}')
else:
    print(f'Using user-specified {n_cpus} CPUs')

pars = { # Not really needed...
    "pop_infected": 100,
    "change_beta": 1,
    "symp_prob": 0.09
}

#%% Configuration
sc.heading('Creating sim configurations...')
sim_configs = []
count = -1

for prev in [0.002, 0.005, 0.01]: #np.linspace(0.001, 0.015, 30): # [0.002, 0.005, 0.01]#0.001 * np.sqrt(2)**np.arange(9): #np.linspace(0.002, 0.02, 5):
    for skey, base_scen in scenarios.items():
        for tidx, (tkey, test) in enumerate(testing.items()):
            for eidx, rand_seed in enumerate(range(n_reps)): # I know...
                count += 1
                p = sc.dcp(pars)
                p['rand_seed'] = rand_seed

                sconf = sc.objdict(count=count, pars=p, pop_size=pop_size, folder=folder)

                # Add controller ######################################
                pole_loc = 0.35
                targets = {
                    #'cases':        200, # per 100k over 2-weeks, from DOH website
                    #'re':           1.0,
                    #'prevalence':   0.003, # KC image from Niket in #nowcasting-inside-voice dated 11/02 using data through 10/23
                    #'yield':        0.029, # 2.4% positive
                    #'tests':        225,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
                    'infected':      prev * pop_size, # prevalence target
                }

                # These come from fit_transmats
                ei = sc.loadobj('EI.obj')
                ir = sc.loadobj('IR.obj')

                seir = cvc.SEIR(pop_size, ei.Mopt, ir.Mopt, ERR=1, beta=0.365, Ipow=0.925)
                ctr = cvc.controller_intervention(seir, targets, pole_loc=pole_loc)
                ######################3################################

                # Modify base_scen with testing intervention
                this_scen = sc.dcp(base_scen)
                for stype, spec in this_scen.items():
                    if spec is not None:
                        spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools
                        if skip_screening:
                            spec['screen_prob'] = 0

                sm = cvsch.schools_manager(this_scen)

                sconf.update(dict(
                    label = f'{prev} + {skey} + {tkey}',
                    prev = prev,
                    skey = skey,
                    tkey = tkey,
                    eidx = eidx,
                    test = test,
                    this_scen = this_scen,
                    sm = sm,
                    ctr = ctr,
                ))

                sim_configs.append(sconf)
print(f'Done: {len(sim_configs)} configurations created')



#%% Running
def create_run_sim(sconf, n_sims):
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
    if save_each_sim:
        filename = os.path.join(folder, 'sims', f'sim{sconf.count}_{skey}.sim')
        sim.save(filename)
        print(f'Saved {filename}')
    sc.toc(T)
    return sim


# Windows requires a main block for running in parallel
if __name__ == '__main__':

    sc.heading('Running sims...')

    TT = sc.tic()

    if parallel:
        sims = sc.parallelize(create_run_sim, iterarg=sim_configs, kwargs=dict(n_sims=len(sim_configs)), ncpus=n_cpus)
    else:
        sims = []
        for sconf in sim_configs:
            sim = create_run_sim(sconf, n_sims=len(sim_configs))
            sims.append(sim)

    if not save_each_sim:
       sc.heading('Saving all sims...')
       filename = os.path.join(folder, 'sims', f'{stem}.sims')
       cv.save(filename, sims)
       print(f'Saved {filename}')

    print('Done.')
    sc.toc(TT)
