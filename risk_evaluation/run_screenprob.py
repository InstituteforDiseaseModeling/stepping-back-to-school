'''
This is the main script used for commissioning the different testing scenarios
(defined in testing_scenarios.py) for the paper results. Run this script first
(preferably on an HPC!), then run the different plotting scripts. Each sim takes
about 1 minute to run. With full settings, there are about 1300 scripts to run;
the test run uses 8.
'''

import os
import numpy as np
import covasim as cv
import sciris as sc
import synthpops as sp
import covasim_schools as cvsch
import testing_scenarios as t_s # From the local folder
import covasim_controller as cvc
import utils as ut


# Check that versions are correct
cv.check_save_version('2.0.0', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

n_reps = 1
pop_size = 223_000
folder = 'v2020-12-16'
stem = f'import_{pop_size}_{n_reps}reps'
skip_screening = False # Set True for the no-screening variant

run_cfg = {
    'alternate_symptomaticity':  True,

    'n_cpus':       None, # Manually set the number of CPUs -- otherwise calculated automatically
    'cpu_thresh':   0.75, # Don't use more than this amount of available CPUs, if number of CPUs is not set
    'mem_thresh':   0.75, # Don't use more than this amount of available RAM, if number of CPUs is not set
    'parallel':     True, # Only switch to False for debugging
    'shrink':       True, #
    'verbose':      0.1 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)
}

def generate_configs():
    # Build simulation configuration
    sc.heading('Creating sim configurations...')
    sim_configs = []
    count = -1

    start_day = '2021-02-01' # first day of school
    scenarios = t_s.generate_scenarios(start_day) # Can potentially select a subset of scenarios
    testing = t_s.generate_testing(start_day) # Potentially select a subset of testing

    # For a test run only use a subset of scenarios
    scenarios = {k:v for k,v in scenarios.items() if k in ['with_countermeasures']}#, 'with_countermeasures', 'k5', 'all_hybrid', 'all_remote']}
    testing = {k:v for k,v in testing.items() if k in ['None']}

    pars = {
        'pop_infected': 100,
        'change_beta': 1,
        #'symp_prob': 0.1 # About 57% of symptomatic infections will be diagnosed 1-(1-0.1)**8, and assuming none of the asymptomatics will be diagnosed, that's about 40% ovrall.  Adding some asymptomatics should result in more than 40% diagnosed, consistent with latest RAINIER modeling. Was 0.09 previously.
        'symp_prob':    0.08,
        'asymp_factor': 0.8,
        'start_day':    '2020-12-01', # First day of sim
        'end_day':      '2021-04-30', # Last day of sim
    }

    import_scens = {
        'none': { 'screen_prob': 0 },
        'half': { 'screen_prob': 0.5 },
        'full': { 'screen_prob': 1 },
    }

    for prev in np.linspace(0.002, 0.02, 4):
        for skey, base_scen in scenarios.items():
            for tkey, test in testing.items():
                for ikey, import_scen in import_scens.items():
                    for eidx in range(n_reps):
                        count += 1
                        p = sc.dcp(pars)
                        p['rand_seed'] = eidx # np.random.randint(1e6)

                        sconf = sc.objdict(count=count, sim_pars=p, pop_size=pop_size, folder=folder)

                        # Add controller ######################################
                        pole_loc = 0.35
                        targets = {
                            'infected': prev * pop_size, # prevalence target
                        }

                        # These come from fit_transmats
                        ei = sc.loadobj('EI.obj')
                        ir = sc.loadobj('IR.obj')

                        seir = cvc.SEIR(pop_size, ei.Mopt, ir.Mopt, ERR=1, beta=0.365, Ipow=0.925)
                        ctr = cvc.controller_intervention(seir, targets, pole_loc=pole_loc, start_day=1)
                        ######################3################################

                        # Modify base_scen with testing intervention
                        this_scen = sc.dcp(base_scen)
                        for stype, spec in this_scen.items():
                            if spec is not None:
                                spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools
                                spec.update(import_scen)

                        sm = cvsch.schools_manager(this_scen)

                        sconf.update(dict(
                            label = f'{prev} + {skey} + {tkey}',
                            prev = prev,
                            skey = skey,
                            tkey = tkey,
                            ikey = ikey,
                            eidx = eidx,
                            test = test,
                            this_scen = this_scen,
                            sm = sm,
                            ctr = ctr,
                        ))

                        sim_configs.append(sconf)

    print(f'Done: {len(sim_configs)} configurations created')
    return sim_configs


# Windows requires a main block for running in parallel
if __name__ == '__main__':
    filename = os.path.join(folder, 'sims', f'{stem}.sims')
    sim_configs = generate_configs()
    sims = ut.run_configs(sim_configs, stem, run_cfg, filename)
