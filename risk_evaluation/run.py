'''
This is the main script used for commissioning the different testing scenarios
(defined in testing_scenarios.py) for the paper results. Run this script first
(preferably on an HPC!), then run the different plotting scripts. Each sim takes
about 1 minute to run. With full settings, there are about 1300 scripts to run;
the test run uses 8.
'''

import os
import argparse
import numpy as np
import covasim as cv
import sciris as sc
import synthpops as sp
import covasim_schools as cvsch
import builder as bld
import plotting as pt
import utils as ut

class Run:
    def __init__(self, name):
        # Check that versions are correct
        cv.check_save_version('2.0.0', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

        self.name = name

        # TODO: make default config`, update with user config
        n_reps = 1
        n_prev = 4
        pop_size = 100_000 #223_000
        folder = 'v2020-12-16'

        self.stem = f'{name}_{pop_size}_{n_reps}reps'
        self.cachefn = os.path.join(folder, 'sims', f'{self.stem}.sims') # Might need to change the extension here, depending in combine.py was used
        self.imgdir = os.path.join(folder, 'img_'+self.stem)

        # TODO: make default config`, update with user config
        run_cfg = {
            'folder':       folder,
            'n_cpus':       None, # Manually set the number of CPUs -- otherwise calculated automatically
            'cpu_thresh':   0.75, # Don't use more than this amount of available CPUs, if number of CPUs is not set
            'mem_thresh':   0.75, # Don't use more than this amount of available RAM, if number of CPUs is not set
            'parallel':     True, # Only switch to False for debugging
            'shrink':       True, #
            'verbose':      0.1 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)
        }

    def build_configs(self):
        # Build simulation configuration
        sc.heading('Creating sim configurations...')
        sim_pars = {
            'verbose': 0.1,
            'pop_infected': 100,
            'pop_size':     pop_size,
            'change_beta':  1,
            'symp_prob':    0.08,
            'asymp_factor': 0.8,
            'start_day':    '2020-12-15', # First day of sim
            'end_day':      '2021-03-31', #2021-04-30', # Last day of sim
        }

        school_start_date = '2021-02-01' # first day of school
        b = bld.Builder(sim_pars, ['with_countermeasures'], ['None'], school_start_date)

        # Add prevalence levels
        prev_levels = {f'{100*p:.1f}%':p for p in np.linspace(0.002, 0.02, n_prev)}
        b.add_level('prev', prev_levels, b.prevctr_func)

        # Sweep over symptom screening
        symp_screens = {
            'No symptom screening': { 'screen_prob': 0 },
            '50% daily screening':  { 'screen_prob': 0.5 },
            '100% daily screening': { 'screen_prob': 1 },
        }
        b.add_level('ikey', symp_screens, b.screenpars_func)

        # Configure alternate sus
        rep_levels = {'Yes' if p else 'No':p for p in [True]}
        b.add_level('AltSus', rep_levels, ut.alternate_symptomaticity)

        # Add school intervention
        for config in b.configs:
            config.interventions.append(cvsch.schools_manager(config.school_config))

        # Add reps
        rep_levels = {f'Run {p}':{'rand_seed':p} for p in range(n_reps)}
        b.add_level('eidx', rep_levels, b.simpars_func)

        return b.get()


    def plot(self, sims, ts_plots=None):
        p = pt.Plotting(sims, self.imgdir)

        p.introductions_reg(hue_key='ikey')
        p.outbreak_reg(hue_key='ikey')

        if ts_plots is None:
            return

        if ts_plots == True:
            ts_plots = [
                dict(label='Prevalence',      channel='n_exposed',      normalize=True),
                dict(label='CumInfections',   channel='cum_infections', normalize=True),
                dict(label='Quarantined',     channel='n_quarantined',  normalize=True),
                dict(label='Newly Diagnosed', channel='new_diagnoses',  normalize=True),
            ]
        p.several_timeseries(ts_plots)


    def run(self, force):
        if args.force or not os.path.isfile(self.cachefn):
            sim_configs = build_configs()
            sims = ut.run_configs(sim_configs, self.stem, run_cfg, self.cachefn) # why is stem needed here?
        else:
            print(f'Loading {self.cachefn}')
            sims = cv.load(self.cachefn) # Use for *.sims

        plot(sims, ts_plots)

'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
'''
