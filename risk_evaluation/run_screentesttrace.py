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


# Check that versions are correct
cv.check_save_version('2.0.0', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

n_reps = 2
n_prev = 10
pop_size = 223_000
folder = 'v2020-12-16'
stem = f'screentesttrace_{pop_size}_{n_reps}reps'

run_cfg = {
    'folder':       folder,
    'n_cpus':       None, # Manually set the number of CPUs -- otherwise calculated automatically
    'cpu_thresh':   0.75, # Don't use more than this amount of available CPUs, if number of CPUs is not set
    'mem_thresh':   0.75, # Don't use more than this amount of available RAM, if number of CPUs is not set
    'parallel':     True, # Only switch to False for debugging
    'shrink':       True, #
    'verbose':      0.1 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)
}

def build_configs():
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
    screen_scens = {
        'None':       { 'screen_prob': 0.0, 'test_prob': 0.0, 'trace_prob': 0.00, 'quar_prob': 0.0 },
        'Baseline':   { 'screen_prob': 0.5, 'test_prob': 0.5, 'trace_prob': 0.75, 'quar_prob': 0.95, 'screen2pcr': 2  },
        'Optimistic': { 'screen_prob': 0.9, 'test_prob': 1.0, 'trace_prob': 0.90, 'quar_prob': 1.00, 'screen2pcr': 1 },
    }
    b.add_level('ikey', screen_scens, b.screenpars_func)

    # Configure alternate sus
    rep_levels = {'Yes' if p else 'No':p for p in [True]}
    b.add_level('AltSus', rep_levels, ut.alternate_symptomaticity)

    # Add school intervention
    for config in b.configs:
        config.interventions += [cvsch.schools_manager(config.school_config)]

    # Add reps
    rep_levels = {f'Run {p}':{'rand_seed':p} for p in range(n_reps)}
    b.add_level('eidx', rep_levels, b.simpars_func)

    return b.get()


def plot(sims, to_plot):
    imgdir = os.path.join(folder, 'img_'+stem)
    p = pt.Plotting(sims, imgdir)

    if to_plot['Prevalence']:
        p.timeseries('n_exposed', 'Prevalence', normalize=True)

    if to_plot['CumInfections']:
        p.timeseries('cum_infections', 'Cumulative Infections', normalize=True)

    if to_plot['Quarantined']:
        p.timeseries('n_quarantined', 'Number Quarantined', normalize=True)

    if to_plot['IntroductionsRegression']:
        p.introductions_reg(hue_key='ikey')

    if to_plot['OutbreakSizeRegression']:
        p.outbreak_reg(hue_key='ikey')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    # Which figures to plot
    to_plot = {
        'Prevalence':               True, # Plot prevalence longitudinally
        'CumInfections':            True, # Plot cumulative infections longitudinally
        'Quarantined':              True,
        'IntroductionsRegression':  True,
        'OutbreakSizeRegression':   True,
        #'Debug trees':              False, # Show each introduced tree for debugging
    }

    cachefn = os.path.join(folder, 'sims', f'{stem}.sims') # Might need to change the extension here, depending in combine.py was used
    if args.force or not os.path.isfile(cachefn):
        sim_configs = build_configs()
        sims = ut.run_configs(sim_configs, stem, run_cfg, cachefn)
    else:
        print(f'Loading {cachefn}')
        sims = cv.load(cachefn) # Use for *.sims

    plot(sims, to_plot)
