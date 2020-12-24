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
import pandas as pd


# Check that versions are correct
cv.check_save_version('2.0.0', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

n_reps = 2
n_prev = 10
pop_size = 223_000
folder = 'v2020-12-16'
stem = f'brokenbubbles_{pop_size}_{n_reps}reps'

run_cfg = {
    'folder':       folder,
    'n_cpus':       15, # Manually set the number of CPUs -- otherwise calculated automatically
    'cpu_thresh':   0.75, # Don't use more than this amount of available CPUs, if number of CPUs is not set
    'mem_thresh':   0.75, # Don't use more than this amount of available RAM, if number of CPUs is not set
    'parallel':     True, # Only switch to False for debugging
    'shrink':       True, #
    'verbose':      1 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)
}


class rewire_bubbles(cv.Intervention):

    def __init__(self, frac_edges_to_rewire=0.5):
        self.frac_edges_to_rewire = frac_edges_to_rewire


    def initialize(self, sim):
        if self.frac_edges_to_rewire == 0:
            return

        school_contacts = []

        sdf = sim.people.contacts['s'].to_df()
        student_flag = np.array(sim.people.student_flag, dtype=bool)
        sdf['p1_student'] = student_flag[sdf['p1']]
        sdf['p2_student'] = student_flag[sdf['p2']]
        school_types = sim.people.school_types
        for school_type, scids in school_types.items():
            for school_id in scids:
                uids = sim.people.schools[school_id] # Dict with keys of school_id and values of uids in that school
                edges_this_school = sdf.loc[ ((sdf['p1'].isin(uids)) | (sdf['p2'].isin(uids))) ]
                # When first implemented , I only rewired schools that were due to open
                # Could do that here, but it requires more information
                # Easier to rewire all school connections, and it shouldn't matter for the ones that do not open
                #if scen[school_type] is None:
                #    school_contacts.append(edges_this_school)
                #else:
                student_to_student_edge_bool = ( edges_this_school['p1_student'] & edges_this_school['p2_student'] )
                student_to_student_edges = edges_this_school.loc[ student_to_student_edge_bool ]
                inds_to_rewire = np.random.choice(student_to_student_edges.index, size=int(self.frac_edges_to_rewire*student_to_student_edges.shape[0]), replace=False)
                if len(inds_to_rewire) == 0:
                    # Nothing to do here!
                    continue
                inds_to_keep = np.setdiff1d(student_to_student_edges.index, inds_to_rewire)

                edges_to_rewire = student_to_student_edges.loc[inds_to_rewire]
                stublist = np.concatenate(( edges_to_rewire['p1'], edges_to_rewire['p2'] ))

                p1_inds = np.random.choice(len(stublist), size=len(stublist)//2, replace=False)
                p2_inds = np.setdiff1d(range(len(stublist)), p1_inds)
                p1 = stublist[p1_inds]
                p2 = stublist[p2_inds]
                new_edges = pd.DataFrame({'p1':p1, 'p2':p2})
                new_edges['beta'] = cv.defaults.default_float(1.0)
                # Remove self loops
                new_edges = new_edges.loc[new_edges['p1'] != new_edges['p2']]

                rewired_student_to_student_edges = pd.concat([
                    student_to_student_edges.loc[inds_to_keep, ['p1', 'p2', 'beta']], # Keep these
                    new_edges])

                print(f'During rewiring, the number of student-student edges went from {student_to_student_edges.shape[0]} to {rewired_student_to_student_edges.shape[0]}')

                other_edges = edges_this_school.loc[ (~edges_this_school['p1_student']) | (~edges_this_school['p2_student']) ]
                rewired_edges_this_school = pd.concat([rewired_student_to_student_edges, other_edges])
                school_contacts.append(rewired_edges_this_school)

        if len(school_contacts) > 0:
            all_school_contacts = pd.concat(school_contacts)
            sim.people.contacts['s'] = cv.Layer().from_df(all_school_contacts)

    def apply(self, sim):
        pass


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


    # Configure alternate sus
    rep_levels = {'Yes' if p else 'No':p for p in [True]}
    b.add_level('AltSus', rep_levels, ut.alternate_symptomaticity)

    # Add cohort rewiring
    bubble_scens = {
        'No rewire':  0.0,
        '50% rewire': 0.5,
        '90% rewire': 0.9,
    }
    def break_bubbles(config, key, rewire_frac): # Generic to screen pars, move to builder
        print(f'Building broken bubbles {key}={rewire_frac}')
        # Should come BEFORE the school intervention
        config.interventions.append(rewire_bubbles(rewire_frac))
        return config
    b.add_level('Rewire Bubbles', bubble_scens, break_bubbles)

    # Add school intervention
    for config in b.configs:
        config.interventions.append(cvsch.schools_manager(config.school_config))

    # Add reps
    rep_levels = {f'Run {p}':{'rand_seed':p} for p in range(n_reps)}
    b.add_level('eidx', rep_levels, b.simpars_func)

    return b.get()


def plot(sims, ts_plots=None):
    imgdir = os.path.join(folder, 'img_'+stem)
    p = pt.Plotting(sims, imgdir)

    p.introductions_reg(hue_key='Rewire Bubbles')
    p.outbreak_reg(hue_key='Rewire Bubbles')

    if ts_plots is not None:
        p.several_timeseries(ts_plots)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    ts_plots = {
        'Prevalence':      dict(channel='n_exposed', normalize=True),
        'CumInfections':   dict(channel='cum_infections', normalize=True),
        'Quarantined':     dict(channel='n_quarantined', normalize=True),
        'Newly Diagnosed': dict(channel='new_diagnoses', normalize=True),
    }

    cachefn = os.path.join(folder, 'sims', f'{stem}.sims') # Might need to change the extension here, depending in combine.py was used
    if args.force or not os.path.isfile(cachefn):
        sim_configs = build_configs()
        sims = ut.run_configs(sim_configs, stem, run_cfg, cachefn)
    else:
        print(f'Loading {cachefn}')
        sims = cv.load(cachefn) # Use for *.sims

    plot(sims, ts_plots)
