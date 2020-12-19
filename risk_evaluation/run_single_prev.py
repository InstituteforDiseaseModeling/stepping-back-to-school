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
import pandas as pd
import sciris as sc
import synthpops as sp
import covasim_schools as cvsch
import testing_scenarios as t_s # From the local folder
import covasim_controller as cvc
import utils as ut
import plotting as pt
import matplotlib.pyplot as plt


# Check that versions are correct
cv.check_save_version('2.0.0', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

force_run = False
pop_size = 223_000
n_reps = 4
folder = 'v2020-12-16'
stem = f'truetest_{pop_size}_{n_reps}reps'
start_day = '2021-02-01' # First day of school
disable_trees = False

run_cfg = {
    'alternate_symptomaticity':  True,
    'n_cpus':       None, # Manually set the number of CPUs -- otherwise calculated automatically
    'cpu_thresh':   0.75, # Don't use more than this amount of available CPUs, if number of CPUs is not set
    'mem_thresh':   0.75, # Don't use more than this amount of available RAM, if number of CPUs is not set
    'parallel':     True, # Only switch to False for debugging
    'shrink':       False, #
    'verbose':      0.1 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)
}

def generate_configs(start_day = '2021-02-01'):
    # Build simulation configuration
    sc.heading('Creating sim configurations...')
    sim_configs = []
    count = -1

    skey = 'with_countermeasures' #skey = 'all_remote'
    base_scen = t_s.generate_scenarios(start_day)[skey]
    if disable_trees:
        for k,v in base_scen.items():
            if v is None:
                continue
            v['save_trees'] = False

    tkey = 'None'
    test = t_s.generate_testing(start_day)[tkey]

    pars = {
        'pop_infected': 100,
        'change_beta':  1,
        'symp_prob':    0.08,
        'asymp_factor': 0.8,
        'start_day':    '2020-12-01', # First day of sim
        'end_day':      '2021-04-30', # Last day of sim
    }

    prev = 0.009 # (Like 0.65*12/9 from RAINIER)

    for eidx in range(n_reps):
        count += 1
        p = sc.dcp(pars)
        p['rand_seed'] = eidx# np.random.randint(1e6)

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
        ctr = cvc.controller_intervention(seir, targets, pole_loc=pole_loc, start_day=1)
        #######################################################

        # Modify base_scen with testing intervention
        this_scen = sc.dcp(base_scen)
        for stype, spec in this_scen.items():
            if spec is not None:
                spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools

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
    return sim_configs


if __name__ == '__main__':
    filename = os.path.join(folder, 'sims', f'{stem}.sims')

    if force_run or not os.path.isfile(filename):
        sim_configs = generate_configs()
        sims = ut.run_configs(sim_configs, stem, run_cfg, filename)
    else:
        sims = cv.load(filename)

    fig, ax_ctrl = plt.subplots()
    fig, ax_diag = plt.subplots()
    exposed_ages = []
    diagnosed_ages = []
    origin = []
    detected = []
    for idx, sim in enumerate(sims):
        # ['cum_infections', 'cum_infectious', 'cum_tests', 'cum_diagnoses', 'cum_recoveries', 'cum_symptomatic', 'cum_severe', 'cum_critical', 'cum_deaths', 'cum_quarantined', 'new_infections', 'new_infectious', 'new_tests', 'new_diagnoses', 'new_recoveries', 'new_symptomatic', 'new_severe', 'new_critical', 'new_deaths', 'new_quarantined', 'n_susceptible', 'n_exposed', 'n_infectious', 'n_symptomatic', 'n_severe', 'n_critical', 'n_diagnosed', 'n_quarantined', 'n_alive', 'prevalence', 'incidence', 'r_eff', 'doubling_time', 'test_yield', 'rel_test_yield', 'date', 't']
        print('\n', '-'*80)

        for group, flags in [ ('Students', sim.people.student_flag), ('Teachers', sim.people.teacher_flag), ('Staff', sim.people.staff_flag), ('All', True*np.ones(len(sim.people))), ('U10', sim.people.age<10) ]:
            uids = [p for p in sim.people.uid if flags[p]]
            infected_uids = [p for p in uids if not sim.people.susceptible[p]]
            symptomatic_uids = [p for p in infected_uids if not np.isnan(sim.people.date_symptomatic[p])]
            exposed_uids = [p for p in uids if sim.people.exposed[p]]
            diagnosed_in_sch = [p for p in uids if sim.people.date_diagnosed[p] >= sim.day(start_day)]
            n = np.sum(flags)

            print(f'{group} ({n}: cum incidence = {100*len(infected_uids) / len(uids):.2f}, prevalence = {100*len(exposed_uids) / len(uids):.2f} ... num diagnosed after school started = {len(diagnosed_in_sch)}, symptomatic infections = {100*len(symptomatic_uids)/len(infected_uids):.2f}')


        exposed_ages.append([sim.people.age[p] for p in sim.people.uid if sim.people.exposed[p]])
        diagnosed_ages.append([sim.people.age[p] for p in sim.people.uid if sim.people.diagnosed[p]])

        for sid,stats in sim.school_stats.items():
            for ob in stats['outbreaks']:
                for ty, lay in zip(ob['Origin type'], ob['Origin layer']):
                    if len(ty) == 0 or lay not in ['h', 'c']:
                        print('Something strange in this tree!')
                        pt.plot_tree(ob['Tree'], stats, sim.pars['n_days'], do_show=True)
                    #size = ob['Infected Student'] + ob['Infected Teacher'] + ob['Infected Staff']
                    origin.append([stats['type'], sim.key1, sim.key2, sim.key3, ty, lay])

                    # OH - but how would you know who the source was?
                    uids = [int(u) for u in ob['Tree'].nodes]
                    data = [v for u,v in ob['Tree'].nodes.data()]
                    #was_detected = [(u,d) for u,d in zip(uids, data) if sim.people.diagnosed[int(u)] and d['type'] != 'Other']
                    was_detected = [(u,d) for u,d in zip(uids, data) if not np.isnan(d['date_diagnosed']) and d['type'] != 'Other']
                    if any(was_detected):
                        first = sorted(was_detected, key=lambda x:x[1]['date_symptomatic'])[0]
                        detected.append([stats['type'], sim.key1, sim.key2, sim.key3, first[1]['type'], 'Unknown'])
                        #pt.plot_tree(ob['Tree'], stats, sim.pars['n_days'], do_show=True)

        ax_ctrl.plot(sim.pars['interventions'][-1].u_k, label=idx)

        ax_diag.plot(14 * sim.results['new_diagnoses'].values / sim.pars['pop_size'] * 100_000)

    ax_diag.set_title('Diagnoses per 100k over 2w')

    fig, axv = plt.subplots(1,2)
    infected_age_dist = np.concatenate(exposed_ages)
    diagnosed_ages_dist = np.concatenate(diagnosed_ages)
    axv[0].hist(infected_age_dist, bins=20)
    axv[0].set_title('Exposed')
    axv[1].hist(diagnosed_ages_dist, bins=20)
    axv[1].set_title('Diagnosed')


    def tab(lbl, df):
        ct = pd.crosstab(df['Type'], df['Layer'])
        ct['total'] = ct.sum(axis=1)
        ct['total'] = 100*ct['total']/ct['total'].sum()
        print('\n'+lbl)
        print(ct)

    odf = pd.DataFrame(origin, columns=['School Type', 'Scenario', 'Dx Screening', 'Prevalence', 'Type', 'Layer'])
    ddf = pd.DataFrame(detected, columns=['School Type', 'Scenario', 'Dx Screening', 'Prevalence', 'Type', 'Layer'])
    es = ddf.loc[ddf['School Type']=='es']

    tab('All', odf)
    tab('All Detected', ddf)
    tab('Detected ES Only', es)

    cv.MultiSim(sims).plot()

    plt.show()
