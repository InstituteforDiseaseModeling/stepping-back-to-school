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
from pathlib import Path
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import copy
import builder as bld
import plotting as pt
import utils as ut


# Check that versions are correct
cv.check_save_version('2.0.0', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

force_run = True
for_presentation = False # Choose between report style and presentation style (different aspect ratio)
simple_plot = False # Boolean flag to select a subset of the scenarios
n_reps = 1
pop_size = 223_000
folder = 'v2020-12-16'
stem = f'import_{pop_size}_{n_reps}reps'

run_cfg = {
    'alternate_symptomaticity':  True, # !!!!!!!!!!!

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

    prev_levels = np.linspace(0.002, 0.02, 20)

    # For a test run only use a subset of scenarios
    scenario_keys = ['with_countermeasures'] # 'with_countermeasures', 'k5', 'all_hybrid', 'all_remote'
    testing_keys = ['None']

    sim_pars = {
        'Baseline': {
            'pop_infected': 100,
            'pop_size':     pop_size,
            'change_beta':  1,
            'symp_prob':    0.08,
            'asymp_factor': 0.8,
            'start_day':    '2020-12-01', # First day of sim
            'end_day':      '2021-04-30', # Last day of sim
        }
    }

    scen_pars = {
        'No symptom screening': { 'screen_prob': 0 },
        '50% daily screening':  { 'screen_prob': 0.5 },
        '100% daily screening': { 'screen_prob': 1 },
    }

    return bld.build(prev_levels, scenario_keys, testing_keys, scen_pars, sim_pars, n_reps, folder)


def plot(sims, to_plot):
    variant = stem + '_simple' if simple_plot else stem
    imgdir = os.path.join(folder, 'img_'+variant)
    Path(imgdir).mkdir(parents=True, exist_ok=True)

    figsize = (12,8) if for_presentation else (12,9.5)
    aspect = 3 if for_presentation else 2.5

    T = sc.tic() # Start timing

    sim_test_names = list(set([sim.tkey for sim in sims]))
    test_names = t_s.testing_map()
    if simple_plot:
        # Select a subset
        test_order = [v[0] for k,v in test_names.items() if k in [ 'None', 'PCR every 4w', 'PCR every 2w', 'PCR every 1w'] if k in sim_test_names]
    else:
        test_order = [v[0] for k,v in test_names.items() if k in sim_test_names]

    inferno_black_bad = copy.copy(mplt.cm.get_cmap('inferno'))
    inferno_black_bad.set_bad((0,0,0))

    #%% Process the simulations
    results = []
    groups = ['students', 'teachers', 'staff']
    origin = []
    detected = []
    for sim in sims:
        first_date = '2021-02-01' # TODO: Read from sim
        last_date = '2021-04-30'
        first_school_day = sim.day(first_date)
        last_school_day = sim.day(last_date)
        sim.tkey = test_names[sim.tkey][0] if sim.tkey in test_names else sim.tkey

        ret = {
            'skey': sim.skey,
            'tkey': sim.tkey,
            'prev': sim.prev,
            'ikey': sim.ikey,
            'rep': sim.eidx,
        }

        ret['n_introductions'] = 0
        ret['in_person_days'] = 0
        ret['introductions'] = []
        ret['introductions_per_100_students'] = []
        ret['introductions_postscreen'] = []
        ret['introductions_postscreen_per_100_students'] = []
        ret['outbreak_size'] = []

        n_schools = {'es':0, 'ms':0, 'hs':0}
        n_schools_with_inf_d1 = {'es':0, 'ms':0, 'hs':0}

        grp_dict = {'Students': ['students'], 'Teachers & Staff': ['teachers', 'staff'], 'Students, Teachers, and Staff': ['students', 'teachers', 'staff']}
        perc_inperson_days_lost = {k:[] for k in grp_dict.keys()}
        attackrate = {k:[] for k in grp_dict.keys()}
        count = {k:0 for k in grp_dict.keys()}
        exposed = {k:0 for k in grp_dict.keys()}
        inperson_days = {k:0 for k in grp_dict.keys()}
        possible_days = {k:0 for k in grp_dict.keys()}

        if sim.results['n_exposed'][first_school_day] == 0:
            print(f'Sim has zero exposed, skipping: {ret}\n')
            continue

        for sid,stats in sim.school_stats.items():
            if stats['type'] not in ['es', 'ms', 'hs']:
                continue

            inf_first = stats['infectious_first_day_school'] # Post-screening
            in_person = stats['in_person']
            exp = stats['newly_exposed']
            num_school_days = stats['num_school_days']
            possible_school_days = np.busday_count(first_date, last_date)
            n_exp = {}
            for grp in groups:
                n_exp[grp] = exp[grp]

            for gkey, grps in grp_dict.items():
                in_person_days = scheduled_person_days = num_exposed = num_people = 0
                for grp in grps:
                    in_person_days += in_person[grp]
                    scheduled_person_days += num_school_days * stats['num'][grp]
                    num_exposed += n_exp[grp]
                    num_people += stats['num'][grp]
                    exposed[gkey] += n_exp[grp]
                    count[gkey] += stats['num'][grp]

                perc_inperson_days_lost[gkey].append(
                    100*(scheduled_person_days - in_person_days)/scheduled_person_days if scheduled_person_days > 0 else 100
                )
                attackrate[gkey].append( 100 * num_exposed / num_people)

                inperson_days[gkey] += in_person_days
                possible_days[gkey] += possible_school_days*num_people

            n_schools[stats['type']] += 1
            if sum([inf_first[g] for g in groups]) > 0:
                n_schools_with_inf_d1[stats['type']] += 1

            ret['n_introductions'] += len(stats['outbreaks'])
            ret['in_person_days'] += np.sum([v for v in stats['in_person'].values()])
            ret['introductions'].append(len(stats['outbreaks']))
            ret['introductions_per_100_students'].append( len(stats['outbreaks']) / stats['num']['students'] * 100 )
            intr_postscreen = len([o for o in stats['outbreaks'] if o['Total infectious days at school']>0]) # len(stats['outbreaks'])
            ret['introductions_postscreen'].append(intr_postscreen)
            ret['introductions_postscreen_per_100_students'].append( intr_postscreen / stats['num']['students'] * 100 )
            ret['outbreak_size'] += [ob['Infected Students'] + ob['Infected Teachers'] + ob['Infected Staff'] for ob in stats['outbreaks']]

            for ob in stats['outbreaks']:
                for ty, lay in zip(ob['Origin type'], ob['Origin layer']):
                    origin.append([stats['type'], sim.skey, sim.tkey, sim.prev, ty, lay])

                    uids = [int(u) for u in ob['Tree'].nodes]
                    data = [v for u,v in ob['Tree'].nodes.data()]
                    was_detected = [(u,d) for u,d in zip(uids, data) if not np.isnan(d['date_diagnosed']) and d['type'] != 'Other']
                    if any(was_detected):
                        first = sorted(was_detected, key=lambda x:x[1]['date_symptomatic'])[0]
                        detected.append([stats['type'], sim.skey, sim.tkey, sim.prev, first[1]['type'], 'Unknown'])

                if to_plot['Debug trees'] and sim.ikey == 'none':# and intr_postscreen > 0:
                    pt.plot_tree(ob['Tree'], stats, sim.pars['n_days'], do_show=True)

        for stype in ['es', 'ms', 'hs']:
            ret[f'{stype}_perc_d1'] = 100 * n_schools_with_inf_d1[stype] / n_schools[stype]

        # Deciding between district and school perspective here
        for gkey in grp_dict.keys():
            ret[f'perc_inperson_days_lost_{gkey}'] = 100*(possible_days[gkey]-inperson_days[gkey])/possible_days[gkey] #np.mean(perc_inperson_days_lost[gkey])
            ret[f'attackrate_{gkey}'] = 100*exposed[gkey] / count[gkey] #np.mean(attackrate[gkey])
            ret[f'count_{gkey}'] = np.sum(count[gkey])

        results.append(ret)

    # Convert results to a dataframe
    df = pd.DataFrame(results)

    # Wrangling
    cols = ['outbreak_size', 'introductions_postscreen_per_100_students', 'introductions_per_100_students']
    q = pd.melt(df, id_vars=['skey', 'tkey', 'ikey', 'prev'], value_vars=cols, var_name='indicator', value_name='value') \
        .set_index(['indicator', 'skey', 'tkey', 'ikey', 'prev'])['value'] \
        .apply(func=lambda x: pd.Series(x)) \
        .stack() \
        .dropna() \
        .to_frame(name='value')
    q.index.rename('outbreak_idx', level=5, inplace=True)


    ################
    # PLOT RESULTS #
    ################

    # TODO: PIE CHART!
    def tab(lbl, df):
        ct = pd.crosstab(df['Type'], df['Dx Screening'])
        ct['total'] = ct.sum(axis=1)
        ct['total'] = 100*ct['total']/ct['total'].sum()
        print('\n'+lbl+'\n', ct)

    odf = pd.DataFrame(origin, columns=['School Type', 'Scenario', 'Dx Screening', 'Prevalence', 'Type', 'Layer'])
    ddf = pd.DataFrame(detected, columns=['School Type', 'Scenario', 'Dx Screening', 'Prevalence', 'Type', 'Layer'])
    es = ddf.loc[ddf['School Type']=='es']

    tab('All', odf)
    tab('All Detected', ddf)
    tab('Detected ES Only', es)


    if to_plot['Prevalence']:
        fig, ax = plt.subplots(figsize=((10,6)))
        l = []
        for sim in sims:
            t = sim.results['t'] #pd.to_datetime(sim.results['date'])
            y = sim.results['n_exposed'].values / sim.pars['pop_size']
            d = pd.DataFrame({'Prevalence': y}, index=pd.Index(data=t, name='Date'))
            d['Prevalence Target'] = f'{sim.prev}'
            d['Scenario'] = f'{sim.skey} + {sim.tkey}'
            d['Rep'] = sim.eidx
            l.append( d )
        d = pd.concat(l).reset_index()

        fig, ax = plt.subplots(figsize=(16,10))
        sns.lineplot(data=d, x='Date', y='Prevalence', hue='Prevalence Target', style='Scenario', palette='cool', ax=ax, legend=False)
        # Y-axis gets messed up when I introduce horizontal lines!
        #for prev in d['Prevalence Target'].unique():
        #    ax.axhline(y=prev, ls='--')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        cv.savefig(os.path.join(imgdir, f'Prevalence.png'), dpi=300)


    # Regression ##################################################################
    if to_plot['IntroductionsRegression']:
        ##### Introductions
        d = q.loc['introductions_postscreen_per_100_students'].reset_index(['ikey', 'prev'])[['ikey', 'prev', 'value']]
        sns.lmplot(data=d, x='prev', y='value', hue='ikey', height=10, x_estimator=np.mean, order=2)
        cv.savefig(os.path.join(imgdir, f'IntroductionsRegression.png'), dpi=300)


    if to_plot['OutbreakSizeRegression']:
        ##### OUTBREAK SIZE
        d = q.loc['outbreak_size'].reset_index(['ikey', 'prev'])[['ikey', 'prev', 'value']]
        sns.lmplot(data=d, x='prev', y='value', hue='ikey', height=10, x_estimator=np.mean, order=2)
        cv.savefig(os.path.join(imgdir, f'OutbreakSizeRegression.png'), dpi=300)


if __name__ == '__main__':

    # Which figures to plot
    to_plot = {
        'Prevalence':               False, # Plot prevalence longitudinally
        'IntroductionsRegression':  True,
        'OutbreakSizeRegression':   True,
        'Debug trees':              False, # Show each introduced tree for debugging
    }

    cachefn = os.path.join(folder, 'sims', f'{stem}.sims') # Might need to change the extension here, depending in combine.py was used
    if force_run or not os.path.isfile(cachefn):
        sim_configs = generate_configs()
        sims = ut.run_configs(sim_configs, stem, run_cfg, cachefn)
    else:
        print(f'Loading {cachefn}')
        sims = cv.load(cachefn) # Use for *.sims

    plot(sims)
