import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import sciris as sc
import covasim as cv
import scenarios as scn
import utils as ut

# Global plotting styles
font_size = 18
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style

class Plotting():

    def __init__(self, sims, imgdir):
        self.sims = sims

        self.imgdir = imgdir
        Path(self.imgdir).mkdir(parents=True, exist_ok=True)

        #for_presentation = False # Choose between report style and presentation style (different aspect ratio)
        #figsize = (12,8) if for_presentation else (12,9.5)
        #aspect = 3 if for_presentation else 2.5

        #inferno_black_bad = copy.copy(mplt.cm.get_cmap('inferno'))
        #inferno_black_bad.set_bad((0,0,0))

        sim_scenario_names = list(set([sim.tags['skey'] for sim in sims]))
        self.scenario_map = scn.scenario_map()
        self.scenario_order = [v[0] for k,v in self.scenario_map.items() if k in sim_scenario_names]

        sim_screen_names = list(set([sim.tags['tkey'] for sim in sims]))
        self.screen_map = scn.screening_map()
        self.screen_order = [v[0] for k,v in self.screen_map.items() if k in sim_screen_names]

        self._process()
        rename = {'skey':'sname', 'tkey':'tname', 'prev':'prev_tgt'}
        keys = [rename[k] if k in rename else k for k in sims[0].tags.keys()]
        self._wrangle(keys)

    def _process(self):
        #%% Process the simulations
        results = []
        groups = ['students', 'teachers', 'staff']
        origin = []
        detected = []
        for sim in self.sims:
            first_date = '2021-02-01' # TODO: Read from sim
            last_date = '2021-04-30'
            first_school_day = sim.day(first_date)
            last_school_day = sim.day(last_date)

            ret = sc.dcp(sim.tags)
            ret['prev_tgt'] = ut.p2f(sim.tags['prev'])

            # Map to friendly names
            skey = sim.tags['skey']
            tkey = sim.tags['tkey']
            ret['sname'] = self.scenario_map[skey][0] if skey in self.scenario_map else skey
            ret['tname'] = self.screen_map[tkey][0] if tkey in self.screen_map else tkey

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
                        origin.append([stats['type'], ret['sname'], ret['tname'], ret['prev_tgt'], ty, lay])

                        uids = [int(u) for u in ob['Tree'].nodes]
                        data = [v for u,v in ob['Tree'].nodes.data()]
                        was_detected = [(u,d) for u,d in zip(uids, data) if not np.isnan(d['date_diagnosed']) and d['type'] != 'Other']
                        if any(was_detected):
                            first = sorted(was_detected, key=lambda x:x[1]['date_symptomatic'])[0]
                            detected.append([stats['type'], ret['sname'], ret['tname'], ret['prev_tgt'], first[1]['type'], 'Unknown'])

                    #if to_plot['Debug trees'] and sim.ikey == 'none':# and intr_postscreen > 0:
                    #    pt.plot_tree(ob['Tree'], stats, sim.pars['n_days'], do_show=True)

            for stype in ['es', 'ms', 'hs']:
                ret[f'{stype}_perc_d1'] = 100 * n_schools_with_inf_d1[stype] / n_schools[stype]

            # Deciding between district and school perspective here
            for gkey in grp_dict.keys():
                ret[f'perc_inperson_days_lost_{gkey}'] = 100*(possible_days[gkey]-inperson_days[gkey])/possible_days[gkey] #np.mean(perc_inperson_days_lost[gkey])
                ret[f'attackrate_{gkey}'] = 100*exposed[gkey] / count[gkey] #np.mean(attackrate[gkey])
                ret[f'count_{gkey}'] = np.sum(count[gkey])

            results.append(ret)

        # Convert results to a dataframe
        self.raw = pd.DataFrame(results)

    def _wrangle(self, keys, outputs=None):
        # Wrangling - build self.results from self.raw
        if outputs == None:
            outputs = ['outbreak_size', 'introductions_postscreen_per_100_students', 'introductions_per_100_students']

        self.results = pd.melt(self.raw, id_vars=keys, value_vars=outputs, var_name='indicator', value_name='value') \
            .set_index(['indicator']+keys)['value'] \
            .apply(func=lambda x: pd.Series(x)) \
            .stack() \
            .dropna() \
            .to_frame(name='value')
        self.results.index.rename('outbreak_idx', level=1+len(keys), inplace=True)


    def source_pie(self):
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


    def introductions_reg(self, hue_key):
        ##### Introductions
        d = self.results.loc['introductions_postscreen_per_100_students'].reset_index([hue_key, 'prev_tgt'])[[hue_key, 'prev_tgt', 'value']]
        sns.lmplot(data=d, x='prev_tgt', y='value', hue=hue_key, height=10, x_estimator=np.mean, order=2)
        cv.savefig(os.path.join(self.imgdir, f'IntroductionsRegression.png'), dpi=300)


    def outbreak_reg(self, hue_key):
        ##### OUTBREAK SIZE
        d = self.results.loc['outbreak_size'].reset_index([hue_key, 'prev_tgt'])[[hue_key, 'prev_tgt', 'value']]
        sns.lmplot(data=d, x='prev_tgt', y='value', hue=hue_key, height=10, x_estimator=np.mean, order=2)
        cv.savefig(os.path.join(self.imgdir, f'OutbreakSizeRegression.png'), dpi=300)

    def timeseries(self, channel, label, normalize):
        fig, ax = plt.subplots(figsize=((10,6)))
        l = []
        for sim in self.sims:
            t = sim.results['t'] #pd.to_datetime(sim.results['date'])
            y = sim.results[channel].values
            if normalize:
                y /= sim.pars['pop_size']
            d = pd.DataFrame({label: y}, index=pd.Index(data=t, name='Date'))
            d['prev_tgt'] = ut.p2f(sim.tags['prev'])
            d['Scenario'] = f'{sim.tags["skey"]} + {sim.tags["tkey"]}'
            d['Rep'] = sim.tags['eidx']
            l.append( d )
        d = pd.concat(l).reset_index()

        fig, ax = plt.subplots(figsize=(16,10))
        sns.lineplot(data=d, x='Date', y=label, hue='prev_tgt', style='Scenario', palette='cool', ax=ax, legend=False)
        # Y-axis gets messed up when I introduce horizontal lines!
        #for prev in d['prev_tgt'].unique():
        #    ax.axhline(y=prev, ls='--')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        cv.savefig(os.path.join(self.imgdir, f'{label}.png'), dpi=300)
        return fig

    def several_timeseries(self, config):
        for lbl,v in config.items():
            self.timeseries(v['channel'], lbl, v['normalize'])


# Tree plotting
def plot_tree(tree, stats, n_days, do_show=False):
    fig, ax = plt.subplots(figsize=(16,10))
    date_range = [n_days, 0]

    # TODO: move tree plotting to a function
    #print(f'Tree {i}', sid, sim.key1, sim.key2, sim.key2)
    #for u,v,w in tree.edges.data():
        #print('\tEDGE', u,v,w)
    #print(f'N{i}', sid, sim.key1, sim.key2, sim.key2, tree.nodes.data())
    for j, (u,v) in enumerate(tree.nodes.data()):
        #print('\tNODE', u,v)
        recovered = n_days if np.isnan(v['date_recovered']) else v['date_recovered']
        col = 'gray' if v['type'] == 'Other' else 'black'
        date_range[0] = min(date_range[0], v['date_exposed']-1)
        date_range[1] = max(date_range[1], recovered+1)
        ax.plot( [v['date_exposed'], recovered], [j,j], '-', marker='o', color=col)
        ax.plot( v['date_diagnosed'], j, marker='d', color='b')
        ax.plot( v['date_infectious'], j, marker='|', color='r', mew=3, ms=10)
        ax.plot( v['date_symptomatic'], j, marker='s', color='orange')
        for day in range(int(v['date_exposed']), int(recovered)):
            if day in stats['uids_at_home'] and int(u) in stats['uids_at_home'][day]:
                plt.plot([day,day+1], [j,j], '-', color='lightgray')
        for t, r in stats['testing'].items():
            for kind, outcomes in r.items():
                if int(u) in outcomes['Positive']:
                    plt.plot(t, j, marker='x', color='red', ms=10, lw=2)
                elif int(u) in outcomes['Negative']:
                    plt.plot(t, j, marker='x', color='green', ms=10, lw=2)

    for t, r in stats['testing'].items():
        ax.axvline(x=t, zorder=-100)
    date_range[1] = min(date_range[1], n_days)
    ax.set_xlim(date_range)
    ax.set_xticks(range(int(date_range[0]), int(date_range[1])))
    ax.set_yticks(range(0, len(tree.nodes)))
    ax.set_yticklabels([f'{int(u)}: {v["type"]}, age {v["age"]}' for u,v in tree.nodes.data()])
    #ax.set_title(f'School {sid}, Tree {i}')

    if do_show:
        plt.show()
    else:
        return fig
