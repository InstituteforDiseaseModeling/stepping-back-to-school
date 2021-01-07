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

class Analysis():

    def __init__(self, sims, imgdir):
        self.sims = sims

        self.imgdir = imgdir
        Path(self.imgdir).mkdir(parents=True, exist_ok=True)

        #for_presentation = False # Choose between report style and presentation style (different aspect ratio)
        #figsize = (12,8) if for_presentation else (12,9.5)
        #aspect = 3 if for_presentation else 2.5

        #inferno_black_bad = copy.copy(mplt.cm.get_cmap('inferno'))
        #inferno_black_bad.set_bad((0,0,0))

        sim_scenario_names = list(set([sim.tags['scen_key'] for sim in sims]))
        self.scenario_map = scn.scenario_map()
        self.scenario_order = [v[0] for k,v in self.scenario_map.items() if k in sim_scenario_names]

        sim_screen_names = list(set([sim.tags['dxscrn_key'] for sim in sims]))
        self.dxscrn_map = scn.screening_map()
        self.screen_order = [v[0] for k,v in self.dxscrn_map.items() if k in sim_screen_names]

        self._process()
        keys = list(sims[0].tags.keys()) + ['Scenario', 'Dx Screening', 'Prevalence Target']
        self._wrangle(keys)


    def _process(self):
        ## Process the simulations
        results = []
        groups = ['students', 'teachers', 'staff']
        stypes = ['es', 'ms', 'hs']

        # For introduction source analysis
        #self.origin = []
        #self.detected = []

        for sim in self.sims:
            first_date = '2021-02-01' # TODO: Read from sim
            last_date = '2021-04-30'
            first_school_day = sim.day(first_date)
            last_school_day = sim.day(last_date)

            ret = sc.dcp(sim.tags)
            ret['Prevalence Target'] = ut.p2f(sim.tags['Prevalence'])

            # Map to friendly names
            skey = sim.tags['scen_key']
            tkey = sim.tags['dxscrn_key']
            ret['Scenario'] = self.scenario_map[skey][0] if skey in self.scenario_map else skey
            ret['Dx Screening'] = self.dxscrn_map[tkey][0] if tkey in self.dxscrn_map else tkey

            ret['n_introductions'] = 0
            ret['cum_incidence'] = []
            ret['in_person_days'] = 0
            #ret['introductions'] = []
            ret['introductions_per_100_students'] = []
            ret['introductions_postscreen'] = []
            ret['introductions_postscreen_per_100_students'] = []
            ret['outbreak_size'] = []
            ret['introduction_first_symptomatic_date'] = []
            ret['introduction_first_symptomatic_size'] = []
            for stype in stypes:
                ret[f'introductions_{stype}'] = []
                ret[f'introductions_postscreen_{stype}'] = []
                ret[f'susceptible_person_days_{stype}'] = []

            for grp in ['Student', 'Teacher', 'Staff']:
                ret[f'introduction_origin_{grp}'] = 0
                ret[f'diagnosed_origin_{grp}'] = 0

            n_schools = {'es':0, 'ms':0, 'hs':0}
            n_schools_with_inf_d1 = {'es':0, 'ms':0, 'hs':0}

            grp_dict = {'Students': ['students'], 'Teachers & Staff': ['teachers', 'staff'], 'Students, Teachers, and Staff': ['students', 'teachers', 'staff']}
            perc_inperson_days_lost = {k:[] for k in grp_dict.keys()}
            count = {k:0 for k in grp_dict.keys()}
            exposed = {k:0 for k in grp_dict.keys()}
            inperson_days = {k:0 for k in grp_dict.keys()}
            possible_days = {k:0 for k in grp_dict.keys()}

            if sim.results['n_exposed'][first_school_day] == 0:
                print(f'Sim has zero exposed, skipping: {ret}\n')
                continue

            for sid,stats in sim.school_stats.items():
                stype = stats['type']
                if stype not in stypes:
                    continue

                inf_first = stats['infectious_first_day_school'] # Post-screening
                in_person = stats['in_person']
                n_exp = stats['newly_exposed']
                num_school_days = stats['num_school_days']
                possible_school_days = np.busday_count(first_date, last_date)

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

                    inperson_days[gkey] += in_person_days
                    possible_days[gkey] += possible_school_days*num_people

                n_schools[stats['type']] += 1
                if sum([inf_first[g] for g in groups]) > 0:
                    n_schools_with_inf_d1[stats['type']] += 1

                # TODO: By school type and student/staff
                ret['n_introductions'] += len(stats['outbreaks'])
                ret['cum_incidence'].append(100 * stats['cum_incidence'] / (stats['num']['students'] + stats['num']['teachers'] + stats['num']['staff']))
                ret['in_person_days'] += np.sum([v for v in stats['in_person'].values()])
                ret[f'introductions_{stype}'].append( len(stats['outbreaks']) )
                ret[f'susceptible_person_days_{stype}'].append( stats['susceptible_person_days'] )
                ret['introductions_per_100_students'].append( len(stats['outbreaks']) / stats['num']['students'] * 100 )
                intr_postscreen = len([o for o in stats['outbreaks'] if o['Total infectious days at school']>0]) # len(stats['outbreaks'])
                ret[f'introductions_postscreen_{stype}'].append( intr_postscreen )
                ret['introductions_postscreen'].append(intr_postscreen)
                ret['introductions_postscreen_per_100_students'].append( intr_postscreen / stats['num']['students'] * 100 )
                ret['outbreak_size'] += [ob['Infected Students'] + ob['Infected Teachers'] + ob['Infected Staff'] for ob in stats['outbreaks']]

                # Origin analysis
                for ob in stats['outbreaks']:
                    if ob['Total infectious days at school']==0:# or ob['Infected Students']+ob['Infected Teachers']+ob['Infected Staff']<2:
                        # Only count outbreaks in which an infectious person was physically at school
                        continue

                    for origin_type, lay in zip(ob['Origin type'], ob['Origin layer']):
                        #self.origin.append([sid, stats['type'], ret['Scenario'], ret['Dx Screening'], ret['Prevalence Target'], origin_type, lay])
                        ret[f'introduction_origin_{origin_type}'] += 1

                        uids = [int(u) for u in ob['Tree'].nodes]
                        data = [v for u,v in ob['Tree'].nodes.data()]
                        was_detected = [(u,d) for u,d in zip(uids, data) if not np.isnan(d['date_diagnosed']) and d['type'] != 'Other']
                        #if any(was_detected):
                        if len(was_detected) > 0:
                            first = sorted(was_detected, key=lambda x:x[1]['date_symptomatic'])[0]
                            #self.detected.append([sid, stats['type'], ret['Scenario'], ret['Dx Screening'], ret['Prevalence Target'], first[1]['type'], 'Unknown'])
                            detected_origin_type = first[1]['type']
                            ret[f'diagnosed_origin_{detected_origin_type}'] += 1
                            ret['introduction_first_symptomatic_date'].append(first[1]['date_symptomatic'])
                            ret['introduction_first_symptomatic_size'].append(ob['Infected Students'] + ob['Infected Teachers'] + ob['Infected Staff'])

                    #if to_plot['Debug trees'] and sim.ikey == 'none':# and intr_postscreen > 0:
                    #    pt.plot_tree(ob['Tree'], stats, sim.pars['n_days'], do_show=True)

            for stype in stypes:
                ret[f'{stype}_perc_d1'] = 100 * n_schools_with_inf_d1[stype] / n_schools[stype]
                # Sums, won't allow for full bootstrap resampling!
                ret[f'introductions_sum_{stype}'] = np.sum(ret[f'introductions_{stype}'])
                ret[f'introductions_postscreen_sum_{stype}'] = np.sum(ret[f'introductions_postscreen_{stype}'])
                ret[f'susceptible_person_days_sum_{stype}'] = np.sum(ret[f'susceptible_person_days_{stype}'])

            # Deciding between district and school perspective here
            for gkey in grp_dict.keys():
                ret[f'perc_inperson_days_lost_{gkey}'] = 100*(possible_days[gkey]-inperson_days[gkey])/possible_days[gkey] #np.mean(perc_inperson_days_lost[gkey])
                ret[f'attackrate_{gkey}'] = 100*exposed[gkey] / count[gkey]
                ret[f'count_{gkey}'] = np.sum(count[gkey])

            results.append(ret)

        # Convert results to a dataframe
        self.raw = pd.DataFrame(results)

    def _wrangle(self, keys, outputs=None):
        # Wrangling - build self.results from self.raw
        stypes = ['es', 'ms', 'hs']
        if outputs == None:
            outputs = ['outbreak_size', 'introductions_postscreen_per_100_students', 'introductions_per_100_students']
            outputs += [f'introductions_{stype}' for stype in stypes]
            outputs += [f'introductions_postscreen_{stype}' for stype in stypes]
            outputs += [f'susceptible_person_days_{stype}' for stype in stypes]
            outputs += ['introduction_first_symptomatic_date', 'introduction_first_symptomatic_size']

        self.results = pd.melt(self.raw, id_vars=keys, value_vars=outputs, var_name='indicator', value_name='value') \
            .set_index(['indicator']+keys)['value'] \
            .apply(func=lambda x: pd.Series(x)) \
            .stack() \
            .dropna() \
            .to_frame(name='value')
        self.results.index.rename('outbreak_idx', level=1+len(keys), inplace=True)

        # Separate because different datatype (ndarray vs float)
        outputs_ts = ['cum_incidence']
        self.restuls_ts = pd.melt(self.raw, id_vars=keys, value_vars=outputs_ts, var_name='indicator', value_name='value') \
            .set_index(['indicator']+keys)['value'] \
            .apply(func=lambda x: pd.Series(x)) \
            .stack() \
            .dropna() \
            .to_frame(name='value')
        self.restuls_ts.index.rename('outbreak_idx', level=1+len(keys), inplace=True)


    def cum_incidence(self):
        def draw_cum_inc(**kwargs):
            data = kwargs['data']
            mat = np.vstack(data['value'])
            #plt.plot(mat.T, lw=0.2)
            sns.heatmap(mat, vmax=kwargs['vmax'])
        d = self.restuls_ts.loc['cum_incidence']
        vmax = np.vstack(d['value']).max().max()
        g = sns.FacetGrid(data=d.reset_index(), row='In-school transmission multiplier', col='Prevalence', height=5, aspect=1.4)
        g.map_dataframe(draw_cum_inc, vmax=vmax)
        g.set(xlim=(40,None)) # TODO determine from school opening day
        cv.savefig(os.path.join(self.imgdir, f'SchoolCumInc.png'), dpi=300)

    def outbreak_size_over_time(self):
        #if 'introduction_first_symptomatic_date' not in self.results:
        #    return

        d = self.results \
            .loc[['introduction_first_symptomatic_date', 'introduction_first_symptomatic_size']] \
            .unstack('indicator')['value']
        print(d.dtypes)
        print(d['introduction_first_symptomatic_date'].unique())
        print(d['introduction_first_symptomatic_size'].unique())
        sns.lmplot(data=d.reset_index(), x='introduction_first_symptomatic_date', y='introduction_first_symptomatic_size', hue='Prevalence', row='In-school transmission multiplier', col='Prevalence', scatter_kws={'s': 7}, x_jitter=True, markers='.', height=10, aspect=1)#, discrete=True, multiple='dodge')
        cv.savefig(os.path.join(self.imgdir, f'OutbreakSizeOverTime.png'), dpi=300)

    def source_pie(self):
        groups = ['Student', 'Teacher', 'Staff']
        cols = [f'introduction_origin_{origin_type}' for origin_type in groups]
        intro_by_origin = self.raw[cols]
        intro_by_origin.rename(columns={f'introduction_origin_{origin_type}':origin_type for origin_type in ['Student', 'Teacher', 'Staff']}, inplace=True)
        intro_by_origin.loc[:, 'Introductions'] = 'All'

        cols = [f'diagnosed_origin_{origin_type}' for origin_type in groups]
        detected_intro_by_origin = self.raw[cols]
        detected_intro_by_origin.rename(columns={f'diagnosed_origin_{origin_type}':origin_type for origin_type in ['Student', 'Teacher', 'Staff']}, inplace=True)
        detected_intro_by_origin.loc[:, 'Introductions'] = 'Diagnosed'

        df = pd.concat([intro_by_origin, detected_intro_by_origin], ignore_index=True)
        df = df.set_index('Introductions').rename_axis('Source', axis=1).stack()
        df.name='Count'
        intr_src = df.reset_index().groupby(['Introductions', 'Source'])['Count'].sum().unstack('Source')


        fig, axv = plt.subplots(1, intr_src.shape[0], figsize=(10,5))
        for ax, (idx, row) in zip(axv, intr_src.iterrows()):
            pie = ax.pie(row.values, explode=[0.05]*len(row), autopct='%.0f%%')
            ax.set_title(idx)
        axv[1].legend(pie[0], intr_src.columns, bbox_to_anchor=(0.0,-0.2), loc='lower center', ncol=intr_src.shape[1], frameon=True)
        cv.savefig(os.path.join(self.imgdir, f'SourcePie.png'), dpi=300)


    def introductions_rate_by_stype(self, xvar, huevar, order=2):
        '''
        # E.g. for bootstrap resampling, if desired (LATER)
        num = self.results.loc['introductions_es']
        den = self.results.loc['susceptible_person_days_es']
        print('INTRODUCTIONS (ES)\n', num)
        print('SUS PERSON DAYS (ES)\n', den)
        '''

        # Just plotting the mean:
        stypes = ['es', 'ms', 'hs']
        d = []
        factor = 100_000
        for i, stype in enumerate(stypes):
            num = factor * self.raw.groupby([huevar, xvar])[f'introductions_postscreen_sum_{stype}'].sum() # without 'Replicate' in index, summing over replicates
            den = self.raw.groupby([huevar, xvar])[f'susceptible_person_days_sum_{stype}'].sum()
            tmp = num/den
            tmp.name=f'Introduction Rate (post-screening) per {factor}'
            d.append(tmp.to_frame())
            d[i]['stype'] = stype
        D = pd.concat(d)

        g = sns.lmplot(data=D.reset_index(), col='stype', x=xvar, y=f'Introduction Rate (post-screening) per {factor}', hue=huevar, height=10)
        g.set(ylim=(0,None))
        cv.savefig(os.path.join(self.imgdir, f'IntroductionRateStype.png'), dpi=300)

    def introductions_rate(self, xvar, huevar, order=2):
        '''
        num = self.results.loc['introductions_es'] # Sum over school types...
        den = self.results.loc['susceptible_person_days_es']
        print('INTRODUCTIONS (ES)\n', num)
        print('SUS PERSON DAYS (ES)\n', den)
        '''

        # Just plotting the mean:
        stypes = ['es', 'ms', 'hs']
        num_cols = [f'introductions_postscreen_sum_{stype}' for stype in stypes]
        den_cols = [f'susceptible_person_days_sum_{stype}' for stype in stypes]

        factor = 100_000
        num = factor * self.raw.groupby([huevar, xvar])[num_cols].sum().sum(axis=1) # without 'Replicate' in index, summing over replicates
        den = self.raw.groupby([huevar, xvar])[den_cols].sum().sum(axis=1)
        d = num/den
        d.name=f'Introduction Rate (post-screening) per {factor}'

        g = sns.lmplot(data=d.reset_index(), x=xvar, y=f'Introduction Rate (post-screening) per {factor}', hue=huevar, height=10, legend_out=False)
        g.set(ylim=(0,None))
        cv.savefig(os.path.join(self.imgdir, f'IntroductionRate.png'), dpi=300)


    def introductions_reg(self, xvar, huevar, order=2):
        #if 'introductions_postscreen_per_100_students' not in self.results:
        #    return

        ##### Introductions
        d = self.results.loc['introductions_postscreen_per_100_students'].reset_index([xvar, huevar])[[xvar, huevar, 'value']]
        g = sns.lmplot(data=d, x=xvar, y='value', hue=huevar, height=10, x_estimator=np.mean, order=order, legend_out=False)
        #ax = g.axes.flat[0]
        #sns.relplot(data=d, x=xvar, y='value', hue=huevar, ax=ax)
        #g = sns.lmplot(data=d, x=xvar, y='value', hue=huevar, markers='.', x_jitter=0.02, height=10, order=order, legend_out=False)
        g.set(ylim=(0,None))
        g.set_ylabels('Introductions (post-screening) per 100 students')
        cv.savefig(os.path.join(self.imgdir, f'IntroductionsRegression.png'), dpi=300)


    def outbreak_reg(self, xvar, huevar, order=2):
        #if 'outbreak_size' not in self.results:
        #    return

        ##### Outbreak size
        d = self.results.loc['outbreak_size'].reset_index([xvar, huevar])[[xvar, huevar, 'value']]
        g =sns.lmplot(data=d, x=xvar, y='value', hue=huevar, height=10, x_estimator=np.mean, order=2, legend_out=False)
        #sns.lmplot(data=d, x=xvar, y='value', hue=huevar, markers='.', x_jitter=0.02, height=10, order=order, legend_out=False)
        g.set(ylim=(0,None))
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
            d['Prevalence Target'] = ut.p2f(sim.tags['Prevalence'])
            d['Scenario'] = f'{sim.tags["scen_key"]} + {sim.tags["dxscrn_key"]}'
            d['Replicate'] = sim.tags['Replicate']
            l.append( d )
        d = pd.concat(l).reset_index()

        fig, ax = plt.subplots(figsize=(16,10))
        sns.lineplot(data=d, x='Date', y=label, hue='Prevalence Target', style='Scenario', palette='cool', ax=ax, legend=False)
        # Y-axis gets messed up when I introduce horizontal lines!
        #for prev in d['Prevalence Target'].unique():
        #    ax.axhline(y=prev, ls='--')
        if normalize:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        cv.savefig(os.path.join(self.imgdir, f'{label}.png'), dpi=300)
        return fig

    def several_timeseries(self, config):
        for config in config:
            self.timeseries(config['channel'], config['label'], config['normalize'])


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
