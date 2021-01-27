'''
Analyze covasim simulation results and produce plots
'''

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

import warnings
warnings.simplefilter('ignore', np.RankWarning)

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

        sim_scenario_names = list(set([sim.tags['scen_key'] for sim in sims]))
        self.scenario_map = scn.scenario_map()
        self.scenario_order = [v[0] for k,v in self.scenario_map.items() if k in sim_scenario_names]

        sim_screen_names = list(set([sim.tags['dxscrn_key'] for sim in sims]))
        self.dxscrn_map = scn.screening_map()
        self.screen_order = [v[0] for k,v in self.dxscrn_map.items() if k in sim_screen_names]

        self._process()
        keys = list(sims[0].tags.keys()) + ['Scenario', 'Dx Screening']
        keys.remove('school_start_date')
        if 'Prevalence' in sims[0].tags.keys():
            keys.append('Prevalence Target')
        self._wrangle(keys)


    def _process(self):
        ## Process the simulations
        results = []
        stypes = ['es', 'ms', 'hs']
        grp_dict = {'Students': ['students'], 'Teachers & Staff': ['teachers', 'staff'], 'Students, Teachers, and Staff': ['students', 'teachers', 'staff']}

        # Loop over each simulation and accumulate stats
        for sim in self.sims:
            first_school_date = sim.tags['school_start_date']
            last_school_date = sim.pars['end_day'].strftime('%Y-%m-%d')
            first_school_day = sim.day(first_school_date)

            # Map the scenario name and type of diagnostic screening, if any, to friendly names
            skey = sim.tags['scen_key']
            tkey = sim.tags['dxscrn_key']

            # Tags usually contain the relevant sweep dimensions
            ret = sc.dcp(sim.tags)
            ret['Scenario'] = self.scenario_map[skey][0] if skey in self.scenario_map else skey
            ret['Dx Screening'] = self.dxscrn_map[tkey][0] if tkey in self.dxscrn_map else tkey

            if 'Prevalence' in sim.tags:
                # Prevalence appears as a string, e.g. 1%, so convert to a float for plotting
                ret['Prevalence Target'] = ut.p2f(sim.tags['Prevalence'])

            # Initialize a dictionary that will be later transformed into a dataframe.
            # Each simulation will get one value for each key, but not that sometimes the value is a list of values
            ret['n_introductions'] = 0 # Number of times covid was introduced, cumulative across all schools
            ret['cum_incidence'] = [] # List of timeseries of cumulative incidences, one per school
            ret['in_person_days'] = 0 # Total number of in-person days
            ret['introductions_per_100_students'] = [] # Number of introductions per 100 students
            ret['first_infectious_day_at_school'] = [] # List of simulation days, e.g. 56 on which an infectious individual first was in-person, one per outbreak.
            ret['outbreak_size'] = [] # A list, each entry is the sum of num students, teachers, and staff who got infected, including the source
            ret['complete'] = [] # Boolean, 1 if the outbreak completed and 0 if it was still ongoing when the simulation ended
            ret['n_infected_by_seed'] = [] # If the outbreak was seeded, the number of secondary infectious caused by the seed
            ret['exports_to_hh'] = [] # The number of direct exports from a member of the school community to households. Exclused further spread within HHs and indirect routes to HHs e.g. via the community.

            for stype in stypes:
                ret[f'introductions_{stype}'] = [] # Introductions by school type
                ret[f'susceptible_person_days_{stype}'] = [] # Susceptible person-days (amongst the school population) by school type

            for grp in ['Student', 'Teacher', 'Staff']:
                ret[f'introduction_origin_{grp}'] = 0 # Introduction origin by group
                ret[f'observed_origin_{grp}'] = 0 # First diagnosed origin by group, intended to model the "observation" process

            if sim.results['n_exposed'][first_school_day] == 0:
                print(f'Sim has zero exposed, skipping: {ret}\n')
                continue

            # Now loop over each school and collect outbreak stats
            for sid,stats in sim.school_stats.items():
                stype = stats['type']
                if stype not in stypes:
                    continue

                # Only count outbreaks in which total infectious days at school is > 0
                outbreaks = [o for o in stats['outbreaks'] if o['Total infectious days at school']>0]

                ret['n_introductions'] += len(outbreaks)
                ret['cum_incidence'].append(100 * stats['cum_incidence'] / (stats['num']['students'] + stats['num']['teachers'] + stats['num']['staff']))
                ret['in_person_days'] += np.sum([v for v in stats['in_person'].values()])
                ret[f'susceptible_person_days_{stype}'].append( stats['susceptible_person_days'] )
                ret[f'introductions_{stype}'].append( len(outbreaks) )
                ret['introductions_per_100_students'].append( len(outbreaks) / stats['num']['students'] * 100 )

                # Insert at beginning for efficiency
                ret['first_infectious_day_at_school'][0:0] = [o['First infectious day at school'] for o in outbreaks]
                ret['outbreak_size'][0:0] = [ob['Infected Students'] + ob['Infected Teachers'] + ob['Infected Staff'] for ob in outbreaks]
                ret['complete'][0:0] = [float(ob['Complete']) for ob in outbreaks]
                ret['n_infected_by_seed'][0:0] = [ob['Num infected by seed'] for ob in outbreaks if ob['Seeded']]
                ret['exports_to_hh'][0:0] = [ob['Exports to household'] for ob in outbreaks]

                for grp in ['Student', 'Teacher', 'Staff']:
                    ret[f'introduction_origin_{grp}'] += len([o for o in outbreaks if grp in o['Origin type']])

                # Determine the type of the first "observed" case, here using the first diagnosed
                for ob in outbreaks:
                    #if to_plot['Debug trees'] and sim.ikey == 'none':# and intr_postscreen > 0:
                    #if not ob['Complete']:
                    #    self.plot_tree(ob['Tree'], stats, sim.pars['n_days'], do_show=True)

                    date = 'date_diagnosed' # 'date_symptomatic'
                    was_detected = [(int(u),d) for u,d in ob['Tree'].nodes.data() if np.isfinite(u) and d['type'] not in ['Other'] and np.isfinite(d[date])]
                    if len(was_detected) > 0:
                        first = sorted(was_detected, key=lambda x:x[1][date])[0]
                        detected_origin_type = first[1]['type']
                        ret[f'observed_origin_{detected_origin_type}'] += 1

            for stype in stypes:
                # Sums, won't allow for full bootstrap resampling!
                ret[f'introductions_sum_{stype}'] = np.sum(ret[f'introductions_{stype}'])
                ret[f'susceptible_person_days_sum_{stype}'] = np.sum(ret[f'susceptible_person_days_{stype}'])

            results.append(ret)

        # Convert results to a dataframe
        self.raw = pd.DataFrame(results)

    def _wrangle(self, keys, outputs=None):
        # Wrangling - build self.results from self.raw
        stypes = ['es', 'ms', 'hs']
        if outputs == None:
            outputs = ['outbreak_size', 'introductions_per_100_students', 'exports_to_hh']
            outputs += [f'introductions_{stype}' for stype in stypes]
            outputs += [f'susceptible_person_days_{stype}' for stype in stypes]
            outputs += ['first_infectious_day_at_school', 'complete']

        self.results = pd.melt(self.raw, id_vars=keys, value_vars=outputs, var_name='indicator', value_name='value') \
            .set_index(['indicator']+keys)['value'] \
            .apply(func=lambda x: pd.Series(x)) \
            .stack() \
            .dropna() \
            .to_frame(name='value')
        self.results.index.rename('outbreak_idx', level=1+len(keys), inplace=True)

        # Separate because different datatype (ndarray vs float)
        outputs_ts = ['cum_incidence', 'n_infected_by_seed']
        self.results_ts = pd.melt(self.raw, id_vars=keys, value_vars=outputs_ts, var_name='indicator', value_name='value') \
            .set_index(['indicator']+keys)['value'] \
            .apply(func=lambda x: pd.Series(x)) \
            .stack() \
            .dropna() \
            .to_frame(name='value')
        self.results_ts.index.rename('outbreak_idx', level=1+len(keys), inplace=True)


    def cum_incidence(self, rowvar=None, colvar=None):
        def draw_cum_inc(**kwargs):
            data = kwargs['data']
            mat = np.vstack(data['value'])
            sns.heatmap(mat, vmax=kwargs['vmax'])
        d = self.results_ts.loc['cum_incidence']
        vmax = np.vstack(d['value']).max().max()
        colwrap=None
        if rowvar is None:
            colwrap=4
        g = sns.FacetGrid(data=d.reset_index(), row=rowvar, col=colvar, col_wrap=colwrap, height=5, aspect=1.4)
        g.map_dataframe(draw_cum_inc, vmax=vmax)

        sch = next(iter(self.sims[0].school_stats))
        start_date = self.sims[0].school_stats[sch]['scenario']['start_date']
        start_day = self.sims[0].day(start_date)
        g.set(xlim=(start_day, None))

        cv.savefig(os.path.join(self.imgdir, f'SchoolCumInc.png'), dpi=300)
        return g

    def outbreak_size_over_time(self, rowvar=None, colvar=None):
        d = self.results \
            .loc[['first_infectious_day_at_school', 'outbreak_size', 'complete']] \
            .unstack('indicator')['value']

        g = sns.lmplot(data=d.reset_index(), x='first_infectious_day_at_school', y='outbreak_size', hue='complete', row=rowvar, col=colvar, scatter_kws={'s': 7}, x_jitter=True, markers='.', height=10, aspect=1)#, discrete=True, multiple='dodge')
        cv.savefig(os.path.join(self.imgdir, f'OutbreakSizeOverTime.png'), dpi=300)
        return g

    def source_dow(self, figsize=(6,6)):
        fig, ax = plt.subplots(1,1,figsize=figsize)
        sns.histplot(np.hstack(self.results_ts.loc['first_infectious_day_at_school']['value']), discrete=True, stat='probability', ax=ax)
        ax.set_xlabel('Simulation Day')
        ax.set_ylabel('Importations (%)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        fig.tight_layout()
        cv.savefig(os.path.join(self.imgdir, f'IntroductionDayOfWeek.png'), dpi=300)
        return fig

    def source_pie(self):
        groups = ['Student', 'Teacher', 'Staff']
        cols = [f'introduction_origin_{origin_type}' for origin_type in groups]
        intro_by_origin = self.raw[cols]
        intro_by_origin.rename(columns={f'introduction_origin_{origin_type}':origin_type for origin_type in ['Student', 'Teacher', 'Staff']}, inplace=True)
        intro_by_origin.loc[:, 'Introductions'] = 'All'

        cols = [f'observed_origin_{origin_type}' for origin_type in groups]
        detected_intro_by_origin = self.raw[cols]
        detected_intro_by_origin.rename(columns={f'observed_origin_{origin_type}':origin_type for origin_type in ['Student', 'Teacher', 'Staff']}, inplace=True)
        detected_intro_by_origin.loc[:, 'Introductions'] = 'First Symptomatic'

        df = pd.concat([intro_by_origin, detected_intro_by_origin], ignore_index=True)
        df = df.set_index('Introductions').rename_axis('Source', axis=1).stack()
        df.name='Count'
        intr_src = df.reset_index().groupby(['Introductions', 'Source'])['Count'].sum().unstack('Source')


        fig, axv = plt.subplots(1, intr_src.shape[0], figsize=(10,6))
        for ax, (idx, row) in zip(axv, intr_src.iterrows()):
            pie = ax.pie(row.values, explode=[0.05]*len(row), autopct='%.0f%%')
            ax.set_title(idx)
        axv[1].legend(pie[0], intr_src.columns, bbox_to_anchor=(0.0,-0.2), loc='lower center', ncol=intr_src.shape[1], frameon=True)
        cv.savefig(os.path.join(self.imgdir, f'SourcePie.png'), dpi=300)
        return fig


    def introductions_rate_by_stype(self, xvar, huevar, colvar='stype', order=2):
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
        cols = [xvar]
        if huevar is not None and huevar != 'stype':
            cols.append(huevar)
        for i, stype in enumerate(stypes):
            num = factor * self.raw.groupby(cols)[f'introductions_sum_{stype}'].sum() # without 'Replicate' in index, summing over replicates
            den = self.raw.groupby(cols)[f'susceptible_person_days_sum_{stype}'].sum()
            tmp = num/den
            tmp.name=f'Introduction rate (per {factor} person-days)'
            d.append(tmp.to_frame())
            d[i]['stype'] = stype
        D = pd.concat(d)

        g = sns.lmplot(data=D.reset_index(), col=colvar, x=xvar, y=f'Introduction rate (per {factor} person-days)', hue=huevar, height=10)
        g.set(ylim=(0,None))
        cv.savefig(os.path.join(self.imgdir, f'IntroductionRateStype.png'), dpi=300)
        return g

    def introductions_rate(self, xvar, huevar, order=2, height=10, aspect=1, ext=None):
        '''
        num = self.results.loc['introductions_es'] # Sum over school types...
        den = self.results.loc['susceptible_person_days_es']
        print('INTRODUCTIONS (ES)\n', num)
        print('SUS PERSON DAYS (ES)\n', den)
        '''

        # Just plotting the mean:
        stypes = ['es', 'ms', 'hs']
        num_cols = [f'introductions_sum_{stype}' for stype in stypes]
        den_cols = [f'susceptible_person_days_sum_{stype}' for stype in stypes]

        factor = 100_000
        cols = [xvar] if huevar is None else [xvar, huevar]
        num = factor * self.raw.groupby(cols)[num_cols].sum().sum(axis=1) # without 'Replicate' in index, summing over replicates
        den = self.raw.groupby(cols)[den_cols].sum().sum(axis=1)
        d = num/den
        d.name=f'Introduction rate (per {factor:,} person-days)'

        g = sns.lmplot(data=d.reset_index(), x=xvar, y=f'Introduction rate (per {factor:,} person-days)', hue=huevar, height=height, aspect=aspect, legend_out=False)
        g.set(ylim=(0,None))
        if xvar == 'Prevalence Target':
            for ax in g.axes.flat:
                ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        plt.tight_layout()
        fn = 'IntroductionRate.png' if ext is None else f'IntroductionRate_{ext}.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=300)
        return g


    def introductions_reg(self, xvar, huevar, order=2):
        ##### Introductions
        cols = [xvar] if huevar is None else [xvar, huevar]
        d = self.results.loc['introductions_per_100_students'].reset_index(cols)#.loc[:,[cols]+['value']]
        g = sns.lmplot(data=d, x=xvar, y='value', hue=huevar, height=10, x_estimator=np.mean, order=order, legend_out=False)
        g.set(ylim=(0,None))
        g.set_ylabels('Introductions (per 100 students over 2mo)')
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, f'IntroductionsRegression.png'), dpi=300)
        return g


    def outbreak_reg(self, xvar, huevar, order=2, height=10, aspect=1, ext=None):
        ##### Outbreak size
        cols = [xvar] if huevar is None else [xvar, huevar]
        d = self.results.loc['outbreak_size'].reset_index(cols)#[[xvar, huevar, 'value']]
        g = sns.lmplot(data=d, x=xvar, y='value', hue=huevar, height=height, aspect=aspect, x_estimator=np.mean, order=order, legend_out=False)
        g.set(ylim=(0,None))
        g.set_ylabels('Outbreak size, including source')

        fn = 'OutbreakSizeRegression.png' if ext is None else f'OutbreakSizeRegression_{ext}.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=300)
        return g

    def exports_reg(self, xvar, huevar, order=2, height=10, aspect=1, ext=None):
        ##### Outbreak size
        cols = [xvar] if huevar is None else [xvar, huevar]
        d = self.results.loc['exports_to_hh'].reset_index(cols)#[[xvar, huevar, 'value']]
        g = sns.lmplot(data=d, x=xvar, y='value', hue=huevar, height=height, aspect=aspect, x_estimator=np.mean, order=order, legend_out=False)
        g.set(ylim=(0,None))
        g.set_ylabels('Exports to HH')
        plt.tight_layout()

        fn = 'ExportsHH.png' if ext is None else f'ExportsHH_{ext}.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=300)
        return g

    def outbreak_R0(self, figsize=(8,6)):
        d = self.results_ts.loc['n_infected_by_seed'].reset_index()
        d['value'] = d['value'].astype(int)
        g = sns.catplot(data=d, x='In-school transmission multiplier', y='value', kind='bar', hue=None, height=6, aspect=1.4, palette="ch:.25")
        for ax in g.axes.flat:
            ax.axhline(y=1, color='k', lw=2, ls='--', zorder=-1)
        g.set_ylabels('Basic reproduction number in school')
        plt.tight_layout()

        fn = 'OutbreakR0.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=300)
        return g


    def timeseries(self, channel, label, normalize):
        l = []
        for sim in self.sims:
            t = sim.results['t'] #pd.to_datetime(sim.results['date'])
            y = sim.results[channel].values
            if normalize:
                y /= sim.pars['pop_size']
            d = pd.DataFrame({label: y}, index=pd.Index(data=t, name='Date'))
            huevar=None
            if 'Prevalence' in sim.tags:
                d['Prevalence Target'] = ut.p2f(sim.tags['Prevalence'])
                huevar='Prevalence Target'
            d['Scenario'] = f'{sim.tags["scen_key"]} + {sim.tags["dxscrn_key"]}'
            d['Replicate'] = sim.tags['Replicate']
            l.append( d )
        d = pd.concat(l).reset_index()

        fig, ax = plt.subplots(figsize=(16,10))
        sns.lineplot(data=d, x='Date', y=label, hue=huevar, style='Scenario', palette='cool', ax=ax, legend=False)
        # Y-axis gets messed up when I introduce horizontal lines
        #for prev in d['Prevalence Target'].unique():
        #    ax.axhline(y=prev, ls='--')
        if normalize:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        cv.savefig(os.path.join(self.imgdir, f'{label}.png'), dpi=300)
        return fig

    def plot_several_timeseries(self, config):
        for config in config:
            self.timeseries(config['channel'], config['label'], config['normalize'])


    # Tree plotting
    def plot_tree(self, tree, stats, n_days, do_show=False):
        fig, ax = plt.subplots(figsize=(16,10))
        date_range = [n_days, 0]

        #print(f'Tree {i}', sid, sim.key1, sim.key2, sim.key2)
        #for u,v,w in tree.edges.data():
            #print('\tEDGE', u,v,w)
        #print(f'N{i}', sid, sim.key1, sim.key2, sim.key2, tree.nodes.data())
        for j, (u,v) in enumerate(tree.nodes.data()):
            print('\tNODE', u,v)
            if v['type'] == 'Seed':
                continue
            recovered = n_days if np.isnan(v['date_recovered']) else v['date_recovered']
            dead = n_days if np.isnan(v['date_dead']) else v['date_dead']
            col = 'gray' if v['type'] == 'Other' else 'black'
            date_range[0] = min(date_range[0], v['date_exposed']-1)
            right = np.nanmin([recovered, dead])
            date_range[1] = max(date_range[1], right+1)
            ax.plot( [v['date_exposed'], right], [j,j], '-', marker='o', color=col)
            ax.plot( v['date_diagnosed'], j, marker='d', color='b')
            ax.plot( v['date_infectious'], j, marker='|', color='r', mew=3, ms=10)
            ax.plot( v['date_symptomatic'], j, marker='s', color='orange')
            if np.isfinite(v['date_dead']):
                ax.plot( v['date_dead'], j, marker='x', color='black',  ms=10)
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
        ax.set_yticklabels([f'{int(u) if np.isfinite(u) else -1}: {v["type"]}, age {v["age"] if "age" in v else -1}' for u,v in tree.nodes.data()])

        if do_show:
            plt.show()
        else:
            return fig
