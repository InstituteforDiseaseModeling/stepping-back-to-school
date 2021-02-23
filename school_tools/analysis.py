'''
Analyze Covasim simulation results and produce plots
'''

#%% Imports

import os
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc
import covasim as cv

import scipy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.ticker as mtick
import seaborn as sns
# import cmasher
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline

from . import scenarios as scn

import warnings
warnings.simplefilter('ignore', np.RankWarning)

# Global plotting styles
dpi = 300
font_size = 20
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style
mplt.rcParams['legend.fontsize'] = 16
mplt.rcParams['legend.title_fontsize'] = 16


__all__ = ['curved_arrow', 'loess_bound', 'Analysis']


#%% Helper functions

def curved_arrow(x, y, style=None, text='', ax=None, color='k', **kwargs):
    '''
    Draw a curved arrow with an optional text label.

    Args:

        x (list/arr): initial and final x-points (2-element list or array)
        y (list/arr): initial and final y-points (2-element list or array)
        style (str): the arrow style
        text (str): text annotation
        ax (axes): the axes instance to draw into; if none, use current
        color (str): color of the arrow
        kwargs (dict): passed to arrowprops

    **Example**::

        pl.scatter(pl.rand(10), pl.rand(10))
        curved_arrow(x=[0.13, 0.67], y=[0.6, 0.2], style="arc3,rad=-0.1", text='I am pointing down', linewidth=2)
    '''
    if ax is None:
        ax = plt.gca()
    ax.annotate(text, xy=(x[1], y[1]), xytext=(x[0], y[0]), xycoords='data', textcoords='data',
        arrowprops=dict(arrowstyle="->", connectionstyle=style, **kwargs))
    return


def loess_bound(x, y, frac=0.2, it=5, n_bootstrap=80, sample_frac=1.0, quantiles=None, npts=None):
    '''
    Fit a LOESS curve, with uncertainty.
    '''

    if quantiles is None:
        quantiles = [0.025, 0.975]
    if npts is None:
        npts = len(x)
    assert len(x) == len(y), 'Vectors must be the same length'

    yarr = np.zeros((npts, n_bootstrap))
    xvec = np.linspace(x.min(), x.max(), npts)

    for k in range(n_bootstrap):
        inds = np.random.choice(len(x), int(len(x)*sample_frac), replace=True)
        yi = y[inds]
        xi = x[inds]
        yl = lowess(yi, xi, frac=frac, it=it, return_sorted=False)
        yarr[:,k] = scipy.interpolate.interp1d(xi, yl, fill_value='extrapolate')(xvec)

    res = sc.objdict()
    res.x = xvec
    res.mean = np.nanmean(yarr, axis=1)
    res.median = np.nanmedian(yarr, axis=1)
    res.low = np.nanquantile(yarr, quantiles[0], axis=1)
    res.high = np.nanquantile(yarr, quantiles[1], axis=1)

    return res


def p2f(x):
    ''' Percentage to float '''
    return float(x.strip('%'))/100


#%% The analysis class

class Analysis:
    '''
    This class contains code to store, process, and plot the results.
    '''

    def __init__(self, sims, imgdir, save_outbreaks=True, verbose=True):
        self.sims = sims
        self.beta0 = self.sims[0].pars['beta'] # Assume base beta is unchanged across sims
        self.save_outbreaks = save_outbreaks
        self.outbreaks = None
        self.imgdir = imgdir
        self.verbose = verbose
        Path(self.imgdir).mkdir(parents=True, exist_ok=True)

        # Settings used for plotting -- various mappings between keys, labels, and indices
        self.factor = 100_000
        self.slabels = ['Elementary', 'Middle', 'High']
        self.smeta = sc.objdict() # Lesser used transformations
        self.smeta.skeys = ['es', 'ms', 'hs']
        self.smeta.keylabels = {k:v for k,v in zip(self.smeta.skeys, self.slabels)}
        self.smeta.keyinds = {k:v for v,k in enumerate(self.smeta.skeys)}
        self.smeta.indkeys = {v:k for v,k in enumerate(self.smeta.skeys)}
        self.smeta.colors = sc.odict({
            'Elementary': (0.30, 0.68, 0.26, 1.0), # Green elementary schools
            'Middle':     (0.21, 0.49, 0.74, 1.0), # Blue middle schools
            'High':       (0.89, 0.10, 0.14, 1.0), # Red high schools
            'All Types Combined': (0, 0, 0, 1), # Black for all
            })

        # Scenario settings
        sim_scenario_names = list(set([sim.tags['scen_key'] for sim in sims]))
        self.scenario_map = scn.scenario_map()
        self.scenario_order = [v[0] for k,v in self.scenario_map.items() if k in sim_scenario_names]

        sim_screen_names = list(set([sim.tags['dxscrn_key'] for sim in sims]))
        self.dxscrn_map = scn.screening_map()
        self.screen_order = [v[0] for k,v in self.dxscrn_map.items() if k in sim_screen_names]

        # Process sims -- note that only keys explicitly named here will be included after wrangling
        self._process()
        keys = list(sims[0].tags.keys()) + ['School Schedule', 'Dx Screening', 'sim_id']
        #keys +=['sim_id', 'school_id', 'school_type'] # Additional tracking keys
        keys.remove('school_start_date')
        if 'Cohort Rewiring' in keys:
            # Change Rewiring --> Mixing
            keys.remove('Cohort Rewiring')
            keys.append('Cohort Mixing')

        if 'location' in keys:
            keys.remove('location')
            keys.append('Location')
        if 'Prevalence' in sims[0].tags.keys():
            keys.append('Prevalence Target')
        self._wrangle(keys)

        return


    def _process(self):
        ''' Process the simulations '''
        if self.verbose:
            print('Processing...')
        results = []
        obdf = sc.objdict(sim=[], school=[], outbreak=[]) # Not a dataframe yet, but will be

        # Loop over each simulation and accumulate stats
        for s,sim in enumerate(self.sims):
            first_school_date = sim.tags['school_start_date']
            first_school_day = sim.day(first_school_date)

            # Map the scenario name and type of diagnostic screening, if any, to friendly names
            skey = sim.tags['scen_key']
            tkey = sim.tags['dxscrn_key']

            # Tags usually contain the relevant sweep dimensions
            ret = sc.dcp(sim.tags)

            if 'Cohort Rewiring' in ret:
                # Change Rewiring --> Mixing
                ret['Cohort Mixing'] = ret.pop('Cohort Rewiring')

            if 'location' in sim.tags:
                ret['Location'] = sim.tags['location'] # Upper case
                ret.pop('location')
            ret['School Schedule'] = self.scenario_map[skey][0] if skey in self.scenario_map else skey
            ret['Dx Screening'] = self.dxscrn_map[tkey][0] if tkey in self.dxscrn_map else tkey

            if 'Prevalence' in sim.tags:
                # Prevalence appears as a string, e.g. 1%, so convert to a float for plotting
                ret['Prevalence Target'] = p2f(sim.tags['Prevalence'])

            # Initialize a dictionary that will be later transformed into a dataframe.
            # Each simulation will get one value for each key, but not that sometimes the value is a list of values
            ret['n_introductions'] = 0 # Number of times covid was introduced, cumulative across all schools
            ret['cum_incidence'] = [] # List of timeseries of cumulative incidences, one per school
            ret['in_person_days'] = 0 # Total number of in-person days
            ret['first_infectious_day_at_school'] = [] # List of simulation days, e.g. 56 on which an infectious individual first was in-person, one per outbreak.
            ret['outbreak_size'] = [] # A list, each entry is the sum of num students, teachers, and staff who got infected, including the source
            ret['complete'] = [] # Boolean, 1 if the outbreak completed and 0 if it was still ongoing when the simulation ended
            ret['n_infected_by_seed'] = [] # If the outbreak was seeded, the number of secondary infectious caused by the seed
            ret['exports_to_hh'] = [] # The number of direct exports from a member of the school community to households. Exclused further spread within HHs and indirect routes to HHs e.g. via the community.
            ret['introductions'] = [] # Number of introductions, overall
            ret['susceptible_person_days'] = [] # Susceptible person-days (amongst the school population)
            ret['sim_id'] = s # Store the simulation ID
            ret['outbreak_sid'] = [] # School ID
            ret['outbreak_stind'] = [] # School type

            for stype in self.slabels:
                ret[f'introductions_{stype}'] = [] # Introductions by school type
                ret[f'susceptible_person_days_{stype}'] = [] # Susceptible person-days (amongst the school population) by school type
                ret[f'outbreak_size_{stype}'] = [] # Susceptible person-days (amongst the school population) by school type

            for grp in ['Student', 'Teacher', 'Staff']:
                ret[f'introduction_origin_{grp}'] = 0 # Introduction origin by group
                ret[f'observed_origin_{grp}'] = 0 # First diagnosed origin by group, intended to model the "observation" process
                ret[f'number_{grp}'] = 0 # Introduction origin by group

            if sim.results['n_exposed'][first_school_day] == 0:
                print(f'Sim has zero exposed, skipping: {ret}\n')
                continue

            # Now loop over each school and collect outbreak stats
            for sid,stats in sim.school_stats.items():
                if stats['type'] not in self.smeta.skeys:
                    continue
                stype = self.smeta.keylabels[stats['type']]
                stind = self.smeta.keyinds[stats['type']]
                #ret['school_id'].append(sid)
                #ret['school_type'].append(stype)

                # Only count outbreaks in which total infectious days at school is > 0
                outbreaks = [o for o in stats['outbreaks'] if o['Total infectious days at school']>0]

                ret['n_introductions'] += len(outbreaks)
                ret['cum_incidence'].append(100 * stats['cum_incidence'] / (stats['num']['students'] + stats['num']['teachers'] + stats['num']['staff']))
                ret['in_person_days'] += np.sum([v for v in stats['in_person'].values()])
                ret['introductions'].append( len(outbreaks) )
                ret[f'introductions_{stype}'].append( len(outbreaks) )
                ret['susceptible_person_days'].append( stats['susceptible_person_days'] )
                ret[f'susceptible_person_days_{stype}'].append( stats['susceptible_person_days'] )

                # Insert at beginning for efficiency
                ret['first_infectious_day_at_school'][0:0] = [o['First infectious day at school'] for o in outbreaks]
                ret['outbreak_size'][0:0] = [ob['Infected Students'] + ob['Infected Teachers'] + ob['Infected Staff'] for ob in outbreaks]
                ret[f'outbreak_size_{stype}'][0:0] = [ob['Infected Students'] + ob['Infected Teachers'] + ob['Infected Staff'] for ob in outbreaks]

                ret['complete'][0:0] = [float(ob['Complete']) for ob in outbreaks]
                ret['n_infected_by_seed'][0:0] = [ob['Num school people infected by seed'] for ob in outbreaks if ob['Seeded']] # Also Num infected by seed
                ret['exports_to_hh'][0:0] = [ob['Exports to household'] for ob in outbreaks]

                # These are the same for all outbreaks, but must be the right length
                ret['outbreak_sid'][0:0] = [int(sid.lstrip('s')) for ob in outbreaks] # School ID is a string, not a number!
                ret['outbreak_stind'][0:0] = [stind for ob in outbreaks]

                grp_map = {'Student':'students', 'Teacher':'teachers', 'Staff':'staff'}
                for grp in ['Student', 'Teacher', 'Staff']:
                    ret[f'introduction_origin_{grp}'] += len([o for o in outbreaks if grp in o['Origin type']])
                    ret[f'number_{grp}'] += stats['num'][grp_map[grp]]

                # Determine the type of the first "observed" case, here using the first diagnosed
                for ob in outbreaks:
                    if self.save_outbreaks:
                        obdf.sim.append(s)
                        obdf.school.append(sid)
                        obdf.outbreak.append(ob)

                    date = 'date_diagnosed' # 'date_symptomatic'
                    was_detected = [(int(u),d) for u,d in ob['Tree'].nodes.data() if np.isfinite(u) and d['type'] not in ['Other'] and np.isfinite(d[date])]
                    if len(was_detected) > 0:
                        first = sorted(was_detected, key=lambda x:x[1][date])[0]
                        detected_origin_type = first[1]['type']
                        ret[f'observed_origin_{detected_origin_type}'] += 1

            for stype in self.slabels:
                # Sums, won't allow for full bootstrap resampling!
                ret[f'introductions_sum_{stype}'] = np.sum(ret[f'introductions_{stype}'])
                ret[f'susceptible_person_days_sum_{stype}'] = np.sum(ret[f'susceptible_person_days_{stype}'])

            results.append(ret)

        # Convert results to a dataframe
        self.raw = pd.DataFrame(results)
        if self.save_outbreaks:
            self.outbreaks = pd.DataFrame(obdf)

        if self.verbose:
            print(f'Processed {len(self.raw)} results.')

        return

    def _wrangle(self, keys, outputs=None):
        if self.verbose:
            print('Wrangling...')

        # Wrangling - build self.results from self.raw
        if outputs == None:
            outputs = ['introductions', 'susceptible_person_days', 'outbreak_size', 'exports_to_hh', 'outbreak_sid', 'outbreak_stind']
            outputs += [f'introductions_{stype}' for stype in self.slabels]
            outputs += [f'outbreak_size_{stype}' for stype in self.slabels]
            outputs += [f'susceptible_person_days_{stype}' for stype in self.slabels]
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

        if self.verbose:
            print(f'Wrangled {len(self.results)} results and {len(self.results_ts)} time series.')

        return


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

        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, 'SchoolCumInc.png'), dpi=dpi)
        return g

    def outbreak_size_over_time(self, rowvar=None, colvar=None):
        d = self.results \
            .loc[['first_infectious_day_at_school', 'outbreak_size', 'complete']] \
            .unstack('indicator')['value']

        g = sns.lmplot(data=d.reset_index(), x='first_infectious_day_at_school', y='outbreak_size', hue='complete', row=rowvar, col=colvar, scatter_kws={'s': 7}, x_jitter=True, markers='.', height=10, aspect=1)#, discrete=True, multiple='dodge')
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, 'OutbreakSizeOverTime.png'), dpi=dpi)
        return g

    def source_dow(self, figsize=(6,6), ext=None, start_day=0, n_days=28):

        # Make figure and histogram
        fig, ax = plt.subplots(1,1,figsize=figsize)
        color = '#006692' # From IDM ppt theme
        sns.histplot(np.hstack(self.results.loc['first_infectious_day_at_school']['value']), discrete=True, stat='probability', ax=ax, color=color, edgecolor='w')

        # Basic annotations
        ax.set_xlabel('Day of week')
        ax.set_ylabel('Proportion of introductions')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        days = np.arange(*np.round(ax.get_xlim()))
        total_days = len(days)
        labels = (['S', 'S', 'M', 'T', 'W', 'T', 'F']*int(np.ceil(total_days/7)))[:total_days]
        ax.set_xticks(days)
        ax.set_xticklabels(labels)
        end_day = min(start_day+n_days-1, total_days-1)
        ax.set_xlim([days[start_day], days[end_day]+0.5]) # Don't show more than 4 weeks
        ax.tick_params(axis='x', which='major', labelsize=16)
        fig.tight_layout()

        # Add labels
        curved_arrow(ax=ax, x=[50, 49], y=[0.20, 0.13], style="arc3,rad=-0.3", text='First in-person day', linewidth=2)
        curved_arrow(ax=ax, x=[63, 67.5], y=[0.1, 0.05], style="arc3,rad=-0.1", text='Weekend', linewidth=2)

        # Finish up
        fn = 'IntroductionDayOfWeek.png' if ext is None else f'IntroductionDayOfWeek_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return fig

    def source_pie(self):
        groups = ['Student', 'Teacher', 'Staff']
        cols = [f'introduction_origin_{origin_type}' for origin_type in groups]
        cols += [f'number_{origin_type}' for origin_type in groups]
        intro_by_origin = self.raw[cols]
        intro_by_origin.rename(columns={f'introduction_origin_{origin_type}':origin_type for origin_type in ['Student', 'Teacher', 'Staff']}, inplace=True)
        intro_by_origin.loc[:, 'Introductions'] = 'Actual Source'

        cols = [f'observed_origin_{origin_type}' for origin_type in groups]
        cols += [f'number_{origin_type}' for origin_type in groups]
        detected_intro_by_origin = self.raw[cols]
        detected_intro_by_origin.rename(columns={f'observed_origin_{origin_type}':origin_type for origin_type in ['Student', 'Teacher', 'Staff']}, inplace=True)
        detected_intro_by_origin.loc[:, 'Introductions'] = 'First Diagnosed'

        df = pd.concat([intro_by_origin, detected_intro_by_origin], ignore_index=True)
        df = df.set_index('Introductions').rename_axis('Source', axis=1).stack()
        df.name='Count'
        intr_src = df.reset_index().groupby(['Introductions', 'Source'])['Count'].sum().unstack('Source')

        intr_src['Kind'] = 'Per-school'
        intr_src.set_index('Kind', append=True, inplace=True)

        d = intr_src.loc[('Actual Source', 'Per-school')]
        intr_src = intr_src.append(pd.DataFrame({'Staff': d['Staff']/d['number_Staff'], 'Student': d['Student']/d['number_Student'], 'Teacher': d['Teacher']/d['number_Teacher'], 'number_Staff': 1, 'number_Student': 1, 'number_Teacher': 1}, index=pd.MultiIndex.from_tuples([("Actual Source", "Per-person")])))

        d = intr_src.loc[('First Diagnosed', 'Per-school')]
        intr_src = intr_src.append(pd.DataFrame({'Staff': d['Staff']/d['number_Staff'], 'Student': d['Student']/d['number_Student'], 'Teacher': d['Teacher']/d['number_Teacher'], 'number_Staff': 1, 'number_Student': 1, 'number_Teacher': 1}, index=pd.MultiIndex.from_tuples([('First Diagnosed', 'Per-person')])))

        intr_src.drop(['number_Staff', 'number_Student', 'number_Teacher'], axis=1, inplace=True)

        def pie(**kwargs):
            data = kwargs['data']
            plt.pie(data[['Student', 'Teacher', 'Staff']].values[0], explode=[0.05]*3, autopct='%.0f%%', normalize=True)
            #plt.legend(p, ['Student', 'Teacher', 'Staff'], bbox_to_anchor=(0.0,-0.2), loc='lower center', ncol=3, frameon=True)

        g = sns.FacetGrid(data=intr_src.reset_index(), row='Kind', row_order=['Per-person', 'Per-school'], col='Introductions', margin_titles=True, despine=False, legend_out=True, height=4)
        g.map_dataframe(pie)
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        # Add legend
        lbls = ['Student', 'Teacher', 'Staff']
        colors = sns.color_palette('tab10').as_hex()[:len(lbls)]
        h = [matplotlib.patches.Patch(color=col, label=lab) for col, lab in zip(colors, lbls)]
        plt.figlegend(handles=h, ncol=len(h), loc='lower center', frameon=False)

        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, 'SourcePie.png'), dpi=dpi)
        return g


    @staticmethod
    def splineplot(**kwargs):
        data = kwargs['data']
        color = kwargs['color']
        xvar = kwargs['xvar']
        yvar = kwargs['yvar']

        mu = data.groupby(xvar).mean().reset_index()
        std = data.groupby(xvar).std().reset_index()
        k = min(3, len(mu[xvar])-1) # Don't try to use more points in the spline than there are data points -- NB, will fail for 1 data point
        mu_spl = UnivariateSpline(mu[xvar], mu[yvar], s=0.1*mu.shape[0], k=k)
        std_spl = UnivariateSpline(std[xvar], std[yvar], s=0.1*std.shape[0], k=k)
        xmin = mu.iloc[0][xvar]
        xmax = mu.iloc[-1][xvar]
        xs = np.linspace(xmin,xmax,50)
        y_pred = mu_spl(xs)
        sigma = std_spl(xs)
        plt.scatter(mu[xvar], mu[yvar], s=40, color=color, alpha=0.8, linewidths=0.5, edgecolors='face', zorder=10)

        #plt.scatter(data[xvar], data[yvar], s=4, color=color, alpha=0.05, linewidths=0.5, edgecolors='face', zorder=10)
        data_range = data[xvar].max() - data[xvar].min()
        nx = data[xvar].nunique()
        noise = 0.2 * data_range / nx / 2
        data.loc[:,f'{xvar}_scatter'] = data[xvar] + np.random.uniform(low=-noise, high=noise, size=data.shape[0])
        plt.scatter(data[f'{xvar}_scatter'], data[yvar], s=8, color=color, alpha=0.2, linewidths=0.5, edgecolors='face', zorder=10)

        label = kwargs['label'] if 'label' in kwargs else None
        plt.plot(xs, y_pred, color=color, lw=3, label=label)
        plt.fill_between(xs, y_pred+1.96*sigma, y_pred-1.96*sigma, color=color, alpha=0.15, zorder=9)


    @staticmethod
    def gpplot(**kwargs):
        data = kwargs['data']
        color = kwargs['color']
        xvar = kwargs['xvar']
        yvar = kwargs['yvar']

        # Instantiate a Gaussian Process model
        kernel = Matern(length_scale=0.1, nu=1.5) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, normalize_y=True)

        # Fit to data using Maximum Likelihood Estimation of the parameters
        gp.fit(data[xvar].values.reshape(-1, 1), data[yvar])

        # Make the prediction on the meshed x-axis
        xmin = data[xvar].min()
        xmax = data[xvar].max()
        x = np.linspace(xmin,xmax,50)
        y_pred, sigma = gp.predict(x.reshape(-1, 1), return_std=True)

        data_range = data[xvar].max() - data[xvar].min()
        nx = data[xvar].nunique()
        noise = 0.1 * data_range / nx / 2
        data.loc[:,f'{xvar}_scatter'] = data[xvar] + np.random.uniform(low=-noise, high=noise, size=data.shape[0])

        #mu = data.groupby(xvar)[yvar].mean().reset_index()
        #std = data.groupby(xvar)[yvar].std().reset_index()

        plt.fill_between(x, y_pred+1.96*sigma, y_pred-1.96*sigma, color=color, alpha=0.15, zorder=9)
        plt.scatter(data[f'{xvar}_scatter'], data[yvar], s=4, color=color, alpha=0.05, linewidths=0.5, edgecolors='face', zorder=10)
        #plt.plot([mu[xvar], mu[xvar]], [mu[yvar]-1.96*std[yvar], mu[yvar]+1.96*std[yvar]], color=color)
        #plt.plot(mu[xvar], mu[yvar], 'o', color=color)
        plt.plot(x, y_pred, color=color, zorder=11, lw=2)


    def gp_reg(self, df, xvar, huevar, hue_order=None, colvar=None, col_order=None, height=6, aspect=1.4, legend=True, cmap='Set1', use_spline=True):
        if huevar is None:
            legend = False
        else:
            if hue_order is not None:
                hue_order = [h for h in hue_order if h in df.reset_index()[huevar].unique()]
            else:
                hue_order = df.reset_index()[huevar].unique()


        g = sns.FacetGrid(data=df.reset_index(), hue=huevar, hue_order=hue_order, col=colvar, col_order=col_order, height=height, aspect=aspect, palette=cmap)
        if use_spline:
            g.map_dataframe(self.splineplot, xvar=xvar, yvar='value') # Switched from gpplot to splineplot to better capture variance trends
        else:
            g.map_dataframe(self.gpplot, xvar=xvar, yvar='value') # Previous implementation

        g.set(xlim=(0,None), ylim=(0,None))
        if xvar in ['Prevalence Target', 'Screen prob']:
            decimals = 1 if xvar == 'Prevalence Target' else 0
            for ax in g.axes.flat:
                ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimals))

        for ax in g.axes.flat:
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.grid(color='lightgray', zorder=-10)

        g.set_xlabels(xvar)
        if xvar == 'Prevalence Target':
            g.set_xlabels('Prevalence of COVID-19 in the community')
        elif xvar == 'Screen prob':
            g.set_xlabels('Daily probability of symptom screening')

        if legend and len(hue_order)>1:
            #g.add_legend() # Ugh, not working due to seaborn bug
            if huevar in df.reset_index():
                title = huevar
                if huevar=='Prevalence Target':
                    title = 'Prevalence'
                    hue_order = [f'{p:.1%}' for p in hue_order]
                elif huevar=='stype':
                    title = 'School Type'
                elif huevar=='Dx Screening':
                    title = 'Diagnostic Screening'
                elif huevar=='In-school transmission multiplier':
                    title = 'Transmission probability'
                    hue_order = [f'{self.beta0*betamult:.1%}' for betamult in hue_order]

                colors = sns.color_palette(cmap).as_hex()[:len(hue_order)] # 'deep'
                #h = [patches.Patch(color=col, label=lab) for col, lab in zip(colors, hue_order)]
                h = [plt.plot(0,0,color=col, label=lab) for col, lab in zip(colors, hue_order)]
                h = [z[0] for z in h]

                plt.legend(handles=h, title=title)

        return g


    def introductions_rate(self, xvar, huevar, height=6, aspect=1.4, ext=None, nboot=50, legend=True):
        cols = [xvar] if huevar is None else [xvar, huevar]
        num = self.results.loc['introductions']
        den = self.results.loc['susceptible_person_days']

        # Bootstrap
        fracs = []
        for i in range(nboot):
            rows = np.random.randint(low=0, high=num.shape[0], size=num.shape[0])
            top = self.factor*num.iloc[rows].groupby(cols).sum()
            bot = den.iloc[rows].groupby(cols).sum()
            fracs.append(top/bot)
        df = pd.concat(fracs)

        hue_order = self.screen_order if huevar == 'Dx Screening' else None
        g = self.gp_reg(df=df, xvar=xvar, huevar=huevar, height=height, aspect=aspect, legend=legend, hue_order=hue_order)

        g.set(ylim=(0,80)) # Consistency across figures
        if xvar == 'Screen prob':
            g.set(xlim=(0,1)) # Consistency across figures
        for ax in g.axes.flat:
            ax.set_ylabel(f'School introduction rate per {self.factor:,}')

        fn = 'IntroductionRate.png' if ext is None else f'IntroductionRate_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def introductions_rate_by_stype(self, xvar, height=6, aspect=1.4, ext=None, nboot=50, legend=True, cmap='Set1'):

        bs = []
        for idx, stype in enumerate(['All Types Combined'] + self.slabels):
            if stype == 'All Types Combined':
                num = pd.concat([self.results.loc[f'introductions_{st}'] for st in self.slabels])
                den = pd.concat([self.results.loc[f'susceptible_person_days_{st}'] for st in self.slabels])

                # Calculate slope
                frac = 100_000*num/den
                frac.reset_index('Prevalence Target', inplace=True)
                mod = sm.OLS(frac['value'], sm.add_constant(1_000 * frac['Prevalence Target']))
                res = mod.fit()
                print(res.summary())
                print(res.params)
            else:
                num = self.results.loc[f'introductions_{stype}']
                den = self.results.loc[f'susceptible_person_days_{stype}']

            fracs = []
            for i in range(nboot):
                rows = np.random.randint(low=0, high=num.shape[0], size=num.shape[0])
                top = self.factor*num.iloc[rows].groupby(xvar).sum()
                bot = den.iloc[rows].groupby(xvar).sum()
                fracs.append(top/bot)
            df = pd.concat(fracs)
            df.loc[:, 'stype'] = stype
            bs.append(df)
        bootstrap=pd.concat(bs)

        g = self.gp_reg(bootstrap, xvar=xvar, huevar='stype', height=height, aspect=aspect, legend=legend, cmap=cmap)
        g.set(ylim=(0,80)) # Consistency across figures
        colors = matplotlib.cm.get_cmap(cmap)
        for ax in g.axes.flat:
            ax.set_ylabel(f'School introduction rate per {self.factor:,}')
            ax.text(0.01,5, f'Slope: {res.params["Prevalence Target"]:.1f} per 0.1% increase\n in community prevalence', color=colors(0))

        fn = 'IntroductionRateStype.png' if ext is None else f'IntroductionRateStype_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def outbreak_reg_facet(self, xvar, huevar, hue_order=None, colvar=None, col_order=None, height=6, aspect=1.4, ext=None, nboot=50, legend=True):
        cols = [xvar] if huevar is None else [xvar, huevar]

        if colvar == 'School Type':
            types = self.slabels
            cols += ['School Type']
        else:
            types = ['All Types Combined']

        dfs = []
        for idx, stype in enumerate(types):
            if stype == 'All Types Combined':
                df = self.results.loc['outbreak_size'].reset_index()##.rename({'value':'Outbreak Size'}, axis=1)
                #xlim = (dfs[xvar].min(), dfs[xvar].max())
            else:
                df = self.results.loc[f'outbreak_size_{stype}'].reset_index()##.rename({'value':'Outbreak Size'}, axis=1)
            df['School Type'] = stype
            dfs.append( df )

        ret = pd.concat(dfs)

        # Bootstrap
        resamples = []
        for i in range(nboot):
            rows = np.random.randint(low=0, high=ret.shape[0], size=ret.shape[0])
            resample_mu = ret.iloc[rows].groupby(cols).mean() # Mean estimator
            resamples.append(resample_mu)
        df = pd.concat(resamples)

        hue_order = self.screen_order if huevar == 'Dx Screening' else hue_order
        g = self.gp_reg(df=df, xvar=xvar, huevar=huevar, hue_order=hue_order, colvar=colvar, col_order=col_order, height=height, aspect=aspect, legend=legend)
        g.set(ylim=(0,None))
        for i, ax in enumerate(g.axes.flat):
            if i == 0:
                ax.set_ylabel('Outbreak size, including source')
            ax.axhline(y=1, ls='--', color='k')

            if xvar == 'In-school transmission multiplier':
                xlim = ax.get_xlim()
                xt = np.linspace(xlim[0], xlim[1], 5)
                ax.set_xticks( xt )
                ax.set_xticklabels( [f'{self.beta0*betamult:.1%}' for betamult in xt] )
                ax.set_xlabel('Transmission probability in schools, per-contact per-day')

        fn = 'OutbreakSizeRegression.png' if ext is None else f'OutbreakSizeRegression_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def outbreak_reg(self, xvar, height=6, aspect=1.4, ext=None, nboot=50, legend=True, ax=None, by_stype=False):
        do_save = False
        if ax == None:
            fig, ax = plt.subplots(figsize=(height*aspect, height))
            do_save=True
        plt.sca(ax)
        types = ['All Types Combined']
        if by_stype:
            types += self.slabels
            fn = 'OutbreakSizeStype.png' if ext is None else f'OutbreakSizeStype_{ext}.png'
        else:
            fn = 'OutbreakSize.png' if ext is None else f'OutbreakSize_{ext}.png'

        for idx, stype in enumerate(types):
            if stype == 'All Types Combined':
                dfs = self.results.loc['outbreak_size'].reset_index().rename({'value':'Outbreak Size'}, axis=1)
                xlim = (dfs[xvar].min(), dfs[xvar].max())
            else:
                dfs = self.results.loc[f'outbreak_size_{stype}'].reset_index().rename({'value':'Outbreak Size'}, axis=1)

            # Bootstrap
            frames = []
            cols = [xvar]
            for i in range(nboot):
                rows = np.random.randint(low=0, high=dfs.shape[0], size=dfs.shape[0])
                frame = dfs.iloc[rows].groupby(cols)['Outbreak Size'].mean()
                frames.append(frame)
            bs = pd.concat(frames)
            color = self.smeta.colors[stype]
            self.splineplot(data=bs.reset_index(), xvar=xvar, yvar='Outbreak Size', color=color, label=stype)

        if legend:
            ax.legend(title='School Type')

        ax.set_ylabel('Outbreak size, including source')
        ax.set_ylim(0,None)
        ax.axhline(y=1, ls='--', color='k')

        if xvar == 'In-school transmission multiplier':
            xlim = ax.set_xlim(xlim)
            xt = np.linspace(xlim[0], xlim[1], 5)
            ax.set_xticks( xt )
            ax.set_xticklabels( [f'{self.beta0*betamult:.1%}' for betamult in xt] )
            ax.set_xlabel('Transmission probability in schools, per-contact per-day')

        plt.tight_layout()
        if do_save:
            cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return ax



    def outbreak_size_distribution(self, xvar, ext=None, height=10, aspect=0.7, jitter=0.125, values=None, legend=False):
        df = self.results.loc['outbreak_size'].reset_index().rename({'value':'Outbreak Size'}, axis=1)
        if values is not None:
            df = df.loc[df[xvar].isin(values)]
        else:
            values = df[xvar].unique()

        if pd.api.types.is_numeric_dtype(df[xvar]):
            df['x_jittered'] = df[xvar] + np.random.normal(scale=jitter, size=df.shape[0])
            cat=False
        else:
            df['x_jittered'] = pd.Categorical(df[xvar]).codes + np.random.normal(scale=jitter, size=df.shape[0])
            cat=True

        g = sns.relplot(data=df, x='x_jittered', y='Outbreak Size', size='Outbreak Size', hue='Outbreak Size', sizes=(4, 1000), palette='copper', height=height, aspect=aspect, alpha=0.7, legend=legend, edgecolor='k', zorder=10)

        for ax in g.axes.flat:
            if cat:
                ax.set_xticks(range(len(values)))
                g.set_xticklabels(values)#, rotation=45)
            res = loess_bound(df[xvar], df['Outbreak Size'], frac=0.5)
            ax2 = ax.twinx()
            #ax2.fill_between(res.x, res.low, res.high, alpha=0.3, zorder=-100)
            ax2.plot(res.x, res.mean, lw=3, zorder=20)
            ax2.set_ylim(1,None)
        g.set(ylim=(0,None))
        if g._legend is not None:
            g._legend.set(frame_on=1)
        g.set_xlabels(xvar)
        plt.tight_layout()

        fn = 'OutbreakSizeDistribution.png' if ext is None else f'OutbreakSizeDistribution_{ext}.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def outbreak_multipanel(self, xvar, ext=None, height=10, aspect=0.7, jitter=0.125, values=None, legend=False, use_spline=True, by_stype=True):
        df = self.results.loc['outbreak_size'].reset_index().rename({'value':'Outbreak Size'}, axis=1)
        df['outbreak_stind'] = self.results.loc['outbreak_stind'].reset_index()['value'] # CK: Must be a better way
        if values is not None:
            df = df.loc[df[xvar].isin(values)]
        else:
            values = df[xvar].unique()

        if pd.api.types.is_numeric_dtype(df[xvar]):
            df['x_jittered'] = df[xvar] + np.random.uniform(low=-jitter/2, high=jitter/2, size=df.shape[0])
        else:
            df['x_jittered'] = pd.Categorical(df[xvar]).codes + np.random.uniform(low=-jitter/2, high=jitter/2, size=df.shape[0])

        fig, axv = plt.subplots(4,1, figsize=(height*aspect, height), sharex=False)

        xt = df[xvar].unique()

        # Panel 0
        if use_spline:
            self.outbreak_reg(xvar, nboot=50, legend=True, ax=axv[0], by_stype=True)

        else:
            sns.regplot(data=df, x=xvar, y='Outbreak Size', scatter=False, order=4, ax=axv[0])

        axv[0].set_xticks(xt)
        axv[0].set_xticklabels([])
        axv[0].set_xlabel('')

        axv[0].axhline(y=1, ls='--', color='k')
        axv[0].set_ylabel('Average outbreak size')

        # Panel 1
        if by_stype:
            palette = [self.smeta.colors[:][i] for i in range(len(self.slabels))]
            hue = 'outbreak_stind'
            for c,label in enumerate(self.slabels):
                axv[1].scatter([np.nan], [np.nan], s=100, c=[palette[c]], label=label)
        else:
            palette = 'rocket'
            hue = 'Outbreak Size'
        sns.scatterplot(data=df, x='x_jittered', y='Outbreak Size', size='Outbreak Size', hue=hue, sizes=(10, 250), palette=palette, alpha=0.6, legend=legend, ax=axv[1])

        axv[1].set_xticks(xt)
        axv[1].set_xticklabels([])
        axv[1].set_xlabel('')
        axv[1].axhline(y=1, color='k', ls='--')

        axv[1].set_ylabel('Individual outbreak size')
        axv[1].legend()

        # Panel 2
        self.outbreak_size_distrib(xvar, ax=axv[2])
        axv[2].set_xticks(xt)
        axv[2].set_xticklabels([])
        axv[2].set_xlabel('')

        # Panel 3
        '''
        d = self.results_ts.loc['n_infected_by_seed'].reset_index()
        d['value'] = d['value'].astype(int)
        xv = d['In-school transmission multiplier'].unique()
        sns.barplot(data=d, x='In-school transmission multiplier', y='value', palette='crest_r', zorder=10, ax=axv[3]) # ch:.25 magma
        for l in axv[3].lines: # Move the error bars in front of the bars
            l.set_zorder(20)
        axv[3].axhline(y=1, color='k', lw=2, ls='--', zorder=-1)

        axv[3].set_xticklabels( [f'{self.beta0*betamult:.1%}' for betamult in xt] )
        axv[3].set_xlabel('Transmission probability in schools, per-contact per-day')
        axv[3].grid(color='lightgray', axis='y', zorder=-10)
        sns.despine(right=False, top=False) # Add spines back

        axv[3].set_ylabel(r'Basic reproduction ($R_{0,s}$)')
        '''
        # Alternate panel 3:
        d = self.results_ts.loc['n_infected_by_seed'].reset_index()
        d['value'] = d['value'].astype(int)
        sns.regplot(data=d, x=xvar, y='value', scatter=False, order=4, ax=axv[3])
        axv[3].set_ylabel(r'Basic reproduction ($R_{0,s}$)')
        axv[3].axhline(y=1, color='k', lw=2, ls='--', zorder=-1)
        axv[3].grid(color='lightgray', axis='y', zorder=-10)

        for i in range(4):
            axv[i].set_ylim(0,None)
            axv[i].set_xlim(axv[3].get_xlim()) # At least make these two line up
            axv[i].set_xticks(xt)
            if i == 3:
                axv[i].set_xticklabels( [f'{self.beta0*betamult:.1%}' for betamult in xt] )
                axv[i].set_xlabel('Transmission probability in schools, per-contact per-day')
        axv[1].set_xlim(df['x_jittered'].min()-0.02, df['x_jittered'].max()+0.02)

        plt.tight_layout()

        fn = 'OutbreakMultiPanel.png' if ext is None else f'OutbreakMultiPanel_{ext}.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return fig



    def outbreak_size_plot_v2(self, xvar, rowvar=None, ext=None, height=6, aspect=1.4, scatter=True, jitter=0.012):
        df = self.results.loc['outbreak_size'].reset_index().rename({'value':'Outbreak Size'}, axis=1)
        if pd.api.types.is_numeric_dtype(df[xvar]):
            #df['x_jittered'] = df[xvar] + np.random.normal(scale=jitter, size=df.shape[0])
            df['x_jittered'] = df[xvar] + np.random.uniform(low=-jitter/2, high=jitter/2, size=df.shape[0])
            cat=False
        else:
            #df['x_jittered'] = pd.Categorical(df[xvar]).codes + np.random.normal(scale=jitter, size=df.shape[0])
            so = self.screen_order.copy() # Axis goes wrong way
            so.reverse()
            cats = [o for o in so if o in df[xvar].unique()]
            df['x_jittered'] = pd.Categorical(df[xvar], categories=so).codes + np.random.uniform(low=-jitter/2, high=jitter/2, size=df.shape[0])
            cat=True

        if rowvar == 'In-school transmission multiplier':
            df.loc[:,'Transmission probability in schools'] = self.beta0*df[rowvar]
            rowvar = 'Transmission probability in schools'

        row_order = np.flip(df[rowvar].unique())

        g = sns.FacetGrid(data=df, row=rowvar, row_order=row_order, height=height, aspect=aspect, sharex=False)
        g.map_dataframe(sns.scatterplot, y='x_jittered', x='Outbreak Size', size='Outbreak Size', hue='Outbreak Size', sizes=(1, 750), palette='rocket', alpha=0.7, linewidths=2, edgecolors='black', vmax=500)
        g.map_dataframe(sns.scatterplot, y='x_jittered', x='Outbreak Size', size='Outbreak Size', sizes=(1, 750), color='black', linewidths=2, alpha=0, edgecolors='black', vmax=500)
        #g.map_dataframe(sns.boxenplot, y=xvar, x='Outbreak Size', palette='rocket')

        g.set_xlabels('Individual outbreak size')
        for ax, v in zip(g.axes.flat, row_order):
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cats)
            ax.set_title(f'Transmission probability in schools: {v:.1%}', fontsize=22)
            #ax.invert_yaxis() # Not working
        plt.tight_layout()

        # axv[1].set_xticks(xt)
        # axv[1].set_xticklabels([])
        # axv[1].set_xlabel('')
        # axv[1].set_ylim(1,None)
        # yt = axv[1].get_yticks()
        # yt[0] = 1
        # axv[1].set_yticks(yt)

        # axv[1].set_ylabel('Individual outbreak size')

        fn = 'OutbreakSizePlot.png' if ext is None else f'OutbreakSizePlot_{ext}.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def outbreak_size_distrib(self, xvar, rowvar=None, ext=None, height=6, aspect=1.4, ax=None):
        df = self.results.loc['outbreak_size'].reset_index().rename({'value':'Outbreak Size Float'}, axis=1)

        lbls = ['Source only', '2-3', '4-9', '10-24', '25+']
        df['Outbreak Size'] = pd.cut(df['Outbreak Size Float'], bins=[1,2,4,10,25,10000], right=False, include_lowest=True, labels=lbls)

        sz = df.groupby([xvar, 'Outbreak Size']).size()
        sz.name='Count'
        sz = sz.unstack('Outbreak Size')
        sz = sz.div(sz.sum(axis=1), axis=0)

        do_save = False
        if ax == None:
            fig, ax = plt.subplots(figsize=(height*aspect, height))
            do_save = True

        # colors = sns.color_palette('Pastel1').as_hex() # Set2_r skips first, using Set2_r as colormap skips every other.
        # cols = {k:colors[8-i] for i,k in enumerate(lbls)}
        # cols = sc.vectocolor(len(lbls)+3, cmap='cmr.flamingo_r')[1:-2]
        cols = sc.vectocolor(len(lbls)+1, cmap='Reds')[1:]
        sz.plot(stacked=True, kind='area', ax=ax, color=cols) # colormap='Reds') # colormap = coolwarm, parula, inferno, RdYlBu_r # Set2_r

        ax.set_xlim(sz.index[0], sz.index[-1])
        if xvar == 'In-school transmission multiplier':
            xlim = ax.get_xlim()
            xt = np.linspace(xlim[0], xlim[-1], 5)
            ax.set_xticks( xt )
            ax.set_xticklabels( [f'{self.beta0*betamult:.1%}' for betamult in xt] )
            ax.set_xlabel('Transmission probability in schools, per-contact per-day')

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0,1)
        ax.set_ylabel('Percent of outbreaks')
        plt.tight_layout()

        if do_save:
            fn = 'OutbreakSizeStacked.png' if ext is None else f'OutbreakSizeStacked_{ext}.png'
            cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return ax


    def outbreak_size_plot(self, xvar, ext=None, height=6, aspect=1.4, scatter=True, loess=True, landscape=True, jitter=0.012):
        '''
        Plot outbreak sizes in various ways.

        Args:
            xvar (str): the variable to use as the x-axis
            ext (str): suffix for filename
            height (float): figure height
            aspect (float): figure aspect ratio
            scatter (bool): show scatter of points
            loess (bool): show loess fit to points
            landscape (bool): flip orientation so x-axis is y-axis
            jitter (float): amount of scatter to add to point locations
        '''

        # Get x and y coordinates of all outbreaks
        df = self.results.reset_index() # Un-melt (congeal?) results
        df = df[df['indicator'] == 'outbreak_size'] # Find rows with outbreak size data
        x = df[xvar].values # Pull out x values
        y = df['value'].values # Pull out y values (i.e., outbreak size)

        # Handle non-numeric x axes
        is_numeric = df[xvar].dtype != 'O' # It's an object, i.e. string
        if not is_numeric: # pragma: no cover
            labels = df[xvar].unique()
            indices = range(len(labels))[::-1]
            labelmap = dict(zip(labels, indices))
            x = np.array([labelmap[i] for i in x]) # Convert from strings to indices

        plt.figure(figsize=(height*aspect, height))

        # Scatter plots
        if scatter:
            dx = x.max() - x.min()
            noise = dx*jitter*(1+np.random.randn(len(x)))
            xjitter = x + noise
            colors = sc.vectocolor(np.sqrt(y), cmap='copper')
            if landscape:
                plt_x = xjitter
                plt_y = y
                lim_func = plt.ylim
            else: # pragma: no cover
                plt_x = y
                plt_y = xjitter
                lim_func = plt.xlim

            plt.scatter(plt_x, plt_y, alpha=0.7, s=800*y/y.max(), c=colors)
            lim_func([-2, y.max()*1.1])

        # Loess plots
        if loess:
            if not landscape: raise NotImplementedError
            res = loess_bound(x, y, frac=0.5)
            plt.fill_between(res.x, res.low, res.high, alpha=0.3)
            plt.plot(res.x, res.mean, lw=3)

        # General settings
        ax = plt.gca()
        if landscape:
            set_xticks = ax.set_xticks
            set_xticklabels = ax.set_xticklabels
            set_xlabel = ax.set_xlabel
            set_ylabel = ax.set_ylabel
        else: # pragma: no cover
            set_xticks = ax.set_yticks
            set_xticklabels = ax.set_yticklabels
            set_xlabel = ax.set_ylabel
            set_ylabel = ax.set_xlabel
        xt = np.linspace(x.min(), x.max(), 6)
        if is_numeric:
            set_xticks(xt)
            set_xticklabels([f'{self.beta0*betamult:.1%}' for betamult in xt])
            set_xlabel('Transmission probability in schools, per-contact per-day')
        else: # pragma: no cover
            set_xticks(indices)
            set_xticklabels(labels)

        set_ylabel('Outbreak size')

        fn = 'OutbreakSizePlot.png' if ext is None else f'OutbreakSizePlot_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return df


    def outbreak_R0(self, figsize=(6*1.4,6)):
        d = self.results_ts.loc['n_infected_by_seed'].reset_index()
        d['value'] = d['value'].astype(int)
        xv = d['In-school transmission multiplier'].unique()
        g = sns.catplot(data=d, x='In-school transmission multiplier', y='value', kind='bar', hue=None, height=6, aspect=1.4, palette="ch:.25", zorder=10)
        for ax in g.axes.flat:
            for l in ax.lines: # Move the error bars in front of the bars
                l.set_zorder(20)
            ax.axhline(y=1, color='k', lw=2, ls='--', zorder=-1)

            xt = ax.get_xticks()
            b = xv[0]
            m = (xv[1]-xv[0]) / (xt[1]-xt[0])
            ax.set_xticklabels( [f'{m*self.beta0*betamult + b:.1%}' for betamult in xt] )
            ax.set_xlabel('Transmission probability in schools, per-contact per-day')
            ax.grid(color='lightgray', axis='y', zorder=-10)
            sns.despine(right=False, top=False) # Add spines back

        g.set_ylabels('Basic reproduction number in school')
        plt.tight_layout()

        fn = 'OutbreakR0.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def exports_reg(self, xvar, huevar, height=6, aspect=1.4, ext=None, nboot=50, legend=True):
        ##### Outbreak size
        cols = [xvar] if huevar is None else [xvar, huevar]
        ret = self.results.loc['exports_to_hh']

        # Bootstrap
        resamples = []
        for i in range(nboot):
            rows = np.random.randint(low=0, high=ret.shape[0], size=ret.shape[0])
            resample_mu = ret.iloc[rows].groupby(cols).mean() # Mean estimator
            resamples.append(resample_mu)
        df = pd.concat(resamples)

        hue_order = self.screen_order if huevar == 'Dx Screening' else None
        g = self.gp_reg(df=df, xvar=xvar, huevar=huevar, hue_order=hue_order, height=height, aspect=aspect, legend=legend)
        for ax in g.axes.flat:
            ax.set_ylabel('Number of exports to households')

        if xvar == 'In-school transmission multiplier':
            xlim = ax.get_xlim()
            xt = np.linspace(xlim[0], xlim[1], 5)
            ax.set_xticks( xt )
            ax.set_xticklabels( [f'{self.beta0*betamult:.1%}' for betamult in xt] )
            ax.set_xlabel('Transmission probability in schools, per-contact per-day')

        fn = 'ExportsHH.png' if ext is None else f'ExportsHH_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
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
                d['Prevalence Target'] = p2f(sim.tags['Prevalence'])
                huevar='Prevalence Target'
            d['School Schedule'] = f'{sim.tags["scen_key"]} + {sim.tags["dxscrn_key"]}'
            d['Replicate'] = sim.tags['Replicate']
            l.append( d )
        d = pd.concat(l).reset_index()

        fig, ax = plt.subplots(figsize=(16,10))
        sns.lineplot(data=d, x='Date', y=label, hue=huevar, style='School Schedule', palette='cool', ax=ax, legend=False)
        # Y-axis gets messed up when I introduce horizontal lines
        #for prev in d['Prevalence Target'].unique():
        #    ax.axhline(y=prev, ls='--')
        if normalize:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, f'{label}.png'), dpi=dpi)
        return fig


    def plot_several_timeseries(self, configs):
        for config in configs:
            self.timeseries(config['channel'], config['label'], config['normalize'])


    def plot_tree(self, outbreak_ind=None, tree=None, stats=None, n_days=None, do_show=False, verbose=False):
        '''
        Plots an infection tree. You can either supply the index of the outbreak,
        or the tree, stats, and n_days directly.
        '''
        fig, ax = plt.subplots(figsize=(16,10))

        # Get the right tree and stats objects
        if tree is None or stats is None:
            if outbreak_ind is None:
                outbreak_ind = 0

            row = self.outbreaks.iloc[outbreak_ind]
            sim_ind = row['sim']
            sim = self.sims[sim_ind]
            school_ind = row['school']
            stats = sim.school_stats[school_ind]
            tree = row['outbreak']['Tree']
            if n_days is None:
                n_days = sim['n_days']

        date_range = [n_days, 0]

        for j, (u,v) in enumerate(tree.nodes.data()):
            if verbose: print('\tNODE', u,v)
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
