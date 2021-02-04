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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from statsmodels.nonparametric.smoothers_lowess import lowess
import scipy
import matplotlib

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


class Analysis():

    def __init__(self, sims, imgdir):
        self.sims = sims
        self.beta0 = self.sims[0].pars['beta'] # Assume base beta is unchanged across sims

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
        stypes = {'es':'Elementary', 'ms':'Middle', 'hs':'High'}
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
            ret['first_infectious_day_at_school'] = [] # List of simulation days, e.g. 56 on which an infectious individual first was in-person, one per outbreak.
            ret['outbreak_size'] = [] # A list, each entry is the sum of num students, teachers, and staff who got infected, including the source
            ret['complete'] = [] # Boolean, 1 if the outbreak completed and 0 if it was still ongoing when the simulation ended
            ret['n_infected_by_seed'] = [] # If the outbreak was seeded, the number of secondary infectious caused by the seed
            ret['exports_to_hh'] = [] # The number of direct exports from a member of the school community to households. Exclused further spread within HHs and indirect routes to HHs e.g. via the community.
            ret['introductions'] = [] # Number of introductions, overall
            ret['susceptible_person_days'] = [] # Susceptible person-days (amongst the school population)

            for stype in stypes.values():
                ret[f'introductions_{stype}'] = [] # Introductions by school type
                ret[f'susceptible_person_days_{stype}'] = [] # Susceptible person-days (amongst the school population) by school type

            for grp in ['Student', 'Teacher', 'Staff']:
                ret[f'introduction_origin_{grp}'] = 0 # Introduction origin by group
                ret[f'observed_origin_{grp}'] = 0 # First diagnosed origin by group, intended to model the "observation" process
                ret[f'number_{grp}'] = 0 # Introduction origin by group

            if sim.results['n_exposed'][first_school_day] == 0:
                print(f'Sim has zero exposed, skipping: {ret}\n')
                continue

            # Now loop over each school and collect outbreak stats
            for sid,stats in sim.school_stats.items():
                if stats['type'] not in stypes.keys():
                    continue
                stype = stypes[stats['type']]

                # Only count outbreaks in which total infectious days at school is > 0
                outbreaks = [o for o in stats['outbreaks'] if o['Total infectious days at school']>0]

                ret['n_introductions'] += len(outbreaks)
                ret['cum_incidence'].append(100 * stats['cum_incidence'] / (stats['num']['students'] + stats['num']['teachers'] + stats['num']['staff']))
                ret['in_person_days'] += np.sum([v for v in stats['in_person'].values()])
                ret[f'introductions'].append( len(outbreaks) )
                ret[f'introductions_{stype}'].append( len(outbreaks) )
                ret[f'susceptible_person_days'].append( stats['susceptible_person_days'] )
                ret[f'susceptible_person_days_{stype}'].append( stats['susceptible_person_days'] )

                # Insert at beginning for efficiency
                ret['first_infectious_day_at_school'][0:0] = [o['First infectious day at school'] for o in outbreaks]
                ret['outbreak_size'][0:0] = [ob['Infected Students'] + ob['Infected Teachers'] + ob['Infected Staff'] for ob in outbreaks]
                ret['complete'][0:0] = [float(ob['Complete']) for ob in outbreaks]
                ret['n_infected_by_seed'][0:0] = [ob['Num school people infected by seed'] for ob in outbreaks if ob['Seeded']] # Also Num infected by seed
                ret['exports_to_hh'][0:0] = [ob['Exports to household'] for ob in outbreaks]

                grp_map = {'Student':'students', 'Teacher':'teachers', 'Staff':'staff'}
                for grp in ['Student', 'Teacher', 'Staff']:
                    ret[f'introduction_origin_{grp}'] += len([o for o in outbreaks if grp in o['Origin type']])
                    ret[f'number_{grp}'] += stats['num'][grp_map[grp]]

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

            for stype in stypes.values():
                # Sums, won't allow for full bootstrap resampling!
                ret[f'introductions_sum_{stype}'] = np.sum(ret[f'introductions_{stype}'])
                ret[f'susceptible_person_days_sum_{stype}'] = np.sum(ret[f'susceptible_person_days_{stype}'])

            results.append(ret)

        # Convert results to a dataframe
        self.raw = pd.DataFrame(results)

    def _wrangle(self, keys, outputs=None):
        # Wrangling - build self.results from self.raw
        stypes = ['Elementary', 'Middle', 'High']
        if outputs == None:
            outputs = ['introductions', 'susceptible_person_days', 'outbreak_size', 'exports_to_hh']
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

        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, f'SchoolCumInc.png'), dpi=dpi)
        return g

    def outbreak_size_over_time(self, rowvar=None, colvar=None):
        d = self.results \
            .loc[['first_infectious_day_at_school', 'outbreak_size', 'complete']] \
            .unstack('indicator')['value']

        g = sns.lmplot(data=d.reset_index(), x='first_infectious_day_at_school', y='outbreak_size', hue='complete', row=rowvar, col=colvar, scatter_kws={'s': 7}, x_jitter=True, markers='.', height=10, aspect=1)#, discrete=True, multiple='dodge')
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, f'OutbreakSizeOverTime.png'), dpi=dpi)
        return g

    def source_dow(self, figsize=(6,6)):

        # Make figure and histogram
        fig, ax = plt.subplots(1,1,figsize=figsize)
        color = '#9B6875' # One of the colors from the other bar graph
        sns.histplot(np.hstack(self.results.loc['first_infectious_day_at_school']['value']), discrete=True, stat='probability', ax=ax, color=color, edgecolor='w')

        # Basic annotations
        ax.set_xlabel('Day of week')
        ax.set_ylabel('Proportion of introductions')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xticks(np.arange(*np.round(ax.get_xlim())))
        ax.set_xticklabels(['S', 'S', 'M', 'T', 'W', 'T', 'F']*4 + ['S'])
        ax.tick_params(axis='x', which='major', labelsize=16)
        fig.tight_layout()

        # Add labels
        curved_arrow(ax=ax, x=[50, 49], y=[0.20, 0.13], style="arc3,rad=-0.3", text='First in-person day', linewidth=2)
        curved_arrow(ax=ax, x=[63, 67.5], y=[0.1, 0.05], style="arc3,rad=-0.1", text='Weekend', linewidth=2)

        # Finish up
        cv.savefig(os.path.join(self.imgdir, f'IntroductionDayOfWeek.png'), dpi=dpi)
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
        print(df.head())
        intr_src = df.reset_index().groupby(['Introductions', 'Source'])['Count'].sum().unstack('Source')

        intr_src['Kind'] = 'Overall'
        intr_src.set_index('Kind', append=True, inplace=True)

        d = intr_src.loc[('Actual Source', 'Overall')]
        intr_src = intr_src.append(pd.DataFrame({'Staff': d['Staff']/d['number_Staff'], 'Student': d['Student']/d['number_Student'], 'Teacher': d['Teacher']/d['number_Teacher'], 'number_Staff': 1, 'number_Student': 1, 'number_Teacher': 1}, index=pd.MultiIndex.from_tuples([("Actual Source", "Per-person")])))

        d = intr_src.loc[('First Diagnosed', 'Overall')]
        intr_src = intr_src.append(pd.DataFrame({'Staff': d['Staff']/d['number_Staff'], 'Student': d['Student']/d['number_Student'], 'Teacher': d['Teacher']/d['number_Teacher'], 'number_Staff': 1, 'number_Student': 1, 'number_Teacher': 1}, index=pd.MultiIndex.from_tuples([('First Diagnosed', 'Per-person')])))

        intr_src.drop(['number_Staff', 'number_Student', 'number_Teacher'], axis=1, inplace=True)

        print(intr_src)

        def pie(**kwargs):
            data = kwargs['data']
            p = plt.pie(data[['Student', 'Teacher', 'Staff']].values[0], explode=[0.05]*3, autopct='%.0f%%', normalize=True)
            #plt.legend(p, ['Student', 'Teacher', 'Staff'], bbox_to_anchor=(0.0,-0.2), loc='lower center', ncol=3, frameon=True)

        g = sns.FacetGrid(data=intr_src.reset_index(), row='Kind', row_order=['Per-person', 'Overall'], col='Introductions', margin_titles=True, despine=False, legend_out=True, height=4)
        g.map_dataframe(pie)
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, f'SourcePie.png'), dpi=dpi)
        return g


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
        fit = gp.fit(data[xvar].values.reshape(-1, 1), data[yvar])

        # Increase the length scale a bit for a smoother fit
        #p = fit.kernel_.get_params()
        #fit.kernel_.set_params(**{'k1__length_scale': 1.0*p['k1__length_scale']})

        # Make the prediction on the meshed x-axis
        xmin = data[xvar].min()
        xmax = data[xvar].max()
        x = np.linspace(xmin,xmax,50)
        y_pred, sigma = gp.predict(x.reshape(-1, 1), return_std=True)

        plt.fill_between(x, y_pred+1.96*sigma, y_pred-1.96*sigma, color=color, alpha=0.15, zorder=9)
        plt.scatter(data[xvar], data[yvar], s=4, color=color, alpha=0.05, linewidths=0.5, edgecolors='face', zorder=10)
        plt.plot(x, y_pred, color=color, zorder=11, lw=2)


    def gp_reg(self, df, xvar, huevar, height=6, aspect=1.4, legend=True, cmap='Set1', hue_order=None):
        if huevar is None:
            legend = False
        else:
            if hue_order is not None:
                hue_order = [h for h in hue_order if h in df.reset_index()[huevar].unique()]
            else:
                hue_order = df.reset_index()[huevar].unique()

        g = sns.FacetGrid(data=df.reset_index(), hue=huevar, hue_order=hue_order, height=height, aspect=aspect, palette=cmap)
        g.map_dataframe(self.gpplot, xvar=xvar, yvar='value')
        plt.grid(color='lightgray', zorder=-10)

        g.set(xlim=(0,None), ylim=(0,None))
        if xvar in ['Prevalence Target', 'Screen prob']:
            decimals = 1 if xvar == 'Prevalence Target' else 0
            for ax in g.axes.flat:
                ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=decimals))

        for ax in g.axes.flat:
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)

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
        factor = 100_000
        cols = [xvar] if huevar is None else [xvar, huevar]
        num = self.results.loc['introductions']
        den = self.results.loc['susceptible_person_days']

        # Bootstrap
        fracs = []
        for i in range(nboot):
            rows = np.random.randint(low=0, high=num.shape[0], size=num.shape[0])
            top = factor*num.iloc[rows].groupby(cols).sum()
            bot = den.iloc[rows].groupby(cols).sum()
            fracs.append(top/bot)
        df = pd.concat(fracs)

        hue_order = self.screen_order if huevar == 'Dx Screening' else None
        g = self.gp_reg(df, xvar, huevar, height, aspect, legend, hue_order=hue_order)
        for ax in g.axes.flat:
            ax.set_ylabel(f'School introduction rate per {factor:,}')

        fn = 'IntroductionRate.png' if ext is None else f'IntroductionRate_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def introductions_rate_by_stype(self, xvar, height=6, aspect=1.4, ext=None, nboot=50, legend=True, cmap='Dark2'):
        stypes = ['High', 'Middle', 'Elementary']
        factor = 100_000

        bs = []
        for idx, stype in enumerate(stypes):
            num = self.results.loc[f'introductions_{stype}']
            den = self.results.loc[f'susceptible_person_days_{stype}']

            fracs = []
            for i in range(nboot):
                rows = np.random.randint(low=0, high=num.shape[0], size=num.shape[0])
                top = factor*num.iloc[rows].groupby(xvar).sum()
                bot = den.iloc[rows].groupby(xvar).sum()
                fracs.append(top/bot)
            df = pd.concat(fracs)
            df.loc[:, 'stype'] = stype
            bs.append(df)
        bootstrap=pd.concat(bs)

        g = self.gp_reg(bootstrap, xvar, 'stype', height, aspect, legend, cmap=cmap)
        for ax in g.axes.flat:
            ax.set_ylabel(f'School introduction rate per {factor:,}')

        fn = 'IntroductionRateStype.png' if ext is None else f'IntroductionRateStype_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g


    def outbreak_reg(self, xvar, huevar, height=6, aspect=1.4, ext=None, nboot=50, legend=True):
        ##### Outbreak size
        cols = [xvar] if huevar is None else [xvar, huevar]
        ret = self.results.loc['outbreak_size']

        # Bootstrap
        resamples = []
        for i in range(nboot):
            rows = np.random.randint(low=0, high=ret.shape[0], size=ret.shape[0])
            resample_mu = ret.iloc[rows].groupby(cols).mean() # Mean estimator
            resamples.append(resample_mu)
        df = pd.concat(resamples)

        hue_order = self.screen_order if huevar == 'Dx Screening' else None
        g = self.gp_reg(df, xvar, huevar, height, aspect, legend, hue_order=hue_order)
        for ax in g.axes.flat:
            ax.set_ylabel('Outbreak size, including source')

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
        if not is_numeric:
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
            else:
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
        else:
            set_xticks = ax.set_yticks
            set_xticklabels = ax.set_yticklabels
            set_xlabel = ax.set_ylabel
            set_ylabel = ax.set_xlabel
        xt = np.linspace(x.min(), x.max(), 6)
        if is_numeric:
            set_xticks(xt)
            set_xticklabels([f'{self.beta0*betamult:.1%}' for betamult in xt])
            set_xlabel('Transmission probability in schools, per-contact per-day')
        else:
            set_xticks(indices)
            set_xticklabels(labels)

        set_ylabel('Outbreak size')

        fn = 'OutbreakSizeRegression.png' if ext is None else f'OutbreakSizeRegression_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return df


    def outbreak_size_distribution(self, row=None, row_order=None, col=None, height=12, aspect=0.7, ext=None, legend=False):
        df = self.results.loc['outbreak_size'].reset_index()
        # df['value_log'] = np.log2(df['value'])
        # xtmax = int(np.ceil(df['value_log'].max()))
        # bins = [0, 1, 2, 5, 10, 20, 50, 100, 200]
        # bins = list(sc.cat(sc.inclusiverange(0, 10, 1),
        #                    sc.inclusiverange(11, 50, 2),
        #                    sc.inclusiverange(51, 100, 5),
        #                    sc.inclusiverange(101, 200, 10)
        #                    ))
        # left_edges = bins[:-1]
        # n_edges = len(bins)-1
        # x = range(n_edges)
        # df['value_bin'] = np.array(pd.cut(df['value'], bins=bins, labels=x))

        # Remove middle column
        val = df[col].unique()[1]
        df = df[df[col] != val]

        if row == 'Dx Screening' and row_order is None:
            row_order = self.screen_order
        g = sns.catplot(data=df, x='value', y=row, order=row_order, col=col, orient='h', kind='boxen', legend=legend, height=height, aspect=aspect)
        g.set_titles(col_template='{col_name}')

        for ax in g.axes.flat:
            try:
                ax.set_title(f'{self.beta0*float(ax.get_title()):.1%}')
            except Exception as E:
                print(f'Warning: could not set title ({str(E)})')
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_xlim([0,50])
            # ax.set_xscale('log')
            # ax.set_xticks(x)
            # ax.set_xticklabels([f'{bins[b+1]}' for b in range(n_edges)])
            # ax.set_xticks(range(xtmax), minor=True)
            # ax.set_xticks(range(0,xtmax,2))
            # ax.set_xticklabels([f'{2**x:.0f}' for x in range(0,xtmax,2)])
        plt.subplots_adjust(bottom=0.05)
        plt.figtext(0.6,0.01,'Outbreak size', ha='center')

        fn = 'OutbreakSizeDistribution.png' if ext is None else f'OutbreakSizeDistribution_{ext}.png'
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, fn), dpi=300)
        plt.show()
        return g

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

    '''
    def exports_reg(self, xvar, huevar, order=2, height=10, aspect=1, ext=None):
        ##### Outbreak size
        cols = [xvar] if huevar is None else [xvar, huevar]
        d = self.results.loc['exports_to_hh'].reset_index(cols)#[[xvar, huevar, 'value']]
        g = sns.lmplot(data=d, x=xvar, y='value', hue=huevar, height=height, aspect=aspect, x_estimator=np.mean, order=order, legend_out=False)
        g.set(ylim=(0,None))
        g.set_ylabels('Exports to HH')
        plt.tight_layout()

        fn = 'ExportsHH.png' if ext is None else f'ExportsHH_{ext}.png'
        cv.savefig(os.path.join(self.imgdir, fn), dpi=dpi)
        return g
    '''

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
        g = self.gp_reg(df, xvar, huevar, height, aspect, legend, hue_order=hue_order)
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
        plt.tight_layout()
        cv.savefig(os.path.join(self.imgdir, f'{label}.png'), dpi=dpi)
        return fig

    def plot_several_timeseries(self, configs):
        for config in configs:
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
