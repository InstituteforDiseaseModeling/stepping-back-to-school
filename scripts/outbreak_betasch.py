'''
Outbreak analysis to sweep in-school transmissibility
'''

import sys
import numpy as np
import school_tools as sct

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import covasim as cv

if __name__ == '__main__':

    # Settings

    args = sct.config.process_inputs(sys.argv)

    sweep_pars = {
        'n_prev': 0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = None

    # Create and run
    mgr = sct.Manager(name='OutbreakBetaSchool', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    g = analyzer.outbreak_size_stacked_distrib(xvar, rowvar=None, ext=None, height=6, aspect=2)

    # Plots
    g = analyzer.outbreak_multipanel(xvar, ext=None, jitter=0.2, values=None, legend=False, height=12, aspect=1.0) # height=10, aspect=0.7,

    analyzer.exports_reg(xvar, huevar)
    analyzer.outbreak_reg(xvar, ext='ppt')
    analyzer.outbreak_reg(xvar, ext='ppt', by_stype=True)
    #analyzer.outbreak_reg_by_stype(xvar, height=6, aspect=1.4, ext='ppt', nboot=50, legend=True)
    #analyzer.outbreak_size_plot(xvar) #xvar, rowvar=None, ext=None, height=6, aspect=1.4, scatter=True, jitter=0.012
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()

    def two_panel(xvar, ext=None, jitter=0.2, values=None, legend=False, height=12, aspect=1.0):
        df = analyzer.results.loc['outbreak_size'].reset_index().rename({'value':'Outbreak Size'}, axis=1)
        if values is not None:
            df = df.loc[df[xvar].isin(values)]
        else:
            values = df[xvar].unique()

        if pd.api.types.is_numeric_dtype(df[xvar]):
            df['x_jittered'] = df[xvar] + np.random.uniform(low=-jitter/2, high=jitter/2, size=df.shape[0])
        else:
            df['x_jittered'] = pd.Categorical(df[xvar]).codes + np.random.uniform(low=-jitter/2, high=jitter/2, size=df.shape[0])

        fig, axv = plt.subplots(2,1, figsize=(height*aspect, height), sharex=False)

        xt = df[xvar].unique()

        # Panel 1
        ax = axv[0]
        g = sns.scatterplot(data=df, x='x_jittered', y='Outbreak Size', size='Outbreak Size', hue='Outbreak Size', sizes=(1, 250), palette='rocket', alpha=0.6, legend=legend, ax=ax)

        ax.set_xticks(xt)
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_xlim(df['x_jittered'].min()-0.02, df['x_jittered'].max()+0.02)
        ax.axhline(y=1, color='k', ls='--')

        ax.set_ylabel('Individual outbreak size')

        # Panel 2
        ax = axv[1]
        analyzer.outbreak_size_stacked_distrib(xvar, ax=ax)
        ax.set_xticks(xt)
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_xlim(df[xvar].min(), df[xvar].max())

        for i in range(2):
            axv[i].set_ylim(0,None)
            axv[i].set_xticks(xt)
            if i == 1:
                axv[i].set_xticklabels( [f'{analyzer.beta0*betamult:.1%}' for betamult in xt] )
                axv[i].set_xlabel('Transmission probability in schools, per-contact per-day')

        plt.tight_layout()

        fn = 'OutbreakTwoPanel.png' if ext is None else f'OutbreakTwoPanel_{ext}.png'
        cv.savefig(os.path.join(analyzer.imgdir, fn), dpi=300)
        return fig

    two_panel(xvar, ext=None, jitter=0.2, values=None, legend=False, height=8, aspect=1.2)


    mgr.tsplots()
