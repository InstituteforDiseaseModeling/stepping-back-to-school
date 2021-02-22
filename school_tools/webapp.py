'''
Functions to support the webapp
'''

import os
import numpy as np
import pandas as pd
import sciris as sc
import pylab as pl
import seaborn as sns
import synthpops as sp
import networkx as nx
import covasim as cv
import covasim_schools as cvsch
from . import config as cfg
from . import manager as man


def get_school_sizes(es=1, ms=1, hs=1):
    ''' Get school sizes from different schools '''
    kwargs = dict(datadir=sp.datadir, location='seattle_metro', state_location='Washington', country_location='usa')
    distrs = sp.get_school_size_distr_by_type(**kwargs)
    brackets = sp.get_school_size_brackets(**kwargs)
    output = sc.objdict(es=np.zeros(es), ms=np.zeros(ms), hs=np.zeros(hs))
    n = sc.objdict(es=es, ms=ms, hs=hs)
    for st in ['es', 'ms', 'hs']:
        if n[st]:
            keys = np.array(list(distrs[st].keys()))
            probs = np.array(list(distrs[st].values()))
            inds = np.random.choice(a=keys, size=n[st], p=probs, replace=True)
            for i,ind in enumerate(inds):
                size = np.random.choice(brackets[ind])
                output[st][i] = size
    return output


def days_to_weekdays(days):
    ''' Simple calculator to take a number of days and round down to the number of weekdays '''
    weeks = days // 7
    remainder = min(5, days % 7)
    weekdays = 5*weeks + remainder
    return weekdays


class IntroCalc(sc.objdict):

    def __init__(self, es=None, ms=None, hs=None, prev=None, school_sizes=None, immunity=None, n_days=None, n_samples=None, diagnostic=None, scheduling=None, symp=None, seed=None):
        '''
        Simple class to calculate and plot introduction rates. The equation is roughly:

            rate = (school size) × (number of days) × (prevalence) × (1 - immunity) × (testing)  × (scheduling)  × (screening)

        This is then used as the rate of a Poisson process to calculate a distribution of introductions for each school.

        Args:
            es (int)            : number of elementary schools
            ms (int)            : number of middle schools
            hs (int)            : number of high schools
            school_sizes (dict) : use these supplied school sizes instead of calculating them
            prev (float)        : prevalence, i.e. case rate per 100,000 over 14 days
            immunity (float)    : immunity level (fraction)
            n_days (int)        : number of days to calculate over
            n_samples (int)     : number of trials to calculate per school
            diagnostic (str)    : type of diagnostic testing; options are None, 'weekly', 'fortnightly'
            scheduling (str)    : type of scheduling; options are None or 'hybrid'
            symp (str)          : type of symptom screening; options are None or 'all'
            seed (int)          : random seed to use
        '''
        if es is None: es = 2
        if ms is None: ms = 2
        if hs is None: hs = 2
        if prev is None: prev = 50
        if immunity is None: immunity = 0.1
        if n_days is None: n_days = 5
        if n_samples is None: n_samples = 200
        self.stypes = ['es', 'ms', 'hs']
        self.slabels = sc.objdict(es='Elementary', ms='Middle', hs='High')
        self.es = es
        self.ms = ms
        self.hs = hs
        self.school_sizes = school_sizes
        self.prev = prev
        self.immunity = immunity
        self.n_days = n_days
        self.n_samples = n_samples
        self.diagnostic = diagnostic
        self.scheduling = scheduling
        self.symp = symp
        self.seed = seed

        # Set efficacies
        self.eff = sc.objdict()
        self.eff.diagnostic = sc.odict({None:1, 'weekly':0.5, 'fortnightly':0.75})
        self.eff.scheduling = sc.odict({None:1, 'hybrid':0.75})
        self.eff.symp = sc.odict({None:1, 'all':0.67})

        # Finalize
        self.initialize()
        return


    def initialize(self):
        ''' Set the school sizes and calculate the samples '''
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.school_sizes is None:
            self.init_schools()
        self.calc_rates()
        self.draw_samples()
        return


    def init_schools(self):
        ''' Randomly draw sizes for each school '''
        self.school_sizes = get_school_sizes(es=self.es, ms=self.ms, hs=self.hs)
        return


    def calc_rates(self):
        ''' The meat of the class: calculate the introduction rate for each school '''
        self.rates = sc.dcp(self.school_sizes) # Each school has an associated rate
        for sch_type in self.stypes:
            for s,sch_size in enumerate(self.school_sizes[sch_type]):

                # Per person risk and immunity
                risk_pp = 1-(1-self.prev/1e5)**self.n_days
                immunity = 1 - self.immunity

                # Interventions
                interv = 1
                interv *= self.eff.diagnostic[self.diagnostic]
                interv *= self.eff.scheduling[self.scheduling]
                interv *= self.eff.symp[self.symp]

                # Calculate the rate
                rate = sch_size * risk_pp * immunity * interv
                self.rates[sch_type][s] = rate

        return


    def draw_samples(self):
        ''' Draws Poisson samples based on the associated rates '''
        # self.samples = pd.DataFrame(columns=['sch_ind', 'sch_type', 'sch_size', 'intro_rate', 'introductions'])
        data = []
        sch_ind = -1
        for sch_type in self.stypes:
            for s,sch_size in enumerate(self.school_sizes[sch_type]):
                sch_ind += 1
                samples = np.random.poisson(lam=self.rates[sch_type][s], size=self.n_samples)
                for sample in samples:
                    data.append(dict(sch_ind=sch_ind, School=self.slabels[sch_type], sch_size=self.school_sizes[sch_type][s], intro_rate=self.rates[sch_type][s], introductions=sample))
        self.samples = pd.DataFrame(data)
        return


    def plot(self):
        ax = sns.violinplot(x="sch_ind", y="introductions", hue="School", data=self.samples, cut=0)
        ax.set_xticks([])
        ax.set_xlabel('Schools')
        ax.set_ylabel(f'Expected introductions over {self.n_days} days')
        fig = pl.gcf()
        return fig


    def to_json(self):
        return self.samples.to_json()


    def to_dict(self):
        return self.samples.to_dict()



def plot_introductions(**kwargs):
    ''' Plot introduction rate '''
    icalc = IntroCalc(**kwargs)
    fig = icalc.plot()
    return icalc, fig


def webapp_popfile(popfile, pop_size, seed):
    ''' Get a consistent name for the population file '''
    if popfile is None:
        popfile = os.path.join(cfg.paths.inputs, f'webapp_{pop_size/1e3:g}k_seed{seed:g}.pop')
    return popfile


def load_trimmed_pop(pop_size=100e3, seed=1, force=False, popfile=None, **kwargs):
    ''' Create a full population, and then trim it down to just the schools '''

    popfile = webapp_popfile(popfile, pop_size, seed)

    # Create or load the initial population
    if force or not os.path.isfile(popfile):
        print(f'Recreating population and saving to {popfile}...')
        kwargs = dict(pop_size=pop_size, location=cfg.pop_pars.location, folder=cfg.paths.inputs, popfile=popfile, **kwargs) # Attempts to use community_contacts=0, rm_layers=['w','c','l'] failed
        pop = cvsch.make_population(**kwargs, rand_seed=seed)
    else:
        print(f'Loading population from {popfile}...')
        pop = cv.load(popfile)

    return pop


class OutbreakCalc:

    def __init__(self, pop_size=None, prev=None, n_days=None, diagnostic=None, scheduling=None, symp=None, seed=None, force=False, **kwargs):
        '''
        Wrapper for the Manager to handle common tasks.

        Args:
            pop_size (int)      : number of people
            prev (float)        : prevalence in the community
            n_days (int)        : number of days to calculate over
            diagnostic (str)    : type of diagnostic testing; options are None, 'weekly', 'fortnightly'
            scheduling (str)    : type of scheduling; options are None or 'hybrid'
            symp (str)          : type of symptom screening; options are None or 'all'
            seed (int)          : random seed to use
            force (bool)        : whether to recreate the population
            kwargs (dict)       : passed to Manager()
        '''
        if pop_size is None: pop_size = 20e3
        if prev is None: prev = 0.02
        if n_days is None: n_days = 5
        if seed is None: seed = 1
        self.pop_size = pop_size
        self.prev = prev
        self.n_days = n_days
        self.diagnostic = diagnostic
        self.scheduling = scheduling
        self.symp = symp
        self.seed = seed
        self.force = force
        self.initialize(**kwargs)
        return

    def initialize(self, **kwargs):
        self.is_run = False
        self.is_analyzed = False
        cfg.set_micro() # Use this for most settings
        cfg.sim_pars.pop_size = self.pop_size # ...but override population size
        cfg.sweep_pars.n_prev = None # ...and prevalence
        cfg.sweep_pars.prev = [self.prev]
        self.pop = load_trimmed_pop(pop_size=self.pop_size, seed=self.seed, force=self.force)
        self.mgr = man.Manager(cfg=cfg, **kwargs)
        return

    def run(self):
        ''' Rerun the analysis, with the custom population '''
        self.mgr.run(force=True, people=self.pop)
        self.is_run = True
        return

    def analyze(self):
        if not self.is_run:
            self.run()
        self.analyzer = self.mgr.analyze()
        self.is_analyzed = True
        return

    def plot(self):
        xvar = 'Prevalence Target'
        if not self.is_analyzed:
            self.analyze()
        return self.analyzer.outbreak_size_plot(xvar=xvar)

    def to_dict(self):
        ''' Only return the outbreak part of the data structure '''
        data = self.analyzer.outbreaks.to_dict()
        for k in data['outbreak'].keys():
            data['outbreak'][k]['sim'] = data['sim'][k]
            data['outbreak'][k]['school'] = data['school'][k]
            data['outbreak'][k]['Tree'] = nx.readwrite.json_graph.node_link_data(data['outbreak'][k]['Tree'])
        return data['outbreak']

    def to_json(self):
        data = self.to_dict()
        return sc.jsonify(data, tostring=True)


def plot_outbreaks(**kwargs):
    ''' Plot introduction rate '''
    ocalc = OutbreakCalc(**kwargs)
    ocalc.plot()
    fig = pl.gcf()
    return ocalc, fig
