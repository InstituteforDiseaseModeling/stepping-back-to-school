'''
Functions to support the webapp. Note:
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

# Optional imports
try:
    import scirisweb as sw
    import altair as alt
except Exception as E:
    class FailedImport:
        def __init__(self, module, E):
            self.module = module
            self.E = E
        def __getattr__(self, key):
            err = f'This function is not available since {self.module} is not found (error: {self.E}). Please reinstall {self.E} and Altair via "pip install -e .[web]"".'
            raise ImportError(err)
    sw = FailedImport(module='scirisweb', E=E)
    alt = FailedImport(module='altair', E=E)



default_prev = 0.25 # Default prevalence, in percent (i.e. 0.25%, not 25%!)
default_n_days = 7
default_pop_size = 50e3


def get_diagnostic_keys(key=None):
    ''' Get diagnostic keys, or check that one is valid '''
    keys = ['None', 'Antigen every 1w teach&staff', 'Antigen every 4w', 'Antigen every 2w', 'Antigen every 1w', 'PCR every 1w']
    return keys


def check_diagnostic_key(key):
    keys = get_diagnostic_keys()
    if key not in keys:
        errormsg = f'Key "{key}" not valid; choices are {keys}'
        raise sc.KeyNotFoundError(errormsg)
    return key



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

    def __init__(self, es=None, ms=None, hs=None, prev=None, school_sizes=None, immunity=None,
                 n_days=None, n_samples=None, diagnostic=None, scheduling=None, seed=None):
        '''
        Simple class to calculate and plot introduction rates. The equation is roughly:

            rate = (school size) × (number of days) × (prevalence) × (1 - immunity) × (testing)  × (scheduling)  × (screening)

        This is then used as the rate of a Poisson process to calculate a distribution of introductions for each school.

        Args:
            es (int)            : number of elementary schools [range: 0,~10]
            ms (int)            : number of middle schools [range: 0,~10]
            hs (int)            : number of high schools [range: 0,~10]
            school_sizes (dict) : use these supplied school sizes instead of calculating them
            prev (float)        : COVID prevalence in the population, in percent [range: 0,100]
            immunity (float)    : immunity level (fraction) [range: 0,1]
            n_days (int)        : number of days to calculate over [range: 1,~365]
            n_samples (int)     : number of trials to calculate per school [range: 1,~1000]
            diagnostic (str)    : type of diagnostic testing; options are None/'None', 'weekly', 'fortnightly'
            scheduling (str)    : type of scheduling; options are None/'None', 'with_countermeasures', 'all_hybrid', 'k5'
            seed (int)          : random seed to use [range:0,~inf]
        '''
        # Set defaults
        if es        is None: es = 2
        if ms        is None: ms = 2
        if hs        is None: hs = 2
        if prev      is None: prev = default_prev
        if immunity  is None: immunity = 0.1
        if n_days    is None: n_days = default_n_days
        if n_samples is None: n_samples = 200
        self.stypes = ['es', 'ms', 'hs']
        self.slabels = sc.objdict(es='Elementary', ms='Middle', hs='High')

        # Store values
        self.es           = es
        self.ms           = ms
        self.hs           = hs
        self.school_sizes = school_sizes
        self.prev         = prev
        self.prev_frac    = self.prev/100.0
        self.immunity     = immunity
        self.n_days       = n_days
        self.n_weekdays   = days_to_weekdays(n_days)
        self.n_samples    = n_samples
        self.diagnostic   = str(diagnostic).lower() # Allow None = 'None' = 'none'
        self.scheduling   = str(scheduling).lower()
        self.seed         = seed

        # Set efficacies
        self.eff = sc.objdict()
        self.eff.diagnostic = sc.odict({'none':1, 'weekly':0.5, 'fortnightly':0.75})
        self.eff.scheduling = sc.odict({'none':1, 'with_countermeasures':0.67, 'all_hybrid':0.5})

        # Finalize initialization
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
                risk_pp = 1-(1-self.prev_frac)**self.n_weekdays
                immunity = 1 - self.immunity

                # Interventions
                interv = self.eff.diagnostic[self.diagnostic] * self.eff.scheduling[self.scheduling]

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


    def plot(self, **kwargs):
        fig = pl.figure(**kwargs)
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



def multi_school_intro_api(web=True, **kwargs):
    ''' Plot introductions for a number of schools '''
    icalc = IntroCalc(**kwargs)
    fig = icalc.plot()
    if web:
        rawdata = icalc.to_dict()
        graphjson = sw.mpld3ify(fig, jsonify=False)  # Convert to dict
        return graphjson, rawdata
    else:
        return fig, icalc


def single_school_intro_api(web=True, which=None, size=None, n_samples=None, **kwargs):
    ''' Plot introductions for a single school '''
    # Handle inputs
    if which     is None: which     = 'es'
    if size      is None: size      = 200
    if n_samples is None: n_samples = 1000
    stypes = ['es', 'ms', 'hs']
    pars = sc.mergedicts({k:0 if k != which else 1 for k in stypes}, {'school_sizes':{k:[] if k != which else [size] for k in stypes}, 'n_samples': n_samples}, kwargs)

    # Calculate
    icalc = IntroCalc(**pars)
    rawdata = icalc.to_dict()

    # Parse
    ints = icalc.samples['introductions'].values
    y,x = np.histogram(ints, bins=np.arange(max(ints)+1))
    y = y/y.sum()*100
    x = x[:-1]

    xlabel = 'Number of introductions '
    ylabel = 'Probability (%)'
    df = pd.DataFrame({xlabel:x, ylabel:y})
    chart = alt.Chart(df).mark_bar(
        size=300/max(ints),
    ).encode(
        alt.X(xlabel),
        alt.Y(ylabel),
        tooltip = [
            alt.Tooltip(xlabel, format='0.0f'),
            alt.Tooltip(ylabel, format='0.1f')]
    ).properties(
        title=f'Expected introductions over {icalc.n_days} days'
    ).interactive()

    # Finish up
    if web:
        graphjson = sc.loadjson(string=chart.to_json()) # To return as a dictionary rather than a string
        return graphjson, rawdata
    else:
        return chart, icalc


def webapp_popfile(popfile, pop_size, seed):
    ''' Get a consistent name for the population file '''
    if popfile is None:
        popfile = os.path.join(cfg.paths.inputs, f'webapp_{pop_size/1e3:g}k_seed{seed:g}.pop')
    return popfile


def load_trimmed_pop(pop_size=50e3, seed=1, force=False, popfile=None, **kwargs):
    ''' Create a full population, and then trim it down to just the schools '''

    popfile = webapp_popfile(popfile, pop_size, seed)

    # Create or load the initial population
    if force or not os.path.isfile(popfile):
        print(f'Recreating population and saving to {popfile}...')
        kwargs = dict(pop_size=pop_size, location=cfg.pop_pars.location, folder=cfg.paths.inputs, popfile=popfile, **kwargs) # TODO: use community_contacts=0, rm_layers=['w','c','l']
        pop = cvsch.make_population(**kwargs, rand_seed=seed)
    else:
        print(f'Loading population from {popfile}...')
        pop = cv.load(popfile)

    return pop


class OutbreakCalc:

    def __init__(self, pop_size=None, prev=None, n_days=None, diagnostic=None, scheduling=None, seed=None, force=False, **kwargs):
        '''
        Wrapper for the Manager to handle common tasks.

        Args:
            pop_size (int)      : number of people
            prev (float)        : COVID prevalence in the population, in percent [range: 0,100]
            n_days (int)        : number of days to calculate over
            diagnostic (str)    : type of diagnostic testing; options are None/'none', 'weekly', 'fortnightly'
            scheduling (str)    : type of scheduling; options are None/'as_normal', 'with_countermeasures', 'all_hybrid', 'k5'
            seed (int)          : random seed to use
            force (bool)        : whether to recreate the population
            kwargs (dict)       : passed to Manager()
        '''
        if pop_size is None: pop_size = default_pop_size
        if prev is None: prev = default_prev
        if n_days is None: n_days = default_n_days
        if seed is None: seed = 1
        self.pop_size   = pop_size
        self.prev       = prev
        self.prev_frac  = prev/100
        self.n_days     = n_days
        self.diagnostic = 'None' if diagnostic in [None,'none'] else diagnostic
        self.scheduling = 'as_normal' if scheduling in [None,'none','None'] else scheduling
        self.seed       = seed
        self.force      = force
        self.kwargs     = kwargs
        self.initialize(**kwargs)
        return

    def initialize(self, **kwargs):
        self.is_run = False
        self.is_analyzed = False
        cfg.set_micro() # Use this for most settings
        cfg.sim_pars.pop_size = self.pop_size # ...but override population size
        cfg.sweep_pars.n_prev = None # ...and prevalence
        cfg.sweep_pars.prev = [self.prev_frac]
        self.pop = load_trimmed_pop(pop_size=self.pop_size, seed=self.seed, force=self.force)

        # Make the manager
        cfg.sweep_pars.update(dict(screen_keys=[self.diagnostic], schcfg_keys=[self.scheduling]))
        self.mgr = man.Manager(cfg=cfg, sweep_pars=cfg.sweep_pars, check_versions=False, **kwargs)
        return

    def run(self):
        ''' Rerun the analysis, with the custom population '''
        print('#'*100)
        print(self.mgr.sweep_pars)
        self.mgr.run(force=True, people=self.pop)
        self.is_run = True
        return

    def analyze(self):
        if not self.is_run:
            self.run()
        self.analyzer = self.mgr.analyze(do_save=False)
        self.is_analyzed = True
        return

    def plot(self):
        xvar = 'Prevalence Target'
        if not self.is_analyzed:
            self.analyze()
        ax = self.analyzer.outbreak_size_distrib(xvar=xvar)
        fig = ax.figure
        return fig

    def plot_trees(self, max_trees=10):
        if not self.is_analyzed:
            self.analyze()
        figs = []
        n_outbreaks = min(len(self.analyzer.outbreaks), max_trees)
        for o in range(n_outbreaks):
            fig = self.analyzer.plot_tree(outbreak_ind=o)
            figs.append(fig)
        return figs


    def to_dict(self):
        ''' Only return the outbreak part of the data structure '''
        data = self.analyzer.outbreaks.to_dict()['outbreak']
        for k in data.keys():
            data[k]['sim'] = data['sim'][k]
            data[k]['school'] = data['school'][k]
            data[k]['Tree'] = nx.readwrite.json_graph.node_link_data(data['outbreak'][k]['Tree'])
            for node_data in data[k]['Tree']['nodes']:
                for param_key, param_value in node_data.items():
                    if sc.isnumber(param_value):
                        if np.isnan(param_value):
                            node_data[param_key] = ""
                        else:
                            node_data[param_key] = float(param_value)
        return data

    def to_json(self):
        data = self.to_dict()
        return sc.jsonify(data, tostring=True)


def outbreak_api(web=True, **kwargs):
    ''' Plot outbreaks '''
    ocalc = OutbreakCalc(**kwargs)
    outbreakfig = ocalc.plot()
    treefigs = ocalc.plot_trees()
    if web:
        rawdata = ocalc.to_dict()
        outbreakgraph = sw.mpld3ify(outbreakfig, jsonify=False)
        treegraphs = [sw.mpld3ify(treefigs, jsonify=False) for fig in treefigs]
        return outbreakgraph, treegraphs, rawdata
    else:
        return outbreakfig, treefigs, ocalc
