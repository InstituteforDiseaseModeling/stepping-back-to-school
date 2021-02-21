'''
Functions to support the webapp
'''

import numpy as np
import sciris as sc
import pylab as pl
import seaborn as sns
import synthpops as sp
import covasim as cv


def get_school_sizes(es=1, ms=1, hs=1, seed=None):
    ''' Get school sizes from different schools '''
    if seed is not None:
        np.random.seed(seed)
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


class IntroCalc(sc.objdict):

    def __init__(self, es=None, ms=None, hs=None, prev=None, immunity=None, n_days=None, n_trials=None, diagnostic=None, scheduling=None, symp=None):
        '''
        Simple class to calculate and plot introduction rates.

        Args:
            es (int): number of elementary schools
            ms (int): number of middle schools
            hs (int): number of high schools
            prev (float): prevalence, i.e. case rate per 100,000 over 14 days
            immunity (float): immunity level (fraction)
            n_days (int): number of days to calculate over
            n_trials (int): number of trials to calculate per school
            diagnostic (str): type of diagnostic testing; options are None, 'weekly', 'fortnightly'
            scheduling (str): type of scheduling; options are None or 'hybrid'
            symp (str): type of symptom screening; options are None or 'all'
        '''
        if es is None: es = 2
        if ms is None: ms = 2
        if hs is None: hs = 2
        if prev is None: prev = 50
        if immunity is None: immunity = 0.1
        if n_days is None: n_days = 5
        if n_trials is None: n_trials = 200
        self.es = es
        self.ms = ms
        self.hs = hs
        self.prev = prev
        self.immunity = immunity
        self.n_days = n_days
        self.n_trials = n_trials
        self.diagnostic = diagnostic
        self.scheduling = scheduling
        self.symp = symp
        self.schools = sc.objdict()
        self.rates = sc.objdict()
        self.results = sc.objdict()
        return

    def calculate(self):
        pass

    def plot(self):
        pass


# def calculate_introductions():


def plot_introductions(es=None, ms=None, hs=None, n_trials=None):
    ''' Plot introduction rate '''
    icalc = IntroCalc(es=es, ms=ms, hs=hs, n_trials=n_trials)
    fig = icalc.plot()
    return fig


if __name__ == '__main__':

    fig = plot_introductions(es=4, ms=3, hs=2)