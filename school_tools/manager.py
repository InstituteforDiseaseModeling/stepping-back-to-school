'''
Build and run simulations
'''

import os
import psutil
import multiprocessing as mp

import numpy as np
import sciris as sc
import covasim as cv
import synthpops as sp

import covasim_controller as cvc
import covasim_schools as cvsch

from . import scenarios as scn
from . import analysis as an
from . import config as cfg
from . import create as cr


__all__ = ['Config', 'Builder', 'Manager', 'create_run_sim', 'run_configs', 'alternate_symptomaticity']


class Config:
    def __init__(self, sim_pars=None, label=None, tags=None):
        self.label = label # TODO: From tags?
        self.tags = {}
        if tags is not None:
            self.tags.update(tags)

        # TODO: Seems necessary to have access to e.g. prognosis parameters, but can work around
        self.sim_pars = sim_pars #cv.make_pars(set_prognoses=True, prog_by_age=True, **sim_pars)
        self.school_config = None
        self.interventions = []
        self.count = 0

    def __repr__(self):
        return f'''
{'-'*80}
Configuration {self.label}:
 * Tags: {self.tags}
 * Pars: {self.sim_pars}
 * School config: {self.school_config}
 * Num interventions: {len(self.interventions)}
 '''


class Builder(sc.prettyobj):
    '''
    Build the run configurations
    '''

    def __init__(self, sim_pars, sweep_pars, paths):

        self.configs = [Config(sim_pars=sim_pars, tags=dict(school_start_date=sweep_pars.school_start_date))]

        # These come from fit_transmats - don't like loading multiple times
        self.ei = sc.loadobj(paths.ei)
        self.ir = sc.loadobj(paths.ir)

        all_scen = scn.generate_scenarios(start_date=sweep_pars.school_start_date, seed_date=sweep_pars.school_seed_date) # Can potentially select a subset of scenarios
        scens = {k:v for k,v in all_scen.items() if k in sweep_pars.schcfg_keys}
        self.add_level('scen_key', scens, self.scen_func)

        if sweep_pars.alt_sus:
            value_labels = {'Yes' if p else 'No':p for p in [True]}
            self.add_level('AltSus', value_labels, alternate_symptomaticity)

        all_screenings = scn.generate_screening(sweep_pars.school_start_date) # Potentially select a subset of diagnostic screenings
        screens = {k:v for k,v in all_screenings.items() if k in sweep_pars.screen_keys}

        # Would like to reuse screenpars_func here
        def screen_func(config, key, test):
            print(f'Building screening parameter {key}={test}')
            for stype, spec in config.school_config.items():
                if spec is not None:
                    spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools
            return config

        self.add_level('dxscrn_key', screens, screen_func)

    @staticmethod
    def scen_func(config, key, school_config):
        print(f'Building school configuration {key}={school_config}')
        config.school_config = sc.dcp(school_config)
        return config

    @staticmethod
    def screenpars_func(config, key, screenpar): # Generic to screen pars, move to builder
        print(f'Building screening parameter {key}={screenpar}')
        for stype, spec in config.school_config.items():
            if spec is not None:
                spec.update(screenpar)
        return config


    @staticmethod
    def simpars_func(config, key, simpar): # Generic to screen pars, move to builder
        print(f'Building simulation parameter {key}={simpar}')
        config.sim_pars.update(simpar)
        return config

    def prevctr_func(self, config, key, prev):
        print(f'Building prevalence controller {key}={prev}')
        pole_loc = 0.35

        seir = cvc.SEIR(config.sim_pars['pop_size'], self.ei.Mopt, self.ir.Mopt, ERR=1, beta=0.365, Ipow=0.925)
        targets = dict(infected= prev * config.sim_pars['pop_size']) # prevalence target
        ctr = cvc.controller_intervention(seir, targets, pole_loc=pole_loc, start_day=1)
        config.interventions += [ctr]
        return config

    def add_level(self, keyname, level, func):

        # Map a function name to method
        if isinstance(func, str):
            try:
                func = dict(
                    scen_func = self.scen_func,
                    screenpars_func = self.screenpars_func,
                    simpars_func = self.simpars_func,
                    prevctr_func = self.prevctr_func,
                    )[func]
            except Exception as E:
                errormsg = f'Could not recognize function "{func}": {str(E)}'
                raise ValueError(errormsg)

        new_configs = []
        for config in self.configs:
            for k,v in level.items():
                cfg = func(sc.dcp(config), k, v)
                cfg.tags[keyname] = k
                new_configs += [cfg]
        self.configs = new_configs

    def __repr__(self):
        ret = ''
        for config in self.configs:
            ret += str(config)
        return ret

    def get(self):
        for i, config in enumerate(self.configs):
            config.count = i
        print(f'Done: {len(self.configs)} configurations created')
        return self.configs


class Manager(sc.objdict):
    '''
    This is the main class used for commissioning the different testing scenarios
    (defined in testing_scenarios.py) for the paper results. Run this script first
    (preferably on an HPC!), then run the different plotting scripts. Each sim takes
    about 1 minute to run. With full settings, there are about 1300 scripts to run;
    the test run uses 8.
    '''

    def __init__(self, name=None, sim_pars=None, sweep_pars=None, run_pars=None, pop_pars=None, paths=None, levels=None, cfg=cfg):

        # Handle inputs
        input_pars = sc.objdict()
        input_pars.sim_pars = sim_pars
        input_pars.sweep_pars = sweep_pars
        input_pars.run_pars = run_pars
        input_pars.pop_pars = pop_pars
        input_pars.paths = paths
        for k,pars in input_pars.items():
            defaults = getattr(cfg, k)
            self[k] = sc.dcp(sc.objdict(sc.mergedicts(defaults, pars))) # Copy the merged objdict

        # Check that versions are correct
        sp_gitinfo = sc.gitinfo(sp.__file__)
        sp_ver = sp.__version__
        sp_expected = '1.4.0'
        if sc.compareversions(sp_ver, sp_expected) < 0:
            errormsg = f'This code is designed to work with SynthPops >= {sp_expected}, but you have {sp_ver}'
            raise ImportError(errormsg)
        cv.check_save_version('2.0.2', folder='gitinfo', comments={'SynthPops':sp_gitinfo})

        self.name = self.__class__.__name__ if name is None else name
        self.levels = sc.promotetolist(levels) # The scenario(s) to run
        self.sims = None  # To be run or loaded by calling run()
        self.analyzer = None

        if self.sweep_pars.prev is None:
            self.sweep_pars.prev = np.linspace(0.002, 0.02, self.sweep_pars.n_prev) # TODO: this might create subtle bugs and shouldn't be hard-coded

        self.stem = f'{self.pop_pars.location}_{self.name}_{self.sim_pars.pop_size}_{self.sweep_pars.n_reps}reps'
        self.dir = os.path.join(self.paths.outputs, self.stem)
        self.cachefn = os.path.join(self.dir, 'results.sims') # Might need to change the extension here, depending if combine.py was used

        self.builder = Builder(self.sim_pars, self.sweep_pars, self.paths)

        return


    def build_configs(self):
        ''' Build simulation configuration '''

        sc.heading('Creating sim configurations...')

        # Add custom levels
        for lvl in self.levels:
            self.builder.add_level(lvl['keyname'], lvl['level'], lvl['func'])

        # Add prevalence levels
        if len(self.sweep_pars['prev']) > 0:
            prev_levels = {f'{100*p:.1f}%':p for p in self.sweep_pars['prev']}
            self.builder.add_level('Prevalence', prev_levels, self.builder.prevctr_func)

        # Add reps
        rep_levels = {f'Run {p}':{'rand_seed':p+cfg.run_pars.base_seed} for p in range(self.sweep_pars['n_reps'])}
        self.builder.add_level('Replicate', rep_levels, self.builder.simpars_func)

        # Add school intervention
        for config in self.builder.configs:
            config.interventions.append(cvsch.schools_manager(config.school_config))

        return self.builder.get()


    def analyze(self, rerun=True):
        ''' Create (and run) the analysis '''
        if self.analyzer is None or rerun:
            self.analyzer = an.Analysis(self.sims, self.dir)
        return self.analyzer

    def regplots(self, xvar, huevar, ts_plots=None):
        ''' Generate regular plots '''

        self.analyze(rerun=False)
        self.analyzer.introductions_rate(xvar, huevar)
        self.analyzer.introductions_rate_by_stype(xvar)
        self.analyzer.outbreak_reg(xvar, huevar)

        return


    def tsplots(self, ts_plots=None):
        ''' Generate time series plots '''

        print('Generating timeseries plots, these take a few minutes...')
        self.analyze(rerun=False)

        if ts_plots is None:
            ts_plots = [
                dict(label='Prevalence',      channel='n_exposed',      normalize=True),
                dict(label='CumInfections',   channel='cum_infections', normalize=True),
                dict(label='Quarantined',     channel='n_quarantined',  normalize=True),
                dict(label='Newly Diagnosed', channel='new_diagnoses',  normalize=False),
            ]

        return self.analyzer.plot_several_timeseries(ts_plots)


    def run(self, force=False):
        ''' Run the sims, or load them from disk '''

        if force or not os.path.isfile(self.cachefn):
            sim_configs = self.build_configs()
            self.sims = run_configs(sim_configs, self.stem, self.run_pars, self.cachefn) # why is stem needed here?
        else:
            print(f'Loading {self.cachefn}')
            self.sims = cv.load(self.cachefn) # Use for *.sims
        return


#%% Running
def create_run_sim(sconf, n_sims, run_config):
    ''' Create and run the actual simulations '''
    label = f'sim {sconf.count} of {n_sims}'
    print(f'Creating and running {label}...')

    T = sc.tic()
    sim = cr.create_sim(sconf.sim_pars, folder=None, max_pop_seeds=cfg.sweep_pars.n_seeds, label=label)

    for intv in sconf.interventions:
        sim['interventions'].append(intv)

    sim.tags = sc.dcp(sconf.tags)

    sim.run()

    if run_config['shrink']:
        sim.shrink() # Do not keep people after run
    sc.toc(T)
    return sim


def run_configs(sim_configs, stem, run_cfg, filename=None):
    n_cpus = run_cfg['n_cpus']
    pop_size = max([c.sim_pars['pop_size'] for c in sim_configs])

    sc.heading('Running sims...')
    TT = sc.tic()
    kwargs = dict(n_sims=len(sim_configs), run_config=run_cfg)
    if run_cfg['parallel']:
        print('...running in parallel')

        sc.heading('Choosing correct number of CPUs...') # TODO: merge with create_pops.py
        if n_cpus is None:
            cpu_limit = int(mp.cpu_count()*run_cfg['cpu_thresh']) # Don't use more than 75% of available CPUs
            ram_available = psutil.virtual_memory().available/1e9
            ram_required = 1.5*pop_size/2.25e5 # Roughly 1.5 GB per 225e3 people
            ram_limit = int(ram_available/ram_required*run_cfg['mem_thresh'])
            n_cpus = min(cpu_limit, ram_limit)
            print(f'{n_cpus} CPUs are being used due to a CPU limit of {cpu_limit} and estimated RAM limit of {ram_limit}')
        else:
            print(f'Using user-specified {n_cpus} CPUs')

        sims = sc.parallelize(create_run_sim, iterarg=sim_configs, kwargs=kwargs, ncpus=n_cpus)
    else:
        print('...running in serial')
        sims = []
        for sconf in sim_configs:
            sim = create_run_sim(sconf, **kwargs)
            sims.append(sim)

    if filename is not None:
        sc.heading('Saving all sims...')
        cv.save(filename, sims)
        print(f'Done, saved {filename}')

    sc.toc(TT)

    return sims


def alternate_symptomaticity(config, key, value):
    print(f'Building alternate symptomaticity {key}={value}')
    if not value: # Only build if value is True
        return config
    if 'prognoses' in config.sim_pars:
        prog = config.sim_pars['prognoses']
    else:
        pars = cv.make_pars(set_prognoses=True, prog_by_age=True, **config.sim_pars)
        prog = pars['prognoses']

    ages = prog['age_cutoffs']
    symp_probs = prog['symp_probs']

    # Source: table 1 from https://arxiv.org/pdf/2006.08471.pdf
    symp_probs[:] = 0.6456
    symp_probs[ages<80] = 0.3546
    symp_probs[ages<60] = 0.3054
    symp_probs[ages<40] = 0.2241
    symp_probs[ages<20] = 0.1809
    prog['symp_probs'] = symp_probs

    config.sim_pars['prognoses'] = sc.dcp(prog) # Ugh

    return config


