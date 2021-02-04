import os
import numpy as np
import covasim as cv
import sciris as sc
import synthpops as sp
import covasim_schools as cvsch
import builder as bld
import analysis as an
import utils as ut
import config as cfg

class Run(sc.objdict):
    '''
    This is the main class used for commissioning the different testing scenarios
    (defined in testing_scenarios.py) for the paper results. Run this script first
    (preferably on an HPC!), then run the different plotting scripts. Each sim takes
    about 1 minute to run. With full settings, there are about 1300 scripts to run;
    the test run uses 8.
    '''

    def __init__(self, name=None, sim_pars=None, sweep_pars=None, run_pars=None, pop_pars=None, paths=None, cfg=cfg):

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
        self.sims = None  # To be run or loaded by calling run()
        self.analyzer = None

        if self.sweep_pars.prev is None:
            self.sweep_pars.prev = np.linspace(0.002, 0.02, self.sweep_pars.n_prev) # TODO: this might create subtle bugs and shouldn't be hard-coded

        self.stem = f'{self.pop_pars.location}_{self.name}_{self.sim_pars.pop_size}_{self.sweep_pars.n_reps}reps'
        self.dir = os.path.join(self.paths.outputs, self.stem)
        self.cachefn = os.path.join(self.dir, 'results.sims') # Might need to change the extension here, depending if combine.py was used

        self.builder = bld.Builder(self.sim_pars, self.sweep_pars, self.paths)

        return


    def build_configs(self):
        ''' Build simulation configuration '''

        sc.heading('Creating sim configurations...')

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
            self.sims = ut.run_configs(sim_configs, self.stem, self.run_pars, self.cachefn) # why is stem needed here?
        else:
            print(f'Loading {self.cachefn}')
            self.sims = cv.load(self.cachefn) # Use for *.sims
        return
