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

class Run:
    '''
    This is the main class used for commissioning the different testing scenarios
    (defined in testing_scenarios.py) for the paper results. Run this script first
    (preferably on an HPC!), then run the different plotting scripts. Each sim takes
    about 1 minute to run. With full settings, there are about 1300 scripts to run;
    the test run uses 8.
    '''

    def __init__(self, name=None, sim_pars=None, sweep_pars=None, run_pars=None):
        # Check that versions are correct
        sp_gitinfo = sc.gitinfo(sp.__file__)
        sp_ver = sp.__version__
        sp_expected = '1.4.0'
        if sc.compareversions(sp_ver, sp_expected) < 0:
            errormsg = f'This code is designed to work with SynthPops >= {sp_expected}, but you have {sp_ver}'
            raise ImportError(errormsg)
        cv.check_save_version('2.0.0', folder='gitinfo', comments={'SynthPops':sp_gitinfo})

        self.name = self.__class__.__name__ if name is None else name
        self.sims = None  # To be run or loaded by calling run()
        self.analyzer=None

        # TODO: move to defaults
        self.sim_pars = sc.dcp(cfg.sim_pars)
        if sim_pars is not None:
            self.sim_pars.update(sim_pars)

        self.sweep_pars = sc.dcp(cfg.sweep_pars)
        if sweep_pars is not None:
            self.sweep_pars.update(sweep_pars)
        if self.sweep_pars.prev is None:
            self.sweep_pars.prev = np.linspace(0.002, 0.02, self.sweep_pars.n_prev) # TODO: this might create subtle bugs and shouldn't be hard-coded

        # TODO: make default config`, update with user config
        self.run_pars = sc.dcp(cfg.run_pars)
        if run_pars is not None:
            self.run_pars.update(run_pars)

        self.paths = sc.dcp(cfg.paths)
        self.stem = f'{self.name}_{self.sim_pars.pop_size}_{self.sweep_pars.n_reps}reps'
        self.cachefn = os.path.join(self.paths.outputs, 'sims', f'{self.stem}.sims') # Might need to change the extension here, depending in combine.py was used
        self.imgdir = os.path.join(self.paths.outputs, 'img_'+self.stem)

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
        rep_levels = {f'Run {p}':{'rand_seed':p} for p in range(self.sweep_pars['n_reps'])}
        self.builder.add_level('Replicate', rep_levels, self.builder.simpars_func)

        # Add school intervention
        for config in self.builder.configs:
            config.interventions.append(cvsch.schools_manager(config.school_config))

        return self.builder.get()


    def analyze(self):
        ''' Create (and run) the analysis '''
        self.analyzer = an.Analysis(self.sims, self.imgdir)
        return self.analyzer

    def regplots(self, xvar, huevar, ts_plots=None, order=2):
        ''' Generate regular plots '''

        if self.analyzer is None:
            self.analyze()

        self.analyzer.introductions_rate(xvar, huevar, order=order)
        self.analyzer.introductions_rate_by_stype(xvar, huevar, order=order)
        self.analyzer.introductions_reg(xvar, huevar, order=order)
        self.analyzer.outbreak_reg(xvar, huevar, order=order)

        return


    def tsplots(self, ts_plots=None):
        ''' Generate time series plots '''

        print('Generating timeseries plots, these take a few minutes...')
        if self.analyzer is None:
            self.analyze()

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
