'''
Build and run simulations
'''

import os
import psutil
import multiprocessing as mp

import numpy as np
import pandas as pd
import sciris as sc
import covasim as cv
import synthpops as sp

import covasim_controller as cvc
import covasim_schools as cvsch

from . import scenarios as scn
from . import analysis as an
from . import config as cfg
from . import create as cr


__all__ = ['Config', 'Builder', 'Manager', 'Vaccine', 'CohortRewiring', 'create_run_sim', 'run_configs', 'alternate_symptomaticity', 'alternate_susceptibility']


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


class Builder:
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

        value_labels = {'Yes' if p else 'No':p for p in sweep_pars.alt_symp}
        self.add_level('AltSymp', value_labels, alternate_symptomaticity)

        value_labels = {'Yes' if p else 'No':p for p in sweep_pars.alt_sus}
        self.add_level('AltSus', value_labels, alternate_susceptibility)

        self.add_level('Cohort Mixing', sweep_pars.cohort_rewiring, self.add_intervention_func)

        self.add_level('Vaccination', sweep_pars.vaccine, self.add_intervention_func)

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
        if 'location' not in sweep_pars:
            sweep_pars.location = [cfg.pop_pars.location]

        locations = {k.split('_')[0].capitalize() : k for k in sweep_pars.location}
        self.add_level('location', locations, loc_func)

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
    def add_intervention_func(config, key, intv):
        print(f'Building intervention {key}')
        if intv is not None:
            print(f'Adding {intv} to interventions')
            config.interventions += sc.dcp(sc.sc_utils.promotetolist(intv))
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

        self.stem = f'{self.name}_{self.sim_pars.pop_size}_{self.sweep_pars.n_reps}reps'
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

    def regplots(self, xvar, huevar, ts_plots=None, height=6, aspect=1.4):
        ''' Generate regular plots '''

        self.analyze(rerun=False)
        self.analyzer.introductions_rate(xvar, huevar, height=height, aspect=aspect)
        self.analyzer.introductions_rate_by_stype(xvar, height=height, aspect=aspect)
        self.analyzer.outbreak_reg_facet(xvar, huevar, height=height, aspect=aspect)

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


class Vaccine(cv.Intervention):
    def __init__(self, rel_sus_mult, symp_prob_mult, teacher_cov=0, staff_cov=0, student_cov=0):
        self._store_args()
        self.cov = dict(teachers=teacher_cov, staff=staff_cov, students=student_cov)
        self.mult = dict(rel_sus=rel_sus_mult, symp_prob=symp_prob_mult) # Could range check

    def initialize(self, sim):
        sch_ids = [sid for st in ['es', 'ms', 'hs'] for sid in sim.people.school_types[st]]
        schoolpeople_uids = [uid for sid in sch_ids for uid in sim.people.schools[sid]]

        for role, flag in zip(['students', 'teachers', 'staff'], [sim.people.student_flag, sim.people.teacher_flag, sim.people.staff_flag]):
            cov = self.cov[role]
            role_uids = [u for u in schoolpeople_uids if flag[u]]
            # Choose who to vx
            tovx = np.random.choice(role_uids, size=np.random.binomial(len(role_uids),cov), replace=False)
            sim.people.rel_sus[tovx] *= self.mult['rel_sus']
            sim.people.symp_prob[tovx] *= self.mult['symp_prob']

    def apply(self, sim):
        pass


class CohortRewiring(cv.Intervention):
    ''' Break up student cohort "bubbles" to represent after-school care, transportation, etc '''

    def __init__(self, frac_edges_to_rewire=0.5):
        self._store_args()
        self.frac_edges_to_rewire = frac_edges_to_rewire

    def initialize(self, sim):
        if self.frac_edges_to_rewire == 0:
            return

        school_contacts = []

        sdf = sim.people.contacts['s'].to_df() # Must happen before School classes parts out the 's' network
        student_flag = np.array(sim.people.student_flag, dtype=bool)
        sdf['p1_student'] = student_flag[sdf['p1']]
        sdf['p2_student'] = student_flag[sdf['p2']]
        school_types = sim.people.school_types
        for school_type, scids in school_types.items():
            for school_id in scids:
                uids = sim.people.schools[school_id] # Dict with keys of school_id and values of uids in that school
                edges_this_school = sdf.loc[ ((sdf['p1'].isin(uids)) | (sdf['p2'].isin(uids))) ]
                student_to_student_edge_bool = ( edges_this_school['p1_student'] & edges_this_school['p2_student'] )
                student_to_student_edges = edges_this_school.loc[ student_to_student_edge_bool ]
                inds_to_rewire = np.random.choice(student_to_student_edges.index, size=int(self.frac_edges_to_rewire*student_to_student_edges.shape[0]), replace=False)
                if len(inds_to_rewire) == 0:
                    # Nothing to do here!
                    continue
                inds_to_keep = np.setdiff1d(student_to_student_edges.index, inds_to_rewire)

                edges_to_rewire = student_to_student_edges.loc[inds_to_rewire]
                stublist = np.concatenate(( edges_to_rewire['p1'], edges_to_rewire['p2'] ))

                def complete_stubs(stublist):
                    try:
                        p1_inds = np.random.choice(len(stublist), size=len(stublist)//2, replace=False)
                        p2_inds = np.setdiff1d(range(len(stublist)), p1_inds)
                        p1 = stublist[p1_inds]
                        p2 = stublist[p2_inds]
                        new_edges = pd.DataFrame({'p1':p1, 'p2':p2})
                        new_edges['beta'] = cv.defaults.default_float(1.0)
                        return new_edges
                    except:
                        df = pd.DataFrame({
                            'p1': pd.Series([], dtype='int32'),
                            'p2': pd.Series([], dtype='int32'),
                            'beta': pd.Series([], dtype='float32')})
                        return df

                new_edges = complete_stubs(stublist)

                # Remove self loops
                self_loops = new_edges.loc[new_edges['p1'] == new_edges['p2']]
                new_edges = new_edges.loc[new_edges['p1'] != new_edges['p2']]

                # One pass at redoing self loops
                stublist = np.concatenate(( self_loops['p1'], self_loops['p2'] ))
                new_edges2 = complete_stubs(stublist)
                if new_edges2.shape[0] > 0:
                    new_edges2 = new_edges2.loc[new_edges2['p1'] != new_edges2['p2']]

                rewired_student_to_student_edges = pd.concat([
                    student_to_student_edges.loc[inds_to_keep, ['p1', 'p2', 'beta']], # Keep these
                    new_edges,   # From completing stubs
                    new_edges2]) # From redrawing self loops

                print(f'During rewiring, the number of student-student edges went from {student_to_student_edges.shape[0]} to {rewired_student_to_student_edges.shape[0]}')

                other_edges = edges_this_school.loc[ (~edges_this_school['p1_student']) | (~edges_this_school['p2_student']) ]
                rewired_edges_this_school = pd.concat([rewired_student_to_student_edges, other_edges])
                school_contacts.append(rewired_edges_this_school)

        if len(school_contacts) > 0:
            all_school_contacts = pd.concat(school_contacts)
            sim.people.contacts['s'] = cv.Layer().from_df(all_school_contacts)

    def apply(self, sim):
        pass



#%% Running
def create_run_sim(sconf, n_sims, run_config):
    ''' Create and run the actual simulations '''

    label = f'sim {sconf.count} of {n_sims}'
    print(f'Creating and running {label}...')

    T = sc.tic()
    n_reps = cfg.sweep_pars.n_reps
    n_pops = cfg.sweep_pars.n_pops
    if n_pops is None: # Backup value so both don't have to always be set
        print(f'n_pops not supplied, resetting to n_reps={n_reps}')
        n_pops = n_reps
    if n_pops < n_reps:
        print(f'Note: you are running with n_reps={n_reps} repetitions, but only n_pops={n_pops}: populations will be resampled')
    sim = cr.create_sim(sconf.sim_pars, folder=None, max_pop_seeds=n_pops, label=label)

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
    if run_cfg['parallel']: # pragma: no cover
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

    config.sim_pars['prognoses'] = sc.dcp(prog)

    return config


def alternate_susceptibility(config, key, value):
    print(f'Building alternate symptomaticity {key}={value}')
    if not value: # Only build if value is True
        return config
    if 'prognoses' in config.sim_pars:
        prog = config.sim_pars['prognoses']
    else:
        pars = cv.make_pars(set_prognoses=True, prog_by_age=True, **config.sim_pars)
        prog = pars['prognoses']

    # Source: Susceptibility to SARS-CoV-2 Infection Among Children and Adolescents Compared With AdultsA Systematic Review and Meta-analysis
    ages = prog['age_cutoffs']
    sus_ORs = prog['sus_ORs']
    sus_ORs[ages<20] = 0.56 #  In this meta-analysis, there is preliminary evidence that children and adolescents have lower susceptibility to SARS-CoV-2, with an odds ratio of 0.56 for being an infected contact compared with adults.
    prog['sus_ORs'] = sus_ORs

    config.sim_pars['prognoses'] = sc.dcp(prog)

    return config


def loc_func(config, key, value):
    print(f'Building location {key}={value}')
    if not value: # Only build if value is True
        return config

    config.sim_pars['location'] = value
    return config
