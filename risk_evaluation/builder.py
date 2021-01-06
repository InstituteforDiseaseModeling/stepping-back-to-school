import covasim_schools as cvsch
import covasim_controller as cvc
import covasim as cv
import sciris as sc
import scenarios as scn

class Config:
    def __init__(self, sim_pars=None, label=None):
        self.label = label # TODO: From tags?
        self.tags = {}

        # TODO: Seems necessary to have access to e.g. prognosis parameters, but can work around
        self.sim_pars = sim_pars #cv.make_pars(set_prognoses=True, prog_by_age=True, **sim_pars)
        self.school_config = None
        self.interventions = []
        self.count = 0

    def __repr__(self):
        return '\n' + '-'*80 + '\n'+ f'Configuration {self.label}:\n\
 * Tags: {self.tags}\n\
 * Pars: {self.sim_pars}\n\
 * School config: {self.school_config}\n\
 * Num interventions: {len(self.interventions)}'


class Builder:
    def __init__(self, sim_pars, schcfg_keys, screen_keys, school_start_date):
        self.configs = [Config(sim_pars=sim_pars)]

        # These come from fit_transmats - don't like loading multiple times
        self.ei = sc.loadobj('EI.obj')
        self.ir = sc.loadobj('IR.obj')

        all_scen = scn.generate_scenarios(school_start_date) # Can potentially select a subset of scenarios
        scens = {k:v for k,v in all_scen.items() if k in schcfg_keys}
        self.add_level('scen_key', scens, self.scen_func)

        all_screenings = scn.generate_screening(school_start_date) # Potentially select a subset of diagnostic screenings
        screens = {k:v for k,v in all_screenings.items() if k in screen_keys}
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

