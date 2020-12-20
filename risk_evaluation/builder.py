import covasim_schools as cvsch
import testing_scenarios as t_s # From the local folder
import covasim_controller as cvc
import sciris as sc

pole_loc = 0.35

def build(prev_levels, scenario_keys, testing_keys, scen_pars, sim_pars, n_reps, folder):
    # Build simulation configuration
    sc.heading('Creating sim configurations...')
    sim_configs = []
    count = -1

    start_day = '2021-02-01' # first day of school
    all_scenarios = t_s.generate_scenarios(start_day) # Can potentially select a subset of scenarios
    scenarios = {k:v for k,v in all_scenarios.items() if k in scenario_keys}

    all_testing = t_s.generate_testing(start_day) # Potentially select a subset of testing
    testing = {k:v for k,v in all_testing.items() if k in testing_keys}

    # These come from fit_transmats
    ei = sc.loadobj('EI.obj')
    ir = sc.loadobj('IR.obj')

    for prev in prev_levels:
        for skey, base_scen in scenarios.items():
            for tkey, test in testing.items():
                for ikey, scen_par in scen_pars.items():
                    for pkey, sim_par in sim_pars.items():
                        for eidx in range(n_reps):
                            count += 1
                            p = sc.dcp(sim_par)
                            p['rand_seed'] = eidx # np.random.randint(1e6)

                            sconf = sc.objdict(count=count, sim_pars=p, pop_size=p['pop_size'], folder=folder)

                            # Add controller
                            seir = cvc.SEIR(p['pop_size'], ei.Mopt, ir.Mopt, ERR=1, beta=0.365, Ipow=0.925)
                            targets = dict(infected= prev * p['pop_size']) # prevalence target
                            ctr = cvc.controller_intervention(seir, targets, pole_loc=pole_loc, start_day=1)

                            # Modify base_scen with testing intervention
                            this_scen = sc.dcp(base_scen)
                            for stype, spec in this_scen.items():
                                if spec is not None:
                                    spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools
                                    spec.update(scen_par)

                            sm = cvsch.schools_manager(this_scen)

                            sconf.update(dict(
                                label = f'{prev} + {skey} + {tkey} + {ikey} + {pkey} + rep{eidx}',
                                prev = prev,
                                skey = skey,
                                tkey = tkey,
                                ikey = ikey,
                                pkey = pkey,
                                eidx = eidx,
                                test = test,
                                this_scen = this_scen,
                                sm = sm,
                                ctr = ctr,
                            ))

                            sim_configs.append(sconf)

    print(f'Done: {len(sim_configs)} configurations created')
    return sim_configs

