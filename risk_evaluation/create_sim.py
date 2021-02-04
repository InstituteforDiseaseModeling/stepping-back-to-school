'''
Main script to create a simulation, the base on which analysis is conducted.
Used in calibration and other downstream activities.
'''

import os
import covasim as cv
import sciris as sc
import covasim_schools as cvsch
import config as cfg


def define_pars(which='best', kind='default', ):
    ''' Define the parameter best guesses and bounds -- used for calibration '''

    pardata = {}
    if kind in ['default', 'both']:
        # The following three parameters are adjusted during calibration.
        pardata.update(dict(
            pop_infected = [200, 100, 300],
            change_beta=[0.6, 0.45, 0.75],
            symp_prob=[0.3, 0.1, 0.5],
        ))

    output = {}
    for key,arr in pardata.items():
        if which == 'best':
            output[key] = arr[0]
        elif which == 'bounds':
            output[key] = arr[1:3]

    return output


def create_sim(params=None, folder=None, popfile_stem=None, max_pop_seeds=5, strategy='clustered', load_pop=True, save_pop=False, people=None,
               label=None, cfg=cfg, **kwargs):
    '''
    Create the simulation for use with schools. This is the main function used to
    create the sim object.

    Args:
        params (dict): the parameters to use for the simulation
        folder (str): where to look for the population file
        popfile_stem (str): filename of population file, minus random seed (which gets added)
        children_equally_sus (bool): whether children should be equally susceptible as adults (for sensitivity)
        alternate_symptomaticity (bool): whether to use symptoms by age from Table 1 in https://arxiv.org/pdf/2006.08471.pdf
        max_pop_seeds (int): maximum number of populations to generate (for use with different random seeds)
        strategy (str): the cohorting strategy to use
        load_pop (bool): whether to load people from disk (otherwise, use supplied or create afresh)
        save_pop (bool): if a population is being generated, whether to save
        people (People): if supplied, use instead of loading from file
        label (str): a name for the simulation
        kwargs (dict): merged with params

    Returns:
        A sim instance
    '''

    # Handle parameters and merge together different sets of defaults

    default_pars = dict(
        pop_scale      = 1,
        pop_type       = 'synthpops',
        rescale        = False, # True causes problems
        start_day      = '2020-11-01',
        end_day        = '2021-04-30',
        rand_seed      = 1,
    )

    p = sc.objdict(sc.mergedicts(default_pars, define_pars(which='best', kind='both'), params, kwargs)) # Get default parameter values
    if 'pop_size' not in p:
        raise Exception('You must provide "pop_size" to create_sim')
    pop_size = p.pop_size

    #%% Define interventions
    symp_prob = p.pop('symp_prob')
    change_beta = p.pop('change_beta')

    tp_pars = dict(
        symp_prob = symp_prob,
        asymp_prob = 0.0035, #0.0022, Increased to represent higher testing volume
        symp_quar_prob = symp_prob,
        asymp_quar_prob = 0.001,
        test_delay = 2.0,
    )

    ct_pars = dict(
        trace_probs = {'w': 0.1, 'c': 0, 'h': 0.9, 's': 0.8}, # N.B. 's' will be ignored if using the Schools class
        trace_time  = {'w': 2,   'c': 0, 'h': 1,   's': 2},
    )

    cb_pars = dict(
        changes=change_beta,
        layers=['w', 'c'],
        label='NPI_work_community',
    )

    ce_pars = dict(
        changes=0.65,
        layers=['w', 'c'],
        label='close_work_community'
    )

    # Define Covasim interventions
    interventions = [
        cv.test_prob(start_day=p.start_day, **tp_pars),
        cv.contact_tracing(start_day=p.start_day, **ct_pars),
        cv.change_beta(days=p.start_day, **cb_pars),
        cv.clip_edges(days=p.start_day, **ce_pars),
        # N.B. Schools are not closed in create_sim, must be handled outside this function
    ]
    for interv in interventions:
        interv.do_plot = False


    #%% Handle population -- NB, although called popfile, might be a People object
    if load_pop: # Load from disk -- normal usage
        pop_seed = p.rand_seed % max_pop_seeds
        popfile = cvsch.pop_path(popfile=None, location=cfg.pop_pars.location, folder=cfg.paths.inputs, strategy=strategy, n=pop_size, rand_seed=pop_seed)
        if os.path.exists(popfile):
            print(f'Loading population from {popfile}')
        else:
            errormsg = f'Popfile "{popfile}" does not exist; run "python create_pops.py" to generate'
            raise FileNotFoundError(errormsg)
    elif people is not None: # People is supplied; use that
        popfile = people
        print('Note: using supplied people')
    else: # Generate
        print('Note: population not supplied; regenerating...')
        popfile = cvsch.make_population(pop_size=p.pop_size, rand_seed=p.rand_seed, max_pop_seeds=max_pop_seeds, do_save=False)

    # Create sim
    sim = cv.Sim(p, popfile=popfile, load_pop=True, label=label, interventions=interventions)

    return sim
