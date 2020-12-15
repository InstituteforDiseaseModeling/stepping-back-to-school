'''
This file defines the different scenarios for use with run_scenarios.
'''

import datetime as dt
import sciris as sc

def scenario(es, ms, hs):
    return {
        'pk': None,
        'es': sc.dcp(es),
        'ms': sc.dcp(ms),
        'hs': sc.dcp(hs),
        'uv': None,
    }

def generate_scenarios(start_day='2020-11-02'):
    ''' Generate scenarios (dictionaries of parameters) for the school intervention '''

    # Increase beta (multiplier) in schools from default of 0.6 to 1.5.  This achieves a R0 in schools of approximately 1.6, a modeling assumption that is consistent with global outbreaks in schools that have had limited countermeasures such as Israel (after masks were removed due to heat).
    base_beta_s = 1.5

    #print('WARNING: SETTING base_beta_s to HIGH value!')
    #base_beta_s = 10000

    scns = sc.odict()

    normal = {
        'start_day': start_day,
        'schedule': 'Full',
        'screen_prob': 0,
        'test_prob': 0, # Amongst those who screen positive
        'screen2pcr': 3, # Days from screening to receiving PCR results
        'trace_prob': 0, # Fraction of newly diagnosed index cases who are traced
        'quar_prob': 0, # Of those reached by contact tracing, this fraction will quarantine
        'ili_prob': 0.002, # Daily ili probability equates to about 10% incidence over the first 3 months of school
        'beta_s': base_beta_s, # No NPI
        'testing': None,
        'save_trees': True,
    }
    scns['as_normal'] = scenario(es=normal, ms=normal, hs=normal)


    full_with_countermeasures = {
        'start_day': start_day,
        'schedule': 'Full',
        'screen_prob': 0.9,
        'test_prob': 0.5, # Amongst those who screen positive
        'screen2pcr': 3, # Days from screening to receiving PCR results
        'trace_prob': 0.75, # Fraction of newly diagnosed index cases who are traced
        'quar_prob': 0.75, # Of those reached by contact tracing, this fraction will quarantine
        'ili_prob': 0.002, # Daily ili probability equates to about 10% incidence over the first 3 months of school
        'beta_s': 0.75 * base_beta_s, # 25% reduction due to NPI
        'testing': None,
        'save_trees': True,
    }

    full_with_optimistic_countermeasures = {
        'start_day': start_day,
        'schedule': 'Full',
        'screen_prob': 0.5,
        'test_prob': 0.8, # Amongst those who screen positive
        'screen2pcr': 1, # Days from screening to receiving PCR results
        'trace_prob': 1.0, # Fraction of newly diagnosed index cases who are traced
        'quar_prob': 1.0, # Of those reached by contact tracing, this fraction will quarantine
        'ili_prob': 0.002, # Daily ili probability equates to about 10% incidence over the first 3 months of school
        'beta_s': 0.75 * base_beta_s, # 25% reduction due to NPI
        'testing': None,
        'save_trees': True,
    }

    # Add screening and NPI
    scns['with_countermeasures'] = scenario(es=full_with_countermeasures, ms=full_with_countermeasures, hs=full_with_countermeasures)
    scns['with_optimistic_countermeasures'] = scenario(es=full_with_optimistic_countermeasures, ms=full_with_optimistic_countermeasures, hs=full_with_optimistic_countermeasures)

    # Add hybrid scheduling
    hybrid = sc.dcp(full_with_countermeasures)
    hybrid['schedule'] = 'Hybrid'
    scns['all_hybrid'] = scenario(es=hybrid, ms=hybrid, hs=hybrid)

    # All remote
    remote = {
        'start_day': start_day,
        'schedule': 'Remote',
        'screen_prob': 0,
        'test_prob': 0,
        'screen2pcr': 3, # Days from screening to receiving PCR results
        'trace_prob': 0,
        'quar_prob': 0,
        'ili_prob': 0,
        'beta_s': 0, # NOTE: No transmission in school layers
        'testing': None,
        'save_trees': True,
    }

    scns['k5'] = scenario(es=full_with_countermeasures, ms=remote, hs=remote)

    scns['all_remote'] = scenario(es=remote, ms=remote, hs=remote)

    return scns

def generate_testing(start_day='2020-11-02'):

    one_week_ahead = (dt.datetime.strptime(start_day, '%Y-%m-%d') - dt.timedelta(days=7)).strftime('%Y-%m-%d')

    # Testing interventions to add
    PCR_1w_prior = [{
        'start_date': one_week_ahead,
        'repeat': None,
        'groups': ['students', 'teachers', 'staff'],
        'coverage': 1,
        'is_antigen': False,
        'sensitivity': 1,
        'delay': 1,
    }]

    PCR_every_4w_starting_1wprior = [{
        'start_date': one_week_ahead,
        'repeat': 28,
        'groups': ['students', 'teachers', 'staff'],
        'coverage': 1,
        'is_antigen': False,
        'sensitivity': 1,
        'delay': 1,
    }]

    PCR_every_2w_starting_1wprior = [{
        'start_date': one_week_ahead,
        'repeat': 14,
        'groups': ['students', 'teachers', 'staff'],
        'coverage': 1,
        'is_antigen': False,
        'sensitivity': 1,
        'delay': 1,
    }]

    PCR_every_1w_starting_1wprior = [{
        'start_date': one_week_ahead,
        'repeat': 7,
        'groups': ['students', 'teachers', 'staff'],
        'coverage': 1,
        'is_antigen': False,
        'sensitivity': 1,
        'delay': 1,
    }]

    PCR_every_1d_starting_1wprior = [{
        'start_date': one_week_ahead,
        'repeat': 1,
        'groups': ['students', 'teachers', 'staff'],
        'coverage': 1,
        'is_antigen': False,
        'sensitivity': 1,
        'delay': 0, # NOTE: no delay!
    }]

    Antigen_every_1w_starting_1wprior_teachersstaff_PCR_followup = [{
        'start_date': one_week_ahead,
        'repeat': 7,
        'groups': ['teachers', 'staff'], # No students
        'coverage': 1,
        'is_antigen': True,
        'symp7d_sensitivity': 0.971, # https://www.fda.gov/media/141570/download
        'other_sensitivity': 0.90, # Modeling assumption
        'specificity': 0.985, # https://www.fda.gov/media/141570/download
        'PCR_followup_perc': 1.0,
        'PCR_followup_delay': 3.0,
    }]


    Antigen_every_4w_starting_1wprior_all_PCR_followup = [{
        'start_date': one_week_ahead,
        'repeat': 28,
        'groups': ['students', 'teachers', 'staff'], # No students
        'coverage': 1,
        'is_antigen': True,
        'symp7d_sensitivity': 0.971, # https://www.fda.gov/media/141570/download
        'other_sensitivity': 0.90, # Modeling assumption
        'specificity': 0.985, # https://www.fda.gov/media/141570/download
        'PCR_followup_perc': 1.0,
        'PCR_followup_delay': 3.0,
    }]


    Antigen_every_2w_starting_1wprior_all_PCR_followup = [{
        'start_date': one_week_ahead,
        'repeat': 14,
        'groups': ['students', 'teachers', 'staff'], # No students
        'coverage': 1,
        'is_antigen': True,
        'symp7d_sensitivity': 0.971, # https://www.fda.gov/media/141570/download
        'other_sensitivity': 0.90, # Modeling assumption
        'specificity': 0.985, # https://www.fda.gov/media/141570/download
        'PCR_followup_perc': 1.0,
        'PCR_followup_delay': 3.0,
    }]

    Antigen_every_1w_starting_1wprior_all_PCR_followup = [{
        'start_date': one_week_ahead,
        'repeat': 7,
        'groups': ['students', 'teachers', 'staff'], # No students
        'coverage': 1,
        'is_antigen': True,
        'symp7d_sensitivity': 0.971, # https://www.fda.gov/media/141570/download
        'other_sensitivity': 0.90, # Modeling assumption
        'specificity': 0.985, # https://www.fda.gov/media/141570/download
        'PCR_followup_perc': 1.0,
        'PCR_followup_delay': 3.0,
    }]

    Antigen_every_2w_starting_1wprior_all_no_followup = [{
        'start_date': one_week_ahead,
        'repeat': 14,
        'groups': ['students', 'teachers', 'staff'], # No students
        'coverage': 1,
        'is_antigen': True,
        'symp7d_sensitivity': 0.971, # https://www.fda.gov/media/141570/download
        'other_sensitivity': 0.90, # Modeling assumption
        'specificity': 0.985, # https://www.fda.gov/media/141570/download
        'PCR_followup_perc': 0.0,
        'PCR_followup_delay': 0.0, # Does not matter with no PCR follow-up
    }]


    return {
        'None': None,
        'PCR 1w prior': PCR_1w_prior,
        'PCR every 4w': PCR_every_4w_starting_1wprior,
        'Antigen every 4w, PCR f/u': Antigen_every_4w_starting_1wprior_all_PCR_followup,
        'Antigen every 1w teach&staff, PCR f/u': Antigen_every_1w_starting_1wprior_teachersstaff_PCR_followup,
        'PCR every 2w': PCR_every_2w_starting_1wprior,
        'Antigen every 2w, no f/u': Antigen_every_2w_starting_1wprior_all_no_followup,
        'Antigen every 2w, PCR f/u': Antigen_every_2w_starting_1wprior_all_PCR_followup,
        'Antigen every 1w, PCR f/u': Antigen_every_1w_starting_1wprior_all_PCR_followup,
        'PCR every 1w': PCR_every_1w_starting_1wprior,
        'PCR every 1d': PCR_every_1d_starting_1wprior,
    }