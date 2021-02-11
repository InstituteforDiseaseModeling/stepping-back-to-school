'''
Outbreak analysis to sweep in-school transmissibility
'''

import sys
import numpy as np
import school_tools as sct

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import covasim as cv

class vaccine(cv.Intervention):

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


if __name__ == '__main__':
    # Settings
    args = sct.config.process_inputs(sys.argv)

    # 63% efficacy against infection and 34% efficacy against symptoms _given infection_ are estimates from personal communication with Mike Famulare
    vx = vaccine(rel_sus_mult=1-.63, symp_prob_mult=1-0.34, teacher_cov=0.8, staff_cov=0.8, student_cov=0)

    sweep_pars = {
        'n_prev': 0, # No controller
        'school_start_date': '2021-02-01',
        'school_seed_date': '2021-02-01',
        'vaccine': {'None':None, 'Vx All Staff':[vx]},
    }

    pop_size = sct.config.sim_pars.pop_size
    sim_pars = {
        'pop_infected': 0, # Do not seed
        'pop_size': pop_size,
        'start_day': '2021-01-31',
        'end_day': '2021-08-31',
        'beta_layer': dict(w=0, c=0), # Turn off work and community transmission
    }

    npi_scens = {x:{'beta_s': 1.5*x} for x in np.linspace(0, 2, 10)}
    levels = [{'keyname':'In-school transmission multiplier', 'level':npi_scens, 'func':'screenpars_func'}]

    xvar = 'In-school transmission multiplier'
    huevar = 'Vaccination'

    # Create and run
    mgr = sct.Manager(name='OutbreakVaccine', sweep_pars=sweep_pars, sim_pars=sim_pars, levels=levels)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    g = analyzer.outbreak_size_stacked_distrib(xvar, rowvar=None, ext=None, height=6, aspect=2)

    # Plots
    g = analyzer.outbreak_multipanel(xvar, ext=None, jitter=0.2, values=None, legend=False, height=12, aspect=1.0) # height=10, aspect=0.7,

    analyzer.exports_reg(xvar, huevar)
    analyzer.outbreak_reg_facet(xvar, huevar, ext='ppt')
    exit()


    #analyzer.outbreak_reg_by_stype(xvar, height=6, aspect=1.4, ext='ppt', nboot=50, legend=True)
    #analyzer.outbreak_size_plot(xvar) #xvar, rowvar=None, ext=None, height=6, aspect=1.4, scatter=True, jitter=0.012
    analyzer.cum_incidence(colvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()


    mgr.tsplots()
