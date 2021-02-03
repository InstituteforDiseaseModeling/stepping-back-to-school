'''
Script for quick and dirty single run of a school-based intervention
TODO: SCRIPT IS OUT OF DATE - NEEDS FIXING!
'''

import os
import sciris as sc
import covasim_schools as cvsch
import covasim_controller as cvc
from risk_evaluation import create_sim as cs
from risk_evaluation.scenarios import generate_scenarios, generate_screening
#import matplotlib as mplt
import matplotlib.pyplot as plt
import covasim as cv

# Global plotting styles
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'Roboto Condensed'

load = False
# NOTE: The following may be bypassed below by hard-coded pop_size and folder
bypass = True
folder = '../risk_evaluation/v2020-12-02'
if bypass:
    pop_size = int(20e3)
else:
    pop_size = int(2.25e5) #50e3 

def test_controller(do_plot=False):
    scen = generate_scenarios()['with_countermeasures']#['all_remote']#['all_hybrid']
    testing = generate_screening()['None']#['Antigen every 1w, PCR f/u']
    for stype, spec in scen.items():
        if spec is not None:
            spec['testing'] = testing
    scen['es']['verbose'] = scen['ms']['verbose'] = scen['hs']['verbose'] = True

    params = {
        'rand_seed': 0,
        'pop_infected': 400,
        'change_beta': 0.5,
        'symp_prob': 0.1
    }

    if bypass: # BYPASS option -- create small population on the fly
        sim = cs.create_sim(params, pop_size=pop_size, load_pop=False)
    else: # Otherwise, load full population from disk
        sim = cs.create_sim(params, pop_size=pop_size, folder=folder, verbose=0.1)

    sm = cvsch.schools_manager(scen)
    sim['interventions'] += [sm]

    targets = {
        'cases':         200, # per 100k over 2-weeks, from DOH website
        #'re':           1.0,
        #'prevalence':   0.003, # KC image from Niket in #nowcasting-inside-voice dated 11/02 using data through 10/23
        'yield':        0.029, # 2.4% positive
        'tests':        225,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
    }

    if load:
        betat = sc.loadobj('betat.obj')
        ctr = cvc.Controller(targets, betat=betat)
    else:
        ctr = cvc.Controller(targets, gain=0.05)
    sim['interventions'] += [ctr]

    if load:
        msim = cv.MultiSim(sim, n_runs=10, reseed=False)
    else:
        msim = cv.MultiSim(sim, n_runs=1)
    msim.run()

    fig, axv = plt.subplots(2,1, figsize=(5,3))
    for sim in msim.sims:
        ctr = sim['interventions'][-1]
        if not load:
            sc.saveobj('betat.obj', ctr.betat)
        axv[0].plot(sim.results['date'], 14*sim.results['new_diagnoses'].values/pop_size * 100000)
        axv[0].axhline(y=targets['cases'], color='c', ls=':')
        axv[1].plot(sim.results['date'], ctr.betat)
    fig.savefig('Controlled.png', dpi=300)

    #sim.save('test.sim')
    #cv.savefig('sim.png')

    return msim


if __name__ == '__main__':
    msim = test_controller(do_plot=True)
    msim.save('msim.obj')
