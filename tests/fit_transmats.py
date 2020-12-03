import scipy.optimize as spo
import os
import sciris as sc
import covasim as cv
from risk_evaluation import create_sim as cs
import covasim_controller as cvc

force_run = False
cachefn = 'sim.obj'
pop_size = 500_000
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1,
    #'end_day': '2020-09-05', # TEMP: shorter duration
}

if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)
    sim.pars['interventions'] = [] # Remove interventions

    sim.run()
    sim.save(cachefn, keep_people=True)
else:
    sim = cv.load(cachefn)

inds = ~sim.people.susceptible
print(f'There were {sum(inds)} exposures')

e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]
ei = cvc.TransitionMatrix(e_to_i, 3)
ei.fit()
sc.saveobj('EI.obj', ei)
fig = ei.plot()

etoi_fn = 'EtoI.png'
print(f'Saving E-->I figure to {etoi_fn}')
fig.savefig(etoi_fn, dpi=300)


i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]
ir = cvc.TransitionMatrix(i_to_r, 7)
ir.fit()
sc.saveobj('IR.obj', ir)
fig = ir.plot()

itor_fn = 'ItoR.png'
print(f'Saving I-->R figure to {itor_fn}')
fig.savefig(itor_fn, dpi=300)
