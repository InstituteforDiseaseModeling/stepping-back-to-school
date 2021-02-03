import os
import sciris as sc
import numpy as np
import covasim as cv
import covasim.utils as cvu
import covasim.misc as cvm
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power as mp
from risk_evaluation import create_sim as cs
from risk_evaluation import config as cfg
import covasim_controller as cvc
np.set_printoptions(suppress=True)

cachefn = 'sim_controlled.obj'
force_run = True # Because the control might be different each time

pop_size = 100_000 #500_000
EI_ref = 0.002 * pop_size # prevalence target
pole_loc = 0.3 # 0.35
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1,
    #'end_day': '2020-09-20',
}

targets = {
    #'cases':        200, # per 100k over 2-weeks, from DOH website
    #'re':           1.0,
    #'prevalence':   0.003, # KC image from Niket in #nowcasting-inside-voice dated 11/02 using data through 10/23
    #'yield':        0.029, # 2.4% positive
    #'tests':        225,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
    'infected':           EI_ref
}

# These come from fit_transmats
ei = sc.loadobj(cfg.paths.ei)
ir = sc.loadobj(cfg.paths.ir)

ref = cvc.ReferenceTrajectory(targets)

def run(n_seeds, n_days, K, ref):
    seir.reset(n_seeds, n_days)

    N = seir.A.shape[0]
    inds = np.r_[1:N-2, N-1]

    for k in range(n_days):
        if k < 15: # start day of controller intervention
            seir.step()
            #seir.X[-1,k+1] = 0 # Erase error
            seir.X[-1,:] = 0 # Erase error
        else:
            Xu = seir.X[inds,k]
            u = -np.dot(K, Xu)
            #u = np.median([u,0,xs]) # !!!
            seir.step(u, ref.get(k))

    return seir.finalize()


seir = cvc.SEIR(pop_size, ei.Mopt, ir.Mopt, ERR=1, beta=0.365, Ipow=0.925)

# Covasim
if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)

    ctr = cvc.controller_intervention(seir, targets, pole_loc=pole_loc)
    #sim.pars['interventions'] = [ctr] # Remove other interventions (hopefully not necessary!)
    sim.pars['interventions'].append(ctr) # Remove other interventions (hopefully not necessary!)
    sim.run()

    sim.save(cachefn, keep_people=True)
else:
    sim = cv.load(cachefn)

N = sim.scaled_pop_size # don't really care about N vs alive...
#alive = N - sim.results['cum_deaths']
S = N - sim.results['cum_infections'].values
E = sim.results['n_exposed'].values - sim.results['n_infectious'].values
I = sim.results['n_infectious'].values
R = sim.results['cum_recoveries']

# SEIR
ct = cvc.Controller(seir, pole_loc=pole_loc)
n_days = sim.pars['n_days'] # cvm.daydiff('2020-09-01', '2021-01-31') #
seir_ret = run(params['pop_infected'], n_days, ct.K, ref)

fig, axv = plt.subplots(2,2, figsize=(16,10))
ax = axv[0,0] # Top left
ax.plot(sim.results['date'], np.vstack([S, E, I, R]).T)

Xerr = seir_ret['e']
Xs = seir_ret['S']
Xe = seir_ret['E']
Xi = seir_ret['I']
Xr = seir_ret['R']
Y = seir_ret['Y']

xx = np.vstack([Xs, Xe, Xi, Xr])
ax.set_prop_cycle(None)
ax.plot(sim.results['date'], xx.T, ls='--')
ax.legend(['Susceptible', 'Exposed', 'Infectious', 'Recovered'])

ax = axv[0,1] # Top right
ax.scatter(S*I/N, sim.results['new_infections'], c=range(len(S)))
SI = Xs*Xi/N
ax.scatter(SI[:-1], -np.diff(Xs), c=range(len(SI[:-1])), marker='x')
ax.set_xlabel('S*I/N')
ax.set_ylabel('New infections')

ax = axv[1,1] # Bottom right
ax.scatter(E, I, c=range(len(S)))
ax.set_prop_cycle(None)
ax.scatter(Xe, Xi, c=range(len(S)), marker='x')
ax.set_xlabel('Exposed')
ax.set_ylabel('Infectious')
ax.legend(['Covasim', 'SEIR'])
fig.tight_layout()

ax = axv[1,0] # Bottom left
ax.plot(sim.results['date'], E+I, 'k')
ax.plot(sim.results['date'], Y, 'k--')
ax.axhline(y=EI_ref, color='r', zorder=-1)
ax.legend(['Covasim Exposed + Infectious', 'SEIR E+I', 'Reference E+I'])

fn = 'ControlledCovasim.png'
print(f'Saving figure to {fn}')
fig.savefig(fn, dpi=300)
