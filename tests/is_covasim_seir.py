import os
import sciris as sc
import numpy as np
import covasim as cv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import covasim_controller as cvc
from risk_evaluation import create_sim as cs

# Global plotting styles
plt.rcParams['font.size'] = 6
plt.rcParams['font.family'] = 'Roboto Condensed'
plt.rcParams['lines.linewidth'] = 0.7

cachefn = 'sim_100k.obj'
force_run = True

pop_size = 100_000
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1,
    #'end_day': '2020-09-05', # TEMP: shorter duration
}

# These come from fit_transmats
ei = sc.loadobj('EI.obj')
ir = sc.loadobj('IR.obj')

EI = ei.Mopt
IR = ir.Mopt

if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)
    sim.pars['interventions'] = [] # Remove interventions

    sim.run()
    sim.save(cachefn, keep_people=True)
else:
    sim = cv.load(cachefn)

seeds = sim.results['n_exposed'][0]
seir = cvc.SEIR(pop_size, EI, IR, ERR=1, beta=0.365, Ipow=0.925) # 0.365, 0.94
seir_results = seir.run(seeds, sim.pars['n_days'])

N = sim.scaled_pop_size # don't really care about N vs alive...
#alive = N - sim.results['cum_deaths']
S = N - sim.results['cum_infections'].values
E = sim.results['n_exposed'].values - sim.results['n_infectious'].values
I = sim.results['n_infectious'].values
R = sim.results['cum_recoveries']

fig = plt.figure(constrained_layout=True, figsize=(5,3))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(sim.results['date'], np.vstack([S, E, I, R]).T)

xx = np.vstack([seir_results['S'], seir_results['E'], seir_results['I'], seir_results['R']])
ax1.set_prop_cycle(None)
ax1.plot(sim.results['date'], xx.T, ls='--')
ax1.legend(['Susceptible', 'Exposed', 'Infectious', 'Recovered'])

ax2.scatter(S*I/N, sim.results['new_infections'], c=range(len(S)), s=3)
SI = seir_results['S']*seir_results['I']/N
ax2.scatter(SI[:-1], -np.diff(seir_results['X'][0]), c=range(len(SI[:-1])), s=3, marker='x')
ax2.set_xlabel('S*I/N')
ax2.set_ylabel('New infections')

ax3.scatter(E, I, c=range(len(S)), s=3)
ax3.set_prop_cycle(None)
ax3.scatter(seir_results['E'], seir_results['I'], c=range(len(S)), s=3, marker='x')
ax3.set_xlabel('Exposed')
ax3.set_ylabel('Infectious')
ax3.legend(['Covasim', 'SEIR'])

fig.savefig('is_covasim_seir.png', dpi=300)
