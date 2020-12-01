import os
import seaborn as sns
import numpy as np
import covasim as cv
import covasim.utils as cvu
import matplotlib.pyplot as plt
from risk_evaluation import create_sim as cs
from fit_distrib import coxian, full, plot_dur

cachefn = 'sim.obj'
force_run = False

pop_size = 500_000
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1
}

#EI = coxian(np.array([0.4950429 , 0.51759924, 0.5865702 ]))
EI = full([0.55914219, 0.24590932, 0.19970447, 0.18187704, 0.80029553, 0.40464155])[:-1,:-1]

#IR = coxian(np.array([0.66154341, 0.61511552, 1.52192331, 0.69897356, 0.6143495, 0.61457423, 0.70117798]))
IR = full([   4.28858941e-01, 5.71137033e-01, 1.28970286e-01, 4.02251590e-06,
              8.71026730e-01, 4.49389364e-01,-1.82612245e-10, 2.93319138e-06,
              5.50610595e-01, 1.56866431e-01, 3.50592514e-10, 1.82838248e-08,
              1.20094344e-08, 8.43133423e-01, 3.17818904e-01, 2.32791592e-09,
              2.82724627e-08, 2.78078486e-08, 1.05252597e-07, 7.73989524e-06,
              9.99993009e-01, 3.35792591e-10, 2.49896534e-09,-2.07656633e-10,
              4.33997639e-08, 6.82173297e-01, 6.98223414e-06, 3.50208520e-01])[:-1,:-1]
nEI = EI.shape[0]
nIR = IR.shape[0]

if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)
    sim.pars['interventions'] = [] # Remove interventions

    print('TEMP: shorter duration')
    sim.pars['rand_seed'] = 0
    sim.pars['end_day'] = '2020-09-05'

    sim.run()
    sim.save(cachefn, keep_people=True)
else:
    sim = cv.load(cachefn)




inds = ~sim.people.susceptible
print(f'There were {sum(inds)} exposures')
e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]
i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]
fig, axv = plt.subplots(1,2,figsize=(16,10))
plot_dur(e_to_i, EI, axv[0])
plot_dur(i_to_r, IR, axv[1])

#print('WARNING: 5x number of seeds!')
seeds = 1 * sim.results['n_exposed'][0]
X = run_SEIR(seeds, sim.pars['n_days'])

N = sim.scaled_pop_size # don't really care about N vs alive...
#alive = N - sim.results['cum_deaths']
S = N - sim.results['cum_infections'].values
E = sim.results['n_exposed'].values - sim.results['n_infectious'].values
I = sim.results['n_infectious'].values
R = sim.results['cum_recoveries']

fig, axv = plt.subplots(1,3, figsize=(16,10))
#axv[0].plot(sim.results['date'], np.vstack([S, E, I, R]).T)
axv[0].plot(sim.results['date'], np.vstack([E, I, R]).T)

Xs = X[0,:]
Xe = np.sum(X[range(1,nEI+1),:], axis=0)
Xi = np.sum(X[range(nEI+1, nEI+nIR+1),:], axis=0)
Xr = X[-1,:]

#xx = np.vstack([Xs, Xe, Xi, Xr])
xx = np.vstack([Xe, Xi, Xr])
axv[0].set_prop_cycle(None)
axv[0].plot(sim.results['date'], xx.T, ls='--')
axv[0].legend(['Exposed', 'Infectious', 'Recovered'])

axv[1].scatter(S*I/N, sim.results['new_infections'], c=range(len(S)))
SI = Xs*Xi/N
axv[1].scatter(SI[:-1], -np.diff(X[0]), c=range(len(SI[:-1])), marker='x')
axv[1].set_xlabel('S*I/N')
axv[1].set_ylabel('New infections')

axv[2].scatter(E, I, c=range(len(S)))
axv[2].set_prop_cycle(None)
axv[2].scatter(Xe, Xi, c=range(len(S)), marker='x')
axv[2].set_xlabel('Exposed')
axv[2].set_ylabel('Infectious')
axv[2].legend(['Covasim', 'SEIR'])
fig.tight_layout()

sim.plot()

plt.show()

'''
cum_infections
cum_infectious
cum_tests
cum_diagnoses
cum_recoveries
cum_symptomatic
cum_severe
cum_critical
cum_deaths
cum_quarantined
new_infections
new_infectious
new_tests
new_diagnoses
new_recoveries
new_symptomatic
new_severe
new_critical
new_deaths
new_quarantined
n_susceptible
n_exposed
n_infectious
n_symptomatic
n_severe
n_critical
n_diagnosed
n_quarantined
n_alive
prevalence
incidence
r_eff
doubling_time
test_yield
rel_test_yield
date
t
'''
