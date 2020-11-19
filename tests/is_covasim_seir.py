import os
import seaborn as sns
import numpy as np
import covasim as cv
import covasim.utils as cvu
import matplotlib.pyplot as plt
from risk_evaluation import create_sim as cs

cachefn = 'sim.obj'
force_run = False

pop_size = 20e4
params = {
    'rand_seed': 0,
    'pop_infected': 1000,
    'change_beta': 0.9,
    'symp_prob': 0.1
}

if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)
    sim.run()
    sim.save(cachefn, keep_people=True)
else:
    sim = cv.load(cachefn)

def step(t, X, A, B, C, beta):
    x = np.matrix(X).T

    u = beta * x[0] * x[2] / pop_size
    y = C*x

    new_x = A*x + B*u

    return np.squeeze(np.asarray(new_x))

def run_SEIR(seeds, n_days, mu_ei, mu_ir):
    nu = 1/mu_ei
    gamma = 1/mu_ir
    beta = 0.13
    dt = 1

    # Full SEIR Dynamics
    Af = np.matrix([[ 1,   0,      0,     0],
                    [ 0, 1-nu,      0,     0],
                    [ 0,  nu, 1-gamma,     0],
                    [ 0,   0,  gamma,     1]    ])
    Bf = np.matrix([ [-1], [1], [0], [0] ])
    Cf = np.block([0,1,1,0])

    n_days = 152
    X = np.zeros((4,n_days+1))
    X[:,0] = [pop_size-seeds, seeds, 0, 0]
    for k in range(n_days):
        X[:,k+1] = step(k*dt, X[:,k], Af, Bf, Cf, beta)

    return X


inds = sim.people.exposed
print(f'There were {sum(inds)} exposures')
fig, axv = plt.subplots(1,2, figsize=(16,10))
t = np.arange(sim.pars['n_days'])
e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]
mu_ei = np.nanmean(e_to_i)
sns.distplot(e_to_i, bins=range(int(max(e_to_i))), ax=axv[0])
axv[0].axvline(x=mu_ei, c='r')
axv[0].fill_between(t, np.exp(-t/mu_ei)/mu_ei, facecolor='k', alpha=0.25)
axv[0].plot(t, np.exp(-t/mu_ei)/mu_ei, 'k')
axv[0].set_title(f'E-->I: Mean {mu_ei:.1f} days')

i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]
mu_ir = np.nanmean(i_to_r)
axv[1].axvline(x=mu_ir, c='r')
axv[1].fill_between(t, np.exp(-t/mu_ir)/mu_ir, facecolor='k', alpha=0.25)
axv[1].plot(t, np.exp(-t/mu_ir)/mu_ir, 'k')
axv[1].set_title(f'I --> R: Mean {mu_ir:.1f} days')
sns.distplot(i_to_r, bins=range(int(max(i_to_r))), ax=axv[1])


X = run_SEIR(sim.results['n_exposed'][0], sim.pars['n_days'], mu_ei, mu_ir)

N = sim.scaled_pop_size # don't really care about N vs alive...
#alive = N - sim.results['cum_deaths']
S = N - sim.results['cum_infections'].values
E = sim.results['n_exposed'].values - sim.results['n_infectious'].values
I = sim.results['n_infectious'].values
R = sim.results['cum_recoveries']

fig, axv = plt.subplots(1,2, figsize=(16,10))
axv[0].plot(sim.results['date'], np.vstack([S, E, I, R]).T)
axv[0].plot(sim.results['date'], X.T, ls='--')
axv[0].legend(['Susceptible', 'Exposed', 'Infectious', 'Recovered'])

axv[1].scatter(S*I/N, sim.results['new_infections'], c=range(len(sim.results['new_infections'])))
SI = X[0]*X[2]/N
axv[1].plot(SI[:-1], -np.diff(X[0]), '-')
axv[1].set_xlabel('S*I/N')
axv[1].set_ylabel('New infections')

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
