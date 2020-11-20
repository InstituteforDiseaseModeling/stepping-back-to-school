import os
import seaborn as sns
import numpy as np
import covasim as cv
import covasim.utils as cvu
import matplotlib.pyplot as plt
from risk_evaluation import create_sim as cs
from fit_distrib import coxian, plot_dur

cachefn = 'sim.obj'
force_run = False

pop_size = 500_000
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1
}

EI = coxian(np.array([0.4950429 , 0.51759924, 0.5865702 ]))
IR = coxian(np.array([0.66154341, 0.61511552, 1.52192331, 0.69897356, 0.6143495, 0.61457423, 0.70117798]))
nEI = EI.shape[0]
nIR = IR.shape[0]

if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)
    sim.pars['interventions'] = [] # Remove interventions
    sim.run()
    sim.save(cachefn, keep_people=True)
else:
    sim = cv.load(cachefn)

def step(t, X, A, B, C, beta):
    x = np.matrix(X).T

    xs = x[0]
    xi = np.sum(x[np.arange(nEI+1, nEI+nIR+1)])
    xi += np.sum(x[np.arange(nEI+1, nEI+4)]) # Double infectivity early

    u = beta * xs*np.power(xi, 1) / pop_size
    y = C*x

    new_x = A*x + B*u

    return np.squeeze(np.asarray(new_x))

def run_SEIR(seeds, n_days):
    nu1 = 0.4
    nu2 = 0.55
    nu3 = 0.55
    gamma1 = 0.15
    gamma2 = 0.2
    dt = 1
    beta = 0.185

    belowEI = np.zeros((nIR, nEI))
    belowEI[0,-1] = 1-EI[-1,-1]

    belowIR = np.zeros(nIR)
    belowIR[-1] = 1-IR[-1,-1]

    # Full SEIR Dynamics
    Af = np.block([ [1,                 np.zeros(nEI),     np.zeros(nIR),       0],
                    [np.zeros((nEI,1)), EI,                np.zeros((nEI,nIR)), np.zeros((nEI,1))],
                    [np.zeros((nIR,1)), belowEI,           IR,                  np.zeros((nIR,1))],
                    [0,                 np.zeros((1,nEI)), belowIR,             1] ])


    Bf = np.zeros((2+nEI+nIR,1))
    Bf[0] = -1
    Bf[1] = 1

    Cf = np.zeros((1,2+nEI+nIR))
    Cf[1:-1] = 1

    n_days = 152
    X = np.zeros((2+nEI+nIR,n_days+1))
    X[0,0] = pop_size-10*seeds
    X[1,0] = 10*seeds
    for k in range(n_days):
        X[:,k+1] = step(k*dt, X[:,k], Af, Bf, Cf, beta)

    return X

inds = ~sim.people.susceptible
print(f'There were {sum(inds)} exposures')
e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]
i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]
fig, axv = plt.subplots(1,2,figsize=(16,10))
plot_dur(e_to_i, EI, axv[0])
plot_dur(i_to_r, IR, axv[1])


X = run_SEIR(sim.results['n_exposed'][0], sim.pars['n_days'])

N = sim.scaled_pop_size # don't really care about N vs alive...
#alive = N - sim.results['cum_deaths']
S = N - sim.results['cum_infections'].values
E = sim.results['n_exposed'].values - sim.results['n_infectious'].values
I = sim.results['n_infectious'].values
R = sim.results['cum_recoveries']

fig, axv = plt.subplots(1,3, figsize=(16,10))
axv[0].plot(sim.results['date'], np.vstack([S, E, I, R]).T)

Xs = X[0,:]
Xe = np.sum(X[range(1,nEI+1),:], axis=0)
Xi = np.sum(X[range(nEI+1, nEI+nIR+1),:], axis=0)
Xr = X[-1,:]

xx = np.vstack([Xs, Xe, Xi, Xr])
axv[0].set_prop_cycle(None)
axv[0].plot(sim.results['date'], xx.T, ls='--')
axv[0].legend(['Susceptible', 'Exposed', 'Infectious', 'Recovered'])

axv[1].scatter(S*I/N, sim.results['new_infections'], c=range(len(S)))
SI = Xs*Xi/N
axv[1].scatter(SI[:-1], -np.diff(X[0]), c=range(len(SI[:-1])), marker='x')
axv[1].set_xlabel('S*I/N')
axv[1].set_ylabel('New infections')

axv[2].scatter(E, I, c=range(len(S)))
axv[0].set_prop_cycle(None)
axv[2].scatter(Xe, Xi, c=range(len(S)), marker='x')
axv[2].set_xlabel('Exposed')
axv[2].set_ylabel('Infectious')
axv[2].legend(['Covasim', 'SEIR'])
fig.tight_layout()

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
