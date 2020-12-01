import os
import seaborn as sns
import numpy as np
import covasim as cv
import covasim.utils as cvu
import matplotlib.pyplot as plt
from fit_distrib import plot_dur
from numpy.linalg import matrix_power as mp
from risk_evaluation import create_sim as cs
import covasim_controller as cvc

plot_dur_dist = False

cachefn = 'sim_controlled.obj'
force_run = True

pop_size = 100_000 #500_000
EI_ref = 0.03 * pop_size # prevalence target
pole_loc = 0.7 # 0.3
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1,
    #'end_day': '2020-09-05',
}

targets = {
    #'cases':        200, # per 100k over 2-weeks, from DOH website
    #'re':           1.0,
    #'prevalence':   0.003, # KC image from Niket in #nowcasting-inside-voice dated 11/02 using data through 10/23
    #'yield':        0.029, # 2.4% positive
    #'tests':        225,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
    'EI':           EI_ref
}

# These come from fit_transmats
ei = sc.loadobj('EI.obj')
ir = sc.loadobj('IR.obj')

seir = cvc.SEIR(pop_size, ei.Mopt, ir.Mopt, beta=0.365, Ipow=0.925)

ct = cvc.Controller(seir)

EI = ei.Mopt
IR = ir.Mopt

nEI = ei.n
nIR = ir.n



A,B,C,K = ct.build_SEIR()

if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)

    ctr = cvc.controller_intervention(targets)
    sim.pars['interventions'] = [ctr] # Remove other interventions (hopefully not necessary!)
    sim.run()
    sim.save(cachefn, keep_people=True)
else:
    sim = cv.load(cachefn)

def step(t, X, A, B, C, K, beta):
    X = np.matrix(X).T
    integrated_error = X[0]
    x = X[1:]

    xs = x[0]
    xi = np.sum(x[np.arange(nEI+1, nEI+nIR+1)])
    xi += np.sum(x[np.arange(nEI+1, nEI+4)]) # Double early infectivity

    Xu = X[[0] + list(range(1+1, nEI+nIR+1+1)),:] # integrated error, E, I
    if t < 15:
        u = beta * xs*xi / pop_size # np.power(xi, 1)
    else:
        u = -K*Xu
    u = np.median([u,0,xs]) # !!!
    y = C*x

    new_integrated_error = integrated_error + y - EI_ref

    new_x = A*x + B*u
    new = np.vstack( [new_integrated_error, new_x] )

    return np.squeeze(np.asarray(new))


def run_SEIR(A, B, C, K, beta, seeds, n_days):
    dt = 1
    X = np.zeros((2+nEI+nIR + 1, n_days+1)) # +1 for error dynamics
    #print('WARNING: 10x number of seeds!')
    #seeds *= 10
    X[1,0] = pop_size-seeds
    X[2,0] = seeds
    for k in range(n_days):
        X[:,k+1] = step(k*dt, X[:,k], A, B, C, K, beta)

    return X

inds = ~sim.people.susceptible
print(f'There were {sum(inds)} exposures')
e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]
i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]
if plot_dur_dist:
    fig, axv = plt.subplots(1,2,figsize=(16,10))
    plot_dur(e_to_i, EI, axv[0])
    plot_dur(i_to_r, IR, axv[1])


#A,B,C,K = build_SEIR()
beta = 0.185
X = run_SEIR(A,B,C,K,beta,sim.results['n_exposed'][0], sim.pars['n_days'])

N = sim.scaled_pop_size # don't really care about N vs alive...
#alive = N - sim.results['cum_deaths']
S = N - sim.results['cum_infections'].values
E = sim.results['n_exposed'].values - sim.results['n_infectious'].values
I = sim.results['n_infectious'].values
R = sim.results['cum_recoveries']

fig, axv = plt.subplots(2,2, figsize=(16,10))
ax = axv[0,0] # Top left
ax.plot(sim.results['date'], np.vstack([S, E, I, R]).T)

Xerr = X[0,:]
Xs = X[1,:]
Xe = np.sum(X[range(1+1,nEI+1+1),:], axis=0)
Xi = np.sum(X[range(nEI+1+1, nEI+nIR+1+1),:], axis=0)
Xr = X[-1,:]

xx = np.vstack([Xs, Xe, Xi, Xr])
ax.set_prop_cycle(None)
ax.plot(sim.results['date'], xx.T, ls='--')
ax.legend(['Susceptible', 'Exposed', 'Infectious', 'Recovered'])

ax = axv[0,1] # Top right
ax.scatter(S*I/N, sim.results['new_infections'], c=range(len(S)))
SI = Xs*Xi/N
ax.scatter(SI[:-1], -np.diff(X[0]), c=range(len(SI[:-1])), marker='x')
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
ax.plot(sim.results['date'], np.asarray(C*X[1:,:])[0], 'k--')
ax.axhline(y=EI_ref, color='r', zorder=-1)
ax.legend(['Covasim Exposed + Infectious', 'SEIR E+I', 'Reference E+I'])


plt.show()
