import os
import seaborn as sns
import numpy as np
import covasim as cv
import covasim.utils as cvu
import matplotlib.pyplot as plt
from risk_evaluation import create_sim as cs
import covasim_controller as cvc
from fit_distrib import coxian, full, plot_dur
from numpy.linalg import matrix_power as mp

do_plot = True

cachefn = 'sim_controlled.obj'
force_run = False

pop_size = 100_000 #500_000
EI_ref = 0.03 * pop_size # prevalence target
pole_loc = 0.3 # 0.3
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1
}

targets = {
    'cases':         200, # per 100k over 2-weeks, from DOH website
    #'re':           1.0,
    #'prevalence':   0.003, # KC image from Niket in #nowcasting-inside-voice dated 11/02 using data through 10/23
    'yield':        0.029, # 2.4% positive
    'tests':        225,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
    'EI': EI_ref
}


#EI = coxian(np.array([0.4950429 , 0.51759924, 0.5865702 ]))
#IR = coxian(np.array([0.66154341, 0.61511552, 1.52192331, 0.69897356, 0.6143495, 0.61457423, 0.70117798]))

EI = full([0.54821612, 0.40217725, 0.24984496, 0.03756642, 0.5213049 ,
       0.40917672])

#IR = coxian(np.array([0.66154341, 0.61511552, 1.52192331, 0.69897356, 0.6143495, 0.61457423, 0.70117798]))
IR = full([ 3.40190364e-01,  3.63341049e-01,  3.35031247e-02,  2.96464203e-01,
        9.65784905e-01,  1.12059492e-02,  4.37587969e-06, -1.21861188e-07,
        9.62334305e-01,  9.75919056e-03, -6.23660031e-10,  1.64779825e-08,
        1.01696020e-02,  9.77812849e-01,  5.15078739e-02,  4.48849543e-10,
        1.53316944e-08,  1.62901373e-02,  1.24279573e-02,  9.48492111e-01,
        3.71067650e-01,  4.71039087e-09,  7.12066962e-04,  3.37529797e-12,
       -5.69699539e-10,  1.97380899e-08,  6.28932347e-01,  5.53031491e-01])


nEI = EI.shape[0]
nIR = IR.shape[0]

def design_controller(A,B,C,pole_loc):
    # Dynamics for feedback controller design
    Af = A[1:-1,1:-1]
    Bf = B[1:-1]
    Cf = C[:,1:-1]

    # Error dynamics: e_dot, E_dot, I_dot
    Ac = np.block( [[1, Cf], [np.zeros((nEI+nIR,1)), Af]] )
    Bc = np.vstack([0, Bf])

    ctrb = np.matrix(np.zeros((nEI+nIR+1, nEI+nIR+1))) # +1 for error dynamics
    ctrb[:,0] = Bc
    for i in range(nEI+nIR):
        ctrb[:,i+1] = Ac*ctrb[:,i] # columns of controllability matrix

    # Characteristic (monic) polynomial coefficients given pole locations:
    alpha = np.poly(pole_loc*np.ones(ctrb.shape[0])) # All poles at same place?

    alpha_c_F = mp(Ac,nEI+nIR+1)
    for i in range(1, nEI+nIR+1):
        alpha_c_F += alpha[i] * mp(Ac,nEI+nIR+1-i)  # Start at 1
    alpha_c_F += alpha[-1] * np.eye(nEI+nIR+1)

    last_row = np.matrix(np.zeros(nEI+nIR+1))
    last_row[:,-1] = 1

    # Ackermann's formula for pole placement
    assert(ctrb.shape[0] == np.linalg.matrix_rank(ctrb))
    K = np.dot(last_row, np.dot(np.linalg.inv(ctrb), alpha_c_F)) # TODO: Solve

    return K

def build_SEIR():
    belowEI = np.zeros((nIR, nEI))
    belowEI[0,:] = 1-np.sum(EI, axis=0)

    #belowIR = np.zeros(nIR)
    belowIR = 1-np.sum(IR, axis=0)

    # Full SEIR Dynamics
    A = np.block([ [1,                 np.zeros(nEI),     np.zeros(nIR),       0],
                    [np.zeros((nEI,1)), EI,                np.zeros((nEI,nIR)), np.zeros((nEI,1))],
                    [np.zeros((nIR,1)), belowEI,           IR,                  np.zeros((nIR,1))],
                    [0,                 np.zeros((1,nEI)), belowIR,             1] ])

    B = np.matrix(np.zeros((2+nEI+nIR,1)))
    B[0] = -1
    B[1] = 1

    C = np.matrix(np.zeros((1,2+nEI+nIR)))
    C[:, 1:-1] = 1

    K = design_controller(A,B,C,pole_loc)

    return A,B,C,K


A,B,C,K = build_SEIR()

if force_run or not os.path.isfile(cachefn):
    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)

    ctr = cvc.controller(targets, gain=0.05)
    sim.pars['interventions'] = [ctr] # Remove interventions (hopefully not necessary!)
    #sim.pars['rand_seed'] = 0
    #sim.pars['end_day'] = '2020-09-05'
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
    #u = beta * xs*xi / pop_size # np.power(xi, 1)
    u = -K*Xu
    print(t, u)
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
fig, axv = plt.subplots(1,2,figsize=(16,10))
if do_plot:
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
