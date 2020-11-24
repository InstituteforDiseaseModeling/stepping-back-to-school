import scipy.optimize as spo
import os
import seaborn as sns
import numpy as np
import covasim as cv
import covasim.utils as cvu
import matplotlib.pyplot as plt

cachefn = 'sim.obj'
sim = cv.load(cachefn)

def coxian(p):
    n = len(p)
    M = np.zeros((n,n))
    np.fill_diagonal(M, 1-p)
    np.put(M, n+(n+1)*np.arange(n-1), p[:-1])
    return M

def full(p):
    n = int((-1 + np.sqrt( (1 + 8*len(p)) ))/2)
    M = np.zeros((n,n))
    np.put(M, np.ravel_multi_index(np.tril_indices(n), M.shape), p)

    #M = np.array([
    #    [1-p[0],                0,              0,      0],
    #    [p[1]*p[0],             1-p[3],         0,      0],
    #    [p[2]*p[0],             p[4]*p[3],      1-p[5], 0],
    #    [(1-p[1]-p[2])*p[0],    (1-p[4])*p[3],  p[5],   1]
    #])

    return M

def add_cum(M):
    F = np.block([[M, np.zeros((M.shape[0],1))],
                  [1-M.sum(axis=0), 1]])
    return F

def plot_dur(data, M, ax, ax2=None):
    mu = np.nanmean(data)
    h,x = np.histogram(data, bins=range(int(max(data))))
    ax.bar(x[:-1], np.cumsum(h)/np.sum(h))


    F = add_cum(M)

    T = int(max(data))
    dur = np.matrix(np.zeros((F.shape[0],T+1)))
    dur[0,0] = 1

    for k in range(T):
        dur[:,k+1] = F*dur[:,k]
    ei = np.squeeze(np.asarray(dur[-1,:]))
    #ax.plot( np.cumsum(ei) / np.sum(ei), marker='o', ms=5, color='r', lw=0.5, mew=4, alpha=0.7)
    ax.plot( np.squeeze(np.asarray(dur[-1,:])), marker='o', ms=5, color='r', lw=0.5, mew=4, alpha=0.7)

    ax.set_title(f'Mean {mu:.1f} days')
    ax.set_xlim([0,T])

    if ax2 is None:
        return

    for i in range(F.shape[0]):
        ei = np.squeeze(np.asarray(dur[i,:]))
        ax2.plot(ei)
    ax2.set_xlim([0,T])

    return

def get_cum(x, cmf):
    M = full(x)
    F = add_cum(M)

    T = len(cmf)
    dur = np.matrix(np.zeros((F.shape[0],T)))
    dur[:,0] = 0
    dur[0,0] = 1
    for k in range(T-1):
        dur[:,k+1] = F*dur[:,k]

    return dur[-1,:]

def err(x, cmf):
    cum = get_cum(x, cmf)
    err = np.max(np.abs(cum - cmf))
    return err

def fit_dist(pmf, n):
    # n is order
    p = int(n*(n+1)/2) # Number of parameters

    cmf = np.cumsum(pmf)

    x0 = 0.5*np.random.rand(p)#0.3*np.ones(order)

    lc = [np.eye(p)]
    inds = [0]
    for col in range(n-1):
        if col == 0:
            # Build inds
            for i in range(1,n):
                inds.append(inds[-1]+i)
            inds = np.array(inds)
        else:
            inds += 1
            inds = inds[1:]
        constraint = np.zeros(p)
        constraint[inds] = 1
        lc.append(constraint)

    A = np.vstack(lc)
    lc = spo.LinearConstraint(A, np.zeros(A.shape[0]), np.ones(A.shape[0]))
    pars = spo.minimize(err, x0, constraints=lc, args=(cmf), method='SLSQP')

    return pars

if __name__ == '__main__':
    inds = ~sim.people.susceptible
    print(f'There were {sum(inds)} exposures')
    e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]

    #print('WARNING: Removing first day!')
    #e_to_i -= 1

    h,x = np.histogram(e_to_i, bins=range(int(max(e_to_i))))
    h = h / np.sum(h)
    fit_ei = fit_dist(h,3) # m3: [0.46110679, 0.45780867, 0.54005729]
    print(fit_ei)
    EI = full(fit_ei['x'])
    print('EI:\n', EI)

    i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]

    #print('WARNING: Removing first day!')
    #i_to_r -= 1

    h,x = np.histogram(i_to_r, bins=range(int(max(i_to_r))))
    #h = h[:16] # First mass only
    h = h / np.sum(h)
    fit_ir = fit_dist(h,7) # 7: [0.67208747, 0.6663703 , 0.67786915, 0.672478  , 0.91466127, 0.6712012 , 0.83614718]
    print(fit_ir)
    IR = full(fit_ir['x'])
    print('IR:\n', IR)

    fig, axv = plt.subplots(2,2,figsize=(16,10))
    plot_dur(e_to_i, EI, axv[0,0], axv[0,1])
    plot_dur(i_to_r, IR, axv[1,0], axv[1,1])
    plt.show()
