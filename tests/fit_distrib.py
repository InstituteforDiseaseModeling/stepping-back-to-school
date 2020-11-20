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

def plot_dur(data, M, ax):
    mu = np.nanmean(data)
    h,x = np.histogram(data, bins=range(int(max(data))))
    ax.bar(x[:-1], np.cumsum(h)/np.sum(h))

    T = int(max(data))
    dur = np.matrix(np.zeros((M.shape[0],T+1)))
    dur[0,0] = 1

    for k in range(T):
        dur[:,k+1] = M*dur[:,k]
    ei = np.squeeze(np.asarray(dur[-1,:]))
    ax.plot( np.cumsum(ei) / np.sum(ei), marker='o', ms=5, color='r', lw=0.5, mew=4, alpha=0.7)
    ax.set_title(f'Mean {mu:.1f} days')
    ax.set_xlim([0,T])

    return


def get_cum(x, cmf):
    M = coxian(x)

    T = len(cmf)
    dur = np.matrix(np.zeros((int(len(x)),T)))
    dur[:,0] = 0
    dur[0,0] = 1
    for k in range(T-1):
        dur[:,k+1] = M*dur[:,k]
    cum = np.cumsum(np.squeeze(np.asarray(dur[-1,:])))
    cum /= cum[-1]

    return cum

def err(x, cmf):
    cum = get_cum(x, cmf)
    err = np.max(np.abs(cum - cmf))
    return err

def fit_dist(pmf, order):
    cmf = np.cumsum(pmf)
    x0 = np.random.rand(order)#0.3*np.ones(order)
    pars = spo.minimize(err, x0, args=(cmf))

    return pars

if __name__ == '__main__':
    inds = ~sim.people.susceptible
    print(f'There were {sum(inds)} exposures')
    e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]
    h,x = np.histogram(e_to_i, bins=range(int(max(e_to_i))))
    h = h / np.sum(h)
    fit_ei = fit_dist(h,3) # m3: [0.46110679, 0.45780867, 0.54005729]
    print(fit_ei)
    EI = coxian(fit_ei['x'])

    i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]
    h,x = np.histogram(i_to_r, bins=range(int(max(i_to_r))))
    #h = h[:16] # First mass only
    h = h / np.sum(h)
    fit_ir = fit_dist(h,7) # 7: [0.67208747, 0.6663703 , 0.67786915, 0.672478  , 0.91466127, 0.6712012 , 0.83614718]
    print(fit_ir)
    IR = coxian(fit_ir['x'])

    fig, axv = plt.subplots(1,2,figsize=(16,10))
    plot_dur(e_to_i, EI, axv[0])
    plot_dur(i_to_r, IR, axv[1])
    plt.show()
