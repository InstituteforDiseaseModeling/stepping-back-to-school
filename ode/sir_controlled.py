import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt

beta = 2
gamma = 0.25

Rt_desired = 4

def sir(t,x,beta,gamma):
    s,i,r = x

    u_beta = np.maximum(0, Rt_desired * gamma / s)

    dsdt = -u_beta*s*i
    didt = u_beta*s*i - gamma*i
    drdt = gamma*i

    return [dsdt, didt, drdt]

sln = spi.solve_ivp(sir, [0,10], [0.999, 0.001, 0], method='RK45', dense_output=True, args=(beta, gamma))

t = np.linspace(0,10,100)

#plt.plot(sln.t, sln.y.T)
z = sln.sol(t)

fig, axv = plt.subplots(1,2,figsize=(16,10))
axv[0].plot(t, z.T)
axv[0].set_xlabel('Time')
axv[0].legend(['Susceptible', 'Infected', 'Recovered'])
axv[0].set_title('SIR System')


dzdt = sir(t, z, beta, gamma)
didt = dzdt[1]
axv[1].plot(t, didt, color='orange')
yl = axv[1].get_ylim()


s = z[0]
axr = axv[1].twinx()  # instantiate a second axes that shares the same x-axis
axr.axhline(y=beta/gamma, color='r')
axr.plot(t, beta/gamma*s, color='r', ls='--')
axr.axhline(y=1.0, color='k', ls=':')
yr = axr.get_ylim()
yr0   = (1 - yr[1]* (0-yl[0]) / (yl[1]-yl[0])) / (1 - (0-yl[0]) / (yl[1]-yl[0]))
axr.set_ylim(yr0, yr[1])

plt.show()
