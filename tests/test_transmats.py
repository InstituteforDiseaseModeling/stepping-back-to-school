'''
The covasim_controller relies on a SEIR representation of the system.  The latent and infectious periods are non-exponential, so here we fit a multi-dimensional state transition matrix to capture the exposed (E) to infectious (I) and infectious to recovered (R) durations.  While we have functional forms for these distribution in Covasim, result will depend on the population age structure and other factors. So here we run a large simulation to capture many infections and then fit TransitionMatrix objects to the resulting outputs.  The simulation, as configured, is rather large and takes about a minute to run.
'''

import covasim_controller as cvc
from risk_evaluation import create_sim as cs

pop_size = 5_000
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1,
    'start_day': '2021-01-01',
    'end_day': '2021-03-01',
}


def test_fit(force_run=True):

    sim = cs.create_sim(params, pop_size=int(pop_size), load_pop=False)
    sim.pars['interventions'] = [] # Remove interventions
    sim.run()

    inds = ~sim.people.susceptible
    print(f'There were {sum(inds)} exposures')

    print('Evaluating E-->I')
    e_to_i = sim.people.date_infectious[inds] - sim.people.date_exposed[inds]
    ei = cvc.TransitionMatrix(e_to_i, 3)
    ei.fit()
    ei.plot()

    print('Evaluating I-->R')
    i_to_r = sim.people.date_recovered[inds] - sim.people.date_infectious[inds]
    ir = cvc.TransitionMatrix(i_to_r, 7)
    ir.fit()
    ir.plot()


if __name__ == '__main__':
    test_fit()
