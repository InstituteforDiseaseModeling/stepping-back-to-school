'''
Test of the SEIR model for the controller
'''

import covasim_controller as cvc
import school_tools as sct

pop_size = 3000
params = {
    'rand_seed': 0,
    'pop_infected': 100,
    'change_beta': 1,
    'symp_prob': 0.1,
    'start_day': '2021-01-01',
    'end_day': '2021-03-01',
}


def test_fit(force_run=True):

    sim = sct.create_sim(params, pop_size=int(pop_size), load_pop=False)
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


if __name__ == '__main__':
    test_fit()
