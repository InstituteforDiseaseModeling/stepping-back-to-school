import pytest
import school_tools as sct
import covasim_schools.school_pop as cvsch_pop
import covasim as cv # For creating interventions
import sciris as sc


# Global settings
is_debugging = False
big_pop = 11_000
small_pop = 3_500


def run_sim(sim, save_file=None):
    sim.run()
    if save_file and is_debugging:
        sim.to_json(filename=save_file, indent=4)
    return sim.results

def verify_end_popcount(expected, results):
    assert expected == results['cum_deaths'][-1] + results['n_alive'][-1]


@pytest.mark.parametrize("specific_pop_size", [
    big_pop,
    small_pop,
])
def test_createsim_size(specific_pop_size):
    sim = sct.create_sim(pop_size=specific_pop_size, load_pop=False)
    results = run_sim(sim=sim)
    verify_end_popcount(expected=specific_pop_size, results=results)
    return


def test_createsim_from_people():

    my_people = cvsch_pop.make_population(
        pop_size=small_pop,
        rand_seed=1,
        max_pop_seeds=None,
        do_save=True,
        location='seattle_metro',
        folder='inputs',
        popfile=None,
        cohorting=True,
        community_contacts=20,
    )
    sim = sct.create_sim(
        pop_size=small_pop,
        people=my_people,
        load_pop=False,
        create_pop=True,
    )
    results = run_sim(sim=sim)
    verify_end_popcount(expected=small_pop, results=results)
    return


def test_createsim_createpop():
    sct.config.sweep_pars.n_pops = 1
    sim = sct.create_sim(
        pop_size=small_pop,
        people=None,
        load_pop=False,
        create_pop=True,
    )
    results = run_sim(sim=sim)
    verify_end_popcount(expected=small_pop, results=results)
    return


def test_school_vaccine_scaling():
    """
    Creates a series of school vaccines of increasing efficacy
    Changes beta to zero for community, home, ltcf and work layers

    Expectation: Number of infections should decrease with increased vaccine effectiveness
    """
    starter_infections = 10
    multipliers = [1.0, 0.5, 0.0]

    my_people = cvsch_pop.make_population(
        pop_size=big_pop,
        rand_seed=1,
        max_pop_seeds=None,
        do_save=True,
        location='seattle_metro',
        folder='inputs',
        popfile=None,
        cohorting=True,
        community_contacts=20,
    )

    final_infection_counts = {}
    for multiplier in multipliers:

        school_vaccine = sct.SchoolVaccine(
            rel_sus_mult=multiplier,
            label='school_vax',
            symp_prob_mult=0,
            teacher_cov=1,
            staff_cov=1,
            student_cov=1,
            school_types=['pk', 'es', 'ms', 'hs', 'uv'],
        )

        sim_pars = {
            'pop_infected': starter_infections,
            'pop_size': big_pop
        }
        only_schools = cv.change_beta(layers=['h', 'c', 'w', 'l'], days=0, changes=0)
        pars = sct.config.get_defaults()
        pars = sc.mergedicts(pars, sim_pars)

        sim_vax = cv.Sim(pars, popfile=my_people, load_pop=True,
                         interventions=[school_vaccine, only_schools]
                         )

        results = run_sim(sim=sim_vax)
        final_infection_counts[multiplier] = results['cum_infections'][-1]

        if is_debugging:
            sim_vax.to_json(filename=f"DEBUG_sim_vax_{multiplier}.json",
                            indent=4)
        if multiplier == 0.0:
            assert results['cum_infections'][0] == results['cum_infections'][-1], 'Should be no new infections'

    print('Final infection counts:', final_infection_counts)
    for m in range(len(multipliers)-1):
        mult1 = multipliers[m]
        mult2 = multipliers[m+1]
        assert mult1 > mult2, 'Multipliers must be monotonic decreasing'
        assert final_infection_counts[mult1] > final_infection_counts[mult2]
    return


if __name__ == '__main__':
    test_createsim_size(specific_pop_size=big_pop)
    test_createsim_from_people()
    test_createsim_createpop()
    test_school_vaccine_scaling()