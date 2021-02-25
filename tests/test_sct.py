import pytest
import school_tools as sct
import covasim_schools.school_pop as cvsch_pop
import covasim as cv # For creating interventions
import sciris as sc

from pathlib import Path

def run_sim(sim, save_file=None):
    sim.run()
    if save_file:
        sim.to_json(
            filename=save_file
            , indent=4
        )
    return sim.results

def verify_end_popcount(expected, results):
    assert expected == \
           results['cum_deaths'][-1] + results['n_alive'][-1]


@pytest.mark.parametrize("specific_pop_size", [
    10_000
    , 12_000
    , 3_500
])
def test_createsim_size(request, specific_pop_size):
    save_file = f"DEBUG_{request.node.name}.json"
    sim = sct.create_sim(
        pop_size=specific_pop_size
        , load_pop=False
    )
    results = run_sim(
        sim=sim
        , save_file=save_file
    )
    verify_end_popcount(
        expected=specific_pop_size
        , results=results
    )
    assert Path(save_file).is_file()
    Path(save_file).unlink(missing_ok=True)
    return


def test_createsim_from_people():
    thirteen_thousand = 13_000

    my_people = cvsch_pop.make_population(
        pop_size=thirteen_thousand
        , rand_seed=1
        , max_pop_seeds=None
        , do_save=True
        , location='seattle_metro'
        , folder='inputs'
        , popfile=None
        , cohorting=True
        , community_contacts=20
    )
    sim_13k = sct.create_sim(
        pop_size=thirteen_thousand
        , people=my_people
        , load_pop=False
        , create_pop=True
    )
    results_13k = run_sim(sim=sim_13k)
    verify_end_popcount(
        expected=thirteen_thousand
        , results=results_13k
    )
    return


def test_createsim_createpop():
    eleven_thousand = 11_000
    sct.config.sweep_pars.n_pops = 1
    path_11k = Path("inputs").joinpath("seattle_metro_clustered_11000_seed0.ppl")
    if path_11k.is_file():
        path_11k.unlink(missing_ok=True)

    sim_11k = sct.create_sim(
        pop_size=eleven_thousand
        , people=None
        , load_pop=True
        , create_pop=True
    )

    results_11k = run_sim(sim=sim_11k)
    verify_end_popcount(
        expected=eleven_thousand
        , results=results_11k
    )
    return


def test_school_vaccine_scaling():
    """
    Creates a series of school vaccines of increasing efficacy
    Changes beta to zero for community, home, ltcf and work layers

    Expectation: Number of infections should decrease with increased vaccine effectiveness
    """
    is_debugging = True

    eleven_thousand = 11_000
    starter_infections = 10

    my_people = cvsch_pop.make_population(
        pop_size=eleven_thousand
        , rand_seed=1
        , max_pop_seeds=None
        , do_save=True
        , location='seattle_metro'
        , folder='inputs'
        , popfile=None
        , cohorting=True
        , community_contacts=20
    )

    final_infection_counts = {}
    for multiplier in [1.0, 0.7, 0.3, 0.0]:

        school_vaccine = sct.SchoolVaccine(
            rel_sus_mult=multiplier
            , label='my_school_vax'
            , symp_prob_mult=0
            , teacher_cov=1
            , staff_cov=1
            , student_cov=1
        )

        sim_pars = {
            'pop_infected': starter_infections,
            'pop_size': eleven_thousand
        }
        only_schools = cv.change_beta(layers=['h', 'c', 'w', 'l']
                                      , days=[0]
                                      , changes=[0]
                                      )
        pars = sct.config.get_defaults()
        pars = sc.mergedicts(pars, sim_pars)

        sim_vax = cv.Sim(pars, popfile=my_people, load_pop=True
                         , interventions=[school_vaccine, only_schools]
                         )

        results_11k = run_sim(sim=sim_vax)
        final_infection_counts[multiplier] = results_11k['cum_infections'][-1]

        if is_debugging:
            sim_vax.to_json(filename=f"DEBUG_sim_vax_{multiplier}.json",
                            indent=4)
        if multiplier == 0.0:
            assert results_11k['cum_infections'][0] == results_11k['cum_infections'][-1]

    if is_debugging:
        print(final_infection_counts)
    assert final_infection_counts[1.0] > final_infection_counts[0.7]
    assert final_infection_counts[0.7] > final_infection_counts[0.3]
    assert final_infection_counts[0.3] > final_infection_counts[0.0]
    return
