'''
Test the covasim_schools module directly
'''

import covasim_schools as cvsch
import school_tools as sct


def test_schools(pop_size=5e3, do_plot=False, debug=True):

    # See school_tools.scenarios for options
    scheduling_scen = 'all_hybrid'
    screening_scen = 'Antigen every 2w'

    scen = sct.generate_scenarios()[scheduling_scen]
    testing = sct.generate_screening()[screening_scen]
    for stype, spec in scen.items():
        if spec is not None:
            spec['testing'] = testing
    scen['es']['verbose'] = scen['ms']['verbose'] = scen['hs']['verbose'] = debug

    sim = sct.create_sim(pop_size=pop_size, load_pop=False) # Always create a new population

    sm = cvsch.schools_manager(scen)
    sim['interventions'] += [sm]

    sim.run(keep_people=debug)

    if do_plot:
        if debug:
            sim.plot(to_plot='overview')
        else:
            sim.plot()
        sim.people.plot()
        sim.people.plot_schools()

    return sim


if __name__ == '__main__':
    sim = test_schools(pop_size=20e3, do_plot=True)
