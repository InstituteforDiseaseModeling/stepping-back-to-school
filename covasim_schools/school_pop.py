'''
Generate a SynthPops population for use with the schools code.
'''

import os
import numpy as np
import sciris as sc
import covasim as cv
import synthpops as sp


def pop_path(popfile=None, folder=None, strategy=None, n=None, rand_seed=None):
    ''' Define the path for the population '''
    if popfile is None:
        name = f'seattle_{strategy}_{int(n)}_seed{rand_seed}.ppl'
        popfile = os.path.join(folder, name)
    return popfile


def make_population(pop_size, rand_seed=1, max_pop_seeds=None, do_save=True, folder='inputs', popfile=None, cohorting=True, n_brackets=20, community_contacts=20,**kwargs):
    '''
    Generate the synthpops population.

    Args:
        pop_size (int): number of people in the model
        rand_seed (int): random seed to use for generating the population
        max_pop_seeds (int): if supplied, take the random seed as modulus of this to limit number of populations generated
        do_save (bool): whether to save the population
        folder (str): if so, the root folder
        popfile (str): if so, where to save it to
        cohorting (bool): whether to use cohorting
        n_brackets (int): whether to use 16- or 20-bracket age bins
        community_contacts (int): how many community contacts there are
        kwargs (dict): passed to sp.make_population()
    '''

    sp.set_nbrackets(n_brackets) # Essential for getting the age distribution right

    pars = sc.objdict(
        n = pop_size,
        rand_seed = rand_seed,

        with_facilities=True,
        use_two_group_reduction = True,
        average_LTCF_degree = 20,
        ltcf_staff_age_min = 20,
        ltcf_staff_age_max = 60,

        country_location = 'usa',
        state_location = 'Washington',
        location = 'seattle_metro',
        use_default = True,

        smooth_ages = True,  # use smooth_ages to smooth out the binned age distribution
        window_length = 7,  # length of window in units of years to average binned age distribution over
        household_method = 'fixed_ages',  # use fixed_ages to match different age distributions more closely

        with_school_types = True,
        average_class_size = 20,
        inter_grade_mixing = 0.1,
        average_student_teacher_ratio = 20,
        average_teacher_teacher_degree = 3,
        teacher_age_min = 25,
        teacher_age_max = 75,

        # If True, the average_all_staff_ratio must be lower than the
        # average_student_teacher_ratio since all staff includes both teachers
        # and non teaching staff and so the ratio should be lower. If False, no
        # non teaching staff will be created in schools.
        with_non_teaching_staff = True,
        average_student_all_staff_ratio = 11,
        average_additional_staff_degree = 20,
        staff_age_min = 20,
        staff_age_max = 75,
    )

    pars.update(kwargs) # Update any parameters

    # For reference re: school_types
    # school_mixing_type = 'random' means that students in the school have edges randomly chosen from other students, teachers, and non teaching staff across the school. Students, teachers, and non teaching staff are treated the same in terms of edge generation.
    # school_mixing_type = 'age_clustered' means that students in the school have edges mostly within their own age/grade, with teachers, and non teaching staff. Strict classrooms are not generated. Teachers have some additional edges with other teachers.
    # school_mixing_type = 'age_and_class_clustered' means that students are cohorted into classes of students of the same age/grade with at least 1 teacher, and then some have contact with non teaching staff. Teachers have some additional edges with other teachers.

    if cohorting:
        strategy = 'clustered'  # students in pre-k, elementary, and middle school are cohorted into strict classrooms
        pars.school_mixing_type = {'pk': 'age_and_class_clustered', 'es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered',
                              'hs': 'random', 'uv': 'random'}
    else:
        strategy = 'normal'
        pars.school_mixing_type = {'pk': 'age_clustered', 'es': 'age_clustered', 'ms': 'age_clustered',
                              'hs': 'random', 'uv': 'random'}

    popfile = pop_path(popfile, folder, strategy, pars.n, pars.rand_seed)

    T = sc.tic()
    print('Making population...')

    # Make the population
    population = sp.make_population(**pars)

    # Convert to a popdict
    popdict = cv.make_synthpop(population=sc.dcp(population), community_contacts=community_contacts)
    school_ids = [None] * int(pop_size)
    teacher_flag = [False] * int(pop_size)
    staff_flag = [False] * int(pop_size)
    student_flag = [False] * int(pop_size)
    school_types = {'pk': [], 'es': [], 'ms': [], 'hs': [], 'uv': []}
    school_type_by_person = [None] * int(pop_size)
    schools = dict()

    for uid,person in population.items():
        if person['scid'] is not None:
            school_ids[uid] = person['scid']
            school_type_by_person[uid] = person['sc_type']
            if person['scid'] not in school_types[person['sc_type']]:
                school_types[person['sc_type']].append(person['scid'])
            if person['scid'] in schools:
                schools[person['scid']].append(uid)
            else:
                schools[person['scid']] = [uid]
            if person['sc_teacher'] is not None:
                teacher_flag[uid] = True
            elif person['sc_student'] is not None:
                student_flag[uid] = True
            elif person['sc_staff'] is not None:
                staff_flag[uid] = True

    assert sum(teacher_flag), 'Uh-oh, no teachers were found: as a school analysis this is treated as an error'
    assert sum(student_flag), 'Uh-oh, no students were found: as a school analysis this is treated as an error'

    # Actually create the people
    people_pars = dict(
        pop_size = pars.n,
        beta_layer = {k:1.0 for k in 'hswcl'}, # Since this is used to define what layers exist
        beta = 1.0, # TODO: this is required for plotting (people.plot()), but shouldn't be
    )

    people = cv.People(people_pars, strict=False, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'],
                          contacts=popdict['contacts'], school_id=np.array(school_ids),
                          schools=schools, school_types=school_types,
                          student_flag=student_flag, teacher_flag=teacher_flag,
                          staff_flag=staff_flag, school_type_by_person=school_type_by_person)

    if do_save:
        print(f'Saving to "{popfile}"...')
        sc.saveobj(popfile, people)

    sc.toc(T)

    print('Done')
    return people
