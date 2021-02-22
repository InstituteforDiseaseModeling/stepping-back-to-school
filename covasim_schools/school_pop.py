'''
Generate a SynthPops population for use with the schools code.
'''

import os
import numpy as np
import sciris as sc
import pylab as pl
import covasim as cv
import synthpops as sp


n_brackets = 20 # This is required to load the correct age distributions


class SchoolPeople(cv.People):
    ''' Subclass of People to add an additional plotting method '''

    def plot_schools(self, figsize=None, dpi=None, to_json=False, outfile=None, do_plot=True, do_show=True, return_results=True):
        ''' Sanity check for school distributions '''

        # Custom configuration
        keys = ['pk', 'es', 'ms', 'hs'] # Exclude universities for this analysis
        ppl_keys = ['all', 'students', 'teachers', 'staff']
        xpeople = np.arange(len(ppl_keys)) # X axis for people
        school_types_by_ind = {}
        for key,vals in self.school_types.items():
            for val in vals:
                if key in keys:
                    school_types_by_ind[val] = key

        results = {}
        for sc_id,sc_type in school_types_by_ind.items():
            thisres = sc.objdict()
            sc_inds = (self.school_id == sc_id)
            thisres.all = cv.true(sc_inds)
            thisres.students = cv.true(np.array(self.student_flag) * sc_inds)
            thisres.teachers = cv.true(np.array(self.teacher_flag) * sc_inds)
            thisres.staff    = cv.true(np.array(self.staff_flag) * sc_inds)
            results[sc_id] = thisres

        # Do plotting
        if do_plot:
            fig = pl.figure(figsize=figsize, dpi=dpi)
            pl.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.5, wspace=0.5)
            n_schools = len(results)
            n_cols = len(ppl_keys) + 1
            count = 0
            for sc_id in results.keys():
                count += 1
                school_type = school_types_by_ind[sc_id]
                ax = pl.subplot(n_schools, n_cols, count)
                thisres = results[sc_id]
                thisres.people_counts = [len(thisres[k]) for k in ppl_keys]
                ax.bar(xpeople, thisres.people_counts)
                ax.set_xticks(xpeople)
                ax.set_xticklabels(ppl_keys)
                title = f'School ID {sc_id}, school type {school_type}, total size: {len(thisres.all)}'
                ax.set_title(title)

                thisres.ages = sc.objdict()
                for key in ppl_keys:
                    count += 1
                    ax = pl.subplot(n_schools, n_cols, count)
                    thisres.ages[key] = self.age[thisres[key]]
                    pl.hist(thisres.ages[key])
                    ax.set_title(f'Ages for {key} in school {sc_id} ({school_type})')

        if to_json:
            sc.savejson(outfile, results, indent=2)

        if do_show:
            pl.show()

        if return_results:
            return results
        elif do_plot:
            return fig
        return


def pop_path(popfile=None, location=None, folder=None, strategy=None, n=None, rand_seed=None):
    ''' Define the path for the population '''
    if folder is None:
        folder = '.' # Current folder
    if popfile is None:
        name = f'{location}_{strategy}_{int(n)}_seed{rand_seed}.ppl'
        popfile = os.path.join(folder, name)
    return popfile


def make_population(pop_size, rand_seed=1, max_pop_seeds=None, do_save=True, location='seattle_metro', folder=None, popfile=None, cohorting=True, community_contacts=20, rm_layers=None, **kwargs):
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
        community_contacts (int): how many community contacts there are
        rm_layers (list): if not None, remove these layers
        kwargs (dict): passed to sp.make_population()
    '''

    sp.set_nbrackets(n_brackets)

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
        location = location,
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

    popfile = pop_path(popfile=popfile, location=location, folder=folder, strategy=strategy, n=pars.n, rand_seed=pars.rand_seed)

    T = sc.tic()
    print('Making population...')

    # Make the population
    population = sp.make_population(**pars)

    # Convert to a popdict
    popdict = cv.make_synthpop(population=sc.dcp(population), community_contacts=community_contacts)
    school_ids = [np.nan] * int(pop_size)
    school_flag = [False] * int(pop_size)
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
            school_flag[uid] = True
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

    # Identical to a Covasim People instance except with the plotting method defined above
    people = SchoolPeople(people_pars, strict=False, uid=popdict['uid'], age=popdict['age'], sex=popdict['sex'],
                          contacts=popdict['contacts'], school_id=np.array(school_ids),
                          schools=schools, school_types=school_types,
                          school_flag=school_flag, student_flag=student_flag, teacher_flag=teacher_flag,
                          staff_flag=staff_flag, school_type_by_person=school_type_by_person)

    if rm_layers is not None:
        for lkey in rm_layers:
            people.contacts.lkey = cv.Layer() # Replace with an empty layer

    if do_save:
        print(f'Saving to "{popfile}"...')
        sc.saveobj(popfile, people)

    sc.toc(T)

    print('Done')
    return people
