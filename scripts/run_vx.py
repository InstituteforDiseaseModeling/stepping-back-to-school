'''
Run a varitey of screening scenarios at a few prevalence levels
'''

import sys
import school_tools as sct

if __name__ == '__main__':

    # Settings
    args = sct.config.process_inputs(sys.argv)

    # 63% efficacy against infection and 34% efficacy against symptoms _given infection_ are estimates from personal communication with Mike Famulare
    realistic_vx = sct.Vaccine(rel_sus_mult=1-0.63, symp_prob_mult=1-0.34, teacher_cov=0.8, staff_cov=0.8, student_cov=0)
    perfect_vx = sct.Vaccine(rel_sus_mult=0, symp_prob_mult=0, teacher_cov=1, staff_cov=1, student_cov=0)

    sweep_pars = {
        'vaccine': {'None':None, 'Realistic Vaccine':[realistic_vx], 'Optimistic Vaccine':[perfect_vx]},
    }

    xvar = 'Prevalence Target'
    huevar = 'Vaccination'

    # Create and run
    mgr = sct.Manager(name='Vaccination', sweep_pars=sweep_pars, sim_pars=None, levels=None)
    mgr.run(args.force)
    analyzer = mgr.analyze()

    # Plots
    mgr.regplots(xvar=xvar, huevar=huevar, height=6, aspect=2.4)
    analyzer.introductions_rate(xvar=xvar, huevar=huevar, height=6, aspect=1.4, ext='ppt')
    analyzer.cum_incidence(colvar=xvar)
    analyzer.introductions_rate_by_stype(xvar=xvar)
    analyzer.outbreak_size_over_time()
    analyzer.source_pie()
    mgr.tsplots()
