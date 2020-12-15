'''
Plot diagnostic screening scenarios, resulting in Figures 1, 2, 4, and 7.
Call this after running run_testing_scenarios.py.
'''

import os
from pathlib import Path
import covasim as cv
import sciris as sc
import numpy as np
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from calibrate_model import evaluate_sim # Local import


# Which figures to plot
to_plot = {'Fig. 1': True, # 3-month attack rate for students and teachers + staff
           'Combined attack rate': False, # Like Fig. 1, but with students, staff, and teachers combined
           'Separate attack rates': False, # Like Fig. 1, but plot attack rates in separate figures
           'Fig. 2': True, # Population-wide reproduction number
           'Fig. 4': True, # Proportion of days that are remote instead of in-person learning
           'Additional tests': False, # Additional diagnostic tests required
           }
tree_plot = False

# Global plotting styles
font_size = 18
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style

# Other configuration
folder = 'v2020-12-02'
variant = 'optimistic_countermeasures_Antigen_10reps'
#variant = 'k5_PCROptimistic_Sweep1reps'

cachefn = os.path.join(folder, 'sims', f'{variant}.sims') # Might need to change the extension here, depending in combine.py was used
simple = False # Boolean flag to select a subset of the scenarios

variant = variant + '_simple' if simple else variant
imgdir = os.path.join(folder, 'img_'+variant)
Path(imgdir).mkdir(parents=True, exist_ok=True)

for_presentation = False # Choose between report style and presentation style (different aspect ratio)
figsize = (12,8) if for_presentation else (12,9.5)
aspect = 3 if for_presentation else 2.5

T = sc.tic() # Start timing

print(f'Loading {cachefn}')
sims = cv.load(cachefn) # Use for *.sims

def ax_col(ax, col='blue'):
    ''' Set standard axis configurations '''
    ax.spines['left'].set_color(col)
    ax.tick_params(axis='y', colors=col)
    ax.xaxis.label.set_color(col)

results = []
byschool = []
groups = ['students', 'teachers', 'staff']

scen_names = sc.odict({ # key1
    'as_normal': 'Full Schedule\nNo Countermeasures',
    'with_countermeasures': 'Full Schedule',
    'all_hybrid': 'Hybrid',
    'k5': 'K-5 In-Person\nOthers Remote',
    'all_remote': 'All Remote',
})
scen_order = scen_names.keys()

blues = plt.cm.get_cmap('Blues')
reds = plt.cm.get_cmap('Reds')
test_names = sc.odict({ # key2
    'None':                                     ('No diagnostic screening',                         'gray'),
    'PCR 1w prior':                             ('PCR one week prior, 1d delay',                    blues(1/6)),
    'PCR every 4w':                             ('Monthly PCR, 1d delay',                           blues(2/6)),
    'Antigen every 1w teach&staff, PCR f/u':    ('Weekly antigen for teachers & staff, PCR f/u',    reds(1/6)),
    'Antigen every 4w, PCR f/u':                ('Monthly antigen, no f/u',                         reds(2/6)),
    'Antigen every 2w, no f/u':                 ('Fortnightly antigen, no f/u',                     reds(3/6)),
    'Antigen every 2w, PCR f/u':                ('Fortnightly antigen, PCR f/u',                    reds(4/6)),
    'PCR every 2w':                             ('Fortnightly PCR, 1d delay',                       blues(3/6)),
    'Antigen every 1w, PCR f/u':                ('Weekly antigen, PCR f/u',                         reds(5/6)),
    'PCR every 1w':                             ('Weekly PCR, 1d delay',                            blues(4/6)),
    'PCR every 1d':                             ('Daily PCR, no delay',                             blues(5/6)),
})

if simple:
    # Select a subset
    test_order = [v[0] for k,v in test_names.items() if k in [ 'None', 'PCR 1w prior', 'Antigen every 1w teach&staff, PCR f/u', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u'] ]
else:
    test_order = [v[0] for k,v in test_names.items()]
test_hue = {v[0]:v[1] for v in test_names.values()}



#%% Process the simulations

for sim in sims:
    first_school_day = sim.day('2020-11-02')
    last_school_day = sim.day('2021-01-31')
    sim.key2 = test_names[sim.key2][0] if sim.key2 in test_names else sim.key2

    ret = {
        'key1': sim.key1,
        'key2': sim.key2,
        'key3': sim.key3,
        'rep': sim.eidx,
    }

    perf = evaluate_sim(sim) # TODO: check if hit desired targets
    ret.update(perf)
    ret['introductions'] = []
    ret['introductions_per_100_students'] = []
    ret['introductions_postscreen'] = []
    ret['introductions_postscreen_per_100_students'] = []
    ret['outbreak_size'] = []

    n_schools = {'es':0, 'ms':0, 'hs':0}
    n_schools_with_inf_d1 = {'es':0, 'ms':0, 'hs':0}

    grp_dict = {'Students': ['students'], 'Teachers & Staff': ['teachers', 'staff'], 'Students, Teachers, and Staff': ['students', 'teachers', 'staff']}
    perc_inperson_days_lost = {k:[] for k in grp_dict.keys()}
    attackrate = {k:[] for k in grp_dict.keys()}
    count = {k:0 for k in grp_dict.keys()}
    exposed = {k:0 for k in grp_dict.keys()}
    inperson_days = {k:0 for k in grp_dict.keys()}
    possible_days = {k:0 for k in grp_dict.keys()}

    first_date = '2020-11-02'
    first = sim.day(first_date)
    last_date = '2021-01-31'
    last = sim.day(last_date)

    if sim.results['n_exposed'][first] == 0:
        print('Sim has zero exposed, skipping')
        print(ret)
        continue

    for sid,stats in sim.school_stats.items():
        if stats['type'] not in ['es', 'ms', 'hs']:
            continue

        inf_first = stats['infectious_first_day_school'] # Post-screening
        in_person = stats['in_person']
        exp = stats['newly_exposed']
        num_school_days = stats['num_school_days']
        possible_school_days = np.busday_count(first_date, last_date)
        n_exp = {}
        for grp in groups:
            n_exp[grp] = exp[grp]

        for gkey, grps in grp_dict.items():
            in_person_days = scheduled_person_days = num_exposed = num_people = 0
            for grp in grps:
                in_person_days += in_person[grp]
                scheduled_person_days += num_school_days * stats['num'][grp]
                num_exposed += n_exp[grp]
                num_people += stats['num'][grp]
                exposed[gkey] += n_exp[grp]
                count[gkey] += stats['num'][grp]

            perc_inperson_days_lost[gkey].append(
                100*(scheduled_person_days - in_person_days)/scheduled_person_days if scheduled_person_days > 0 else 100
            )
            attackrate[gkey].append( 100 * num_exposed / num_people)

            inperson_days[gkey] += in_person_days
            possible_days[gkey] += possible_school_days*num_people

        n_schools[stats['type']] += 1
        if sum([inf_first[g] for g in groups]) > 0:
            n_schools_with_inf_d1[stats['type']] += 1

        byschool.append({
            'sid': sid,
            'type': stats['type'],
            'key1': sim.key1, # Filtered to just one scenario (key1)
            'key2': sim.key2,
            'key3': sim.key3,
            'n_students': stats['num']['students'],
            'n': sum([stats['num'][g] for g in groups]),
            'd1 infectious': sum([inf_first[g] for g in groups]),
            'd1 bool': sum([inf_first[g] for g in groups]) > 0,
            'PCR': stats['n_tested']['PCR'],
            'Antigen': stats['n_tested']['Antigen'],
            'Days': last_school_day - first_school_day,
            'Pop*Scale': sim.pars['pop_size']*sim.pars['pop_scale'],
        })

        #n_introductions = len(stats['outbreaks'])
        #ret['introductions'][np.min([n_introductions,len(ret['introductions'])-1])] += 1
        ret['introductions'].append(len(stats['outbreaks']))
        ret['introductions_per_100_students'].append( len(stats['outbreaks']) / stats['num']['students'] * 100 )
        intr_postscreen = len([o for o in stats['outbreaks'] if o['Total infectious days at school']>0]) # len(stats['outbreaks'])
        ret['introductions_postscreen'].append(intr_postscreen)
        ret['introductions_postscreen_per_100_students'].append( intr_postscreen / stats['num']['students'] * 100 )
        ret['outbreak_size'] += [ob['Infected Student'] + ob['Infected Teacher'] + ob['Infected Staff'] for ob in stats['outbreaks']]

        for i,tree in enumerate(stats['school_trees']):
            if tree_plot and sim.key2 == 'Monthly PCR, 1d delay':
                fig, ax = plt.subplots(figsize=(16,10))
                date_range = [sim.pars['n_days'],0]

            #print(f'Tree {i}', sid, sim.key1, sim.key2, sim.key2)
            #for u,v,w in tree.edges.data():
                #print('\tEDGE', u,v,w)
            #print(f'N{i}', sid, sim.key1, sim.key2, sim.key2, tree.nodes.data())
            for j, (u,v) in enumerate(tree.nodes.data()):
                #print('\tNODE', u,v)
                recovered = sim.pars['n_days'] if np.isnan(v['date_recovered']) else v['date_recovered']
                if tree_plot and sim.key2 == 'Monthly PCR, 1d delay':
                    col = 'gray' if v['type'] == 'Other' else 'black'
                    date_range[0] = min(date_range[0], v['date_exposed']-1)
                    date_range[1] = max(date_range[1], recovered+1)
                    ax.plot( [v['date_exposed'], recovered], [j,j], '-', marker='o', color=col)
                    ax.plot( v['date_diagnosed'], j, marker='d', color='b')
                    ax.plot( v['date_infectious'], j, marker='|', color='r', mew=3, ms=10)
                    ax.plot( v['date_symptomatic'], j, marker='s', color='orange')
                    for day in range(int(v['date_exposed']), int(recovered)):
                        if day in stats['uids_at_home'] and int(u) in stats['uids_at_home'][day]:
                            plt.plot([day,day+1], [j,j], '-', color='lightgray')
                    for t, r in stats['testing'].items():
                        for kind, outcomes in r.items():
                            if int(u) in outcomes['Positive']:
                                plt.plot(t, j, marker='x', color='hotpink', ms=10)
                            elif int(u) in outcomes['Negative']:
                                plt.plot(t, j, marker='x', color='green', ms=10)

            if tree_plot and sim.key2 == 'Monthly PCR, 1d delay':
                for t, r in stats['testing'].items():
                    ax.axvline(x=t, zorder=-100)
                date_range[1] = min(date_range[1], sim.pars['n_days'])
                ax.set_xlim(date_range)
                ax.set_xticks(range(int(date_range[0]), int(date_range[1])))
                ax.set_yticks(range(0, len(tree.nodes)))
                ax.set_yticklabels([f'{u}: {v["type"]} {v["age"]}' for u,v in tree.nodes.data()])
                ax.set_title(f'School {sid}, Tree {i}')
                #plt.show()
                plt.close('all')


    for stype in ['es', 'ms', 'hs']:
        ret[f'{stype}_perc_d1'] = 100 * n_schools_with_inf_d1[stype] / n_schools[stype]

    # Deciding between district and school perspective here
    for gkey in grp_dict.keys():
        ret[f'perc_inperson_days_lost_{gkey}'] = 100*(possible_days[gkey]-inperson_days[gkey])/possible_days[gkey] #np.mean(perc_inperson_days_lost[gkey])
        ret[f'attackrate_{gkey}'] = 100*exposed[gkey] / count[gkey] #np.mean(attackrate[gkey])
        ret[f'count_{gkey}'] = np.sum(count[gkey])

    results.append(ret)

# Convert results to a dataframe
df = pd.DataFrame(results)

################
#%% PLOT RESULTS
################

def plot_prev():
    l = []
    for sim in sims:
        t = sim.results['t'] #pd.to_datetime(sim.results['date'])
        y = sim.results['n_exposed'].values / sim.pars['pop_size']
        d = pd.DataFrame({'Prevalence': y}, index=pd.Index(data=t, name='Date'))
        d['Prevalence Target'] = f'{sim.key3}'
        d['Scenario'] = f'{sim.key1} + {sim.key2}'
        d['Rep'] = sim.eidx
        l.append( d )
    d = pd.concat(l).reset_index()

    fig, ax = plt.subplots(figsize=(16,10))
    sns.lineplot(data=d, x='Date', y='Prevalence', hue='Prevalence Target', style='Scenario', palette='cool', ax=ax, legend=False)
    # Y-axis gets messed up when I introduce horizontal lines!
    #for prev in d['Prevalence Target'].unique():
    #    ax.axhline(y=prev, ls='--')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))

    cv.savefig(os.path.join(imgdir, f'Prevalence.png'), dpi=300)

plot_prev()

#msim = cv.MultiSim(sims)
#msim.plot(to_plot={'Prevalence':['n_exposed']}, do_show=False)
#cv.savefig(os.path.join(imgdir, f'Prevalence.png'), dpi=300)

d = df[['key1', 'key2', 'key3', 'outbreak_size']]
l = []
for k,dat in d.groupby(['key1', 'key2', 'key3']):
    for entry in np.hstack(dat['outbreak_size']):
        l.append([k[0], k[1], k[2], entry])
osdf = pd.DataFrame(l, columns=['key1', 'key2', 'key3', 'outbreak_size'])

d = df[['key1', 'key2', 'key3', 'introductions_per_100_students', 'introductions_postscreen_per_100_students']]
l = []
for k,dat in d.groupby(['key1', 'key2', 'key3']):
    for intr, intr_ps in zip(np.hstack(dat['introductions_per_100_students']), np.hstack(dat['introductions_postscreen_per_100_students'])):
        l.append([k[0], k[1], k[2], intr, intr_ps])
idf = pd.DataFrame(l, columns=['key1', 'key2', 'key3', 'introductions_per_100_students', 'introductions_postscreen_per_100_students'])


d = df[['key1', 'key2', 'key3', 'introductions_postscreen_per_100_students']]
l = []
for k,dat in d.groupby(['key1', 'key2', 'key3']):
    for entry in np.hstack(dat['introductions_postscreen_per_100_students']):
        l.append([k[0], k[1], k[2], entry])
isdf = pd.DataFrame(l, columns=['key1', 'key2', 'key3', 'introductions_postscreen_per_100_students'])

####################

fig, ax = plt.subplots(figsize=(6,5))
norm = mplt.colors.LogNorm()#vmin=1, vmax=50)
import copy
my_cmap = copy.copy(mplt.cm.get_cmap('inferno')) # copy the default cmap
my_cmap.set_bad((0,0,0))
h = ax.hist2d(isdf['key3'], isdf['introductions_postscreen_per_100_students'], norm=norm, bins=15, cmap=my_cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('Prevalence')
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=1))
ax.set_ylabel('Introductions per 100 students\nover 3 months')
plt.tight_layout()
cv.savefig(os.path.join(imgdir, f'IntroductionsHeatmap.png'), dpi=300)

#ho = [test_names[x][0] for x in ['None', 'PCR every 4w', 'PCR every 2w', 'PCR every 1w']]
ho = [test_names[x][0] for x in ['None', 'Antigen every 4w, PCR f/u', 'Antigen every 2w, PCR f/u', 'Antigen every 1w, PCR f/u']]
#osdf['log10_outbreak_size'] = np.log10(osdf['outbreak_size'])

### OUTBREAK SIZE #############################################################
osdf = osdf[osdf['outbreak_size'] > 1]
g = sns.catplot(data=osdf, x='outbreak_size', y='key2', order=ho, col='key3', orient='h', kind='boxen', legend=False)
#g = sns.catplot(data=osdf, x='outbreak_size', y='key2', order=ho, col='key3', orient='h', kind='violin', area='area', inner='quartile', legend=False)
#g = sns.catplot(data=osdf, x='outbreak_size', y='key2', order=ho, col='key3', kind='bar', legend=False)
g.set_titles(col_template='{col_name}')
for ax in g.axes.flat:
    ax.set_title(f'{100*float(ax.get_title()):.1f}%')
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.figtext(0.6,0.02,'Outbreak size', ha='center')
plt.tight_layout()
cv.savefig(os.path.join(imgdir, f'OutbreakSize.png'), dpi=300)
###############################################################################

def histplot(**kwargs):
    d = kwargs.pop('data')
    x1 = kwargs.pop('x1')
    x2 = kwargs.pop('x2') if 'x2' in kwargs else None
    col1 = kwargs.pop('color')
    #https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    import colorsys
    c = colorsys.rgb_to_hls(*col1)
    amount = 0.5
    col2 = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    sns.histplot(data=d, x=x1, color=col1, zorder=-1, **kwargs)
    if x2 is not None:
        sns.histplot(data=d, x=x2, color=col2, zorder=1, **kwargs)


### INTRODUCTIONS #############################################################
#g = sns.catplot(data=idf, x='introductions', y='key2', order=ho, col='key3', orient='h', kind='boxen', legend=False)
#g = sns.displot(data=idf, x='introductions_postscreen_per_100_students', row='key2', row_order=ho, col='key3', hue='key2', kind='hist', legend=False, stat='probability', common_norm=False, discrete=True, height=3, aspect=2.5) # , binrange=(0,10)
g = sns.FacetGrid(data=idf, row='key2', row_order=ho, col='key3', hue='key2', height=3, aspect=2.5) # , binrange=(0,10)
g.map_dataframe(histplot, x1='introductions_postscreen_per_100_students', x2=None, stat='probability', common_norm=False, discrete=True)#, stat='probability', common_norm=False)
g.set_titles(col_template='{col_name}')
for ax in g.axes.flat:
    #ax.set_title(f'{100*float(ax.get_title()):.1f}%')
    #ax.set_xlim(0,10)
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.figtext(0.6,0.02,'Number of unique introductions per 100 students per school over 3-months', ha='center')
plt.tight_layout()
cv.savefig(os.path.join(imgdir, f'Introductions.png'), dpi=300)
###############################################################################


### INTRODUCTIONS 2 ###########################################################

g = sns.FacetGrid(data=idf, row='key2', row_order=ho, col='key3', hue='key2', height=3, aspect=2.5) # , binrange=(0,10)
g.map_dataframe(histplot, x1='introductions_postscreen_per_100_students',  x2='introductions_per_100_students', discrete=True)#, stat='probability', common_norm=False)
g.set_titles(col_template='{col_name}')
for ax in g.axes.flat:
    #ax.set_title(f'{100*float(ax.get_title()):.1f}%')
    #ax.set_xlim(0,10)
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.figtext(0.6,0.02,'Number of unique introductions per 100 students per school over 3-months', ha='center')
plt.tight_layout()
cv.savefig(os.path.join(imgdir, f'Introductions2.png'), dpi=300)
###############################################################################

#sns.displot(idf, x='introductions', hue='key3', col='key2', stat='probability', common_norm=False)
#idf_subset = idf.loc[idf['key2']=='No diagnostic screening']
fig, ax = plt.subplots(figsize=(8,6))
##right = 5
##idf.loc[idf['introductions_per_100_students']>right, 'introductions_per_100_students'] = right
cmap = {
    0.01:(0.9603888539940703, 0.3814317878772117, 0.8683117650835491),
    0.005: (0.6423044349219739, 0.5497680051256467, 0.9582651433656727),
    0.002: (0.22335772267769388, 0.6565792317435265, 0.8171355503265633)
}
sns.histplot(isdf, x='introductions_postscreen_per_100_students', hue='key3', stat='probability', binwidth=0.15, common_norm=False, element='step', palette=cmap, legend=True, linewidth=2, ax=ax)
#sns.histplot(idf, x='introductions', hue='key3', stat='probability', binwidth=1, common_norm=False, discrete=True, palette='tab10', legend=True, linewidth=2, ax=ax)
##ax.set_xlim(left=0 ,right=right)
ax.set_xlabel('Introductions post-screen per 100 students over 3-months')
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
handles, labels = ax.get_legend_handles_labels()
#print(handles, labels)
#print([f'{100*float(l):.1f}%' for l in labels])
#ax.legend(handles, [f'{100*float(l):.1f}%' for l in labels], title='Prevalence')
old_legend = ax.legend_
handles = old_legend.legendHandles
labels = [f'{100*float(t.get_text()):.1f}%' for t in old_legend.get_texts()]
ax.legend(handles, labels, title='Prevalence')
plt.tight_layout()
cv.savefig(os.path.join(imgdir, f'Intros.png'), dpi=300)

def f(d):
    s = pd.Series(np.bincount(d['introductions'], minlength=25))
    s = 100 * s / s.sum()
    s['key3'] = d['key3']
    return s

for grp, dat in df.groupby(['key1', 'key2']):
    zzz = dat[['key3', 'rep', 'introductions']].apply(f, axis=1)
    zzz = zzz.set_index('key3').unstack().reset_index()
    zzz.rename({0:'Probability (%)'}, axis=1, inplace=True)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.lineplot(data=zzz, x='level_0', y='Probability (%)', hue='key3', palette='Set1', ax=ax)
    h, l = ax.get_legend_handles_labels()#title='Prevalence')
    l = [f'{100*float(q)}%' for q in l]
    ax.legend(h,l,title='Prevalence')
    ax.set_title(grp)
    ax.set_xlabel('Number of unique introductions per school over 3-months')
    plt.tight_layout()
    fn = '-'.join([g.replace('/','_') for g in grp])
    cv.savefig(os.path.join(imgdir, f'{fn}.png'), dpi=300)
    plt.close(fig)

#d = pd.melt(df, id_vars=['key3', 'rep'], value_vars=['introductions'], var_name='Group', value_name='introductions')
#print(d)
#sns.lineplot(data=d, x='', y=0, hue='key3')

'''
def plot_one(data, color):
    intr = np.vstack([np.bincount(x, minlength=25) for x in data['introductions'].tolist()])
    z = pd.DataFrame(intr, columns=pd.RangeIndex(1,intr.shape[1]+1))
    z = z.div(z.sum(axis=1), axis=0)
    z = z.transpose().stack().reset_index()
    z.rename({0:'count'}, axis=1, inplace=True)
    return sns.lineplot(data=z, x='level_0', y='count', color=color)

#fig, ax = plt.subplots(figsize=(16,10))
def plot_distrib(**kwds):
    data = kwds['data']
    color = kwds['color']
    for grp, dat in data.groupby(['key1', 'key2', 'key3']):
        r = plot_one(dat, color)
    return r

d = pd.melt(df, id_vars=['key1', 'key2', 'key3', 'rep'], value_vars=['introductions'], var_name='Group', value_name='introductions')
g = sns.FacetGrid(data=d, col='key1', row='key2', hue='key3', height=5, aspect=1.4, legend_out=True) # col='key1', 
g.map_dataframe( plot_distrib )#, hue_order=test_order, order=so, palette=test_hue)
g.add_legend()
cv.savefig(os.path.join(imgdir, f'FG_no_leg.png'), dpi=300)
'''

#plt.plot(df['introductions'])



# Wrap up
sc.toc(T)
print('Done.')
