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

# Global plotting styles
font_size = 18
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style

# Other configuration
folder = 'v2020-12-02'
#variant = 'k5_AntigenFreq_10reps'
variant = 'k5_PCROptimistic_Sweep1reps'

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
    'Antigen every 1w teach&staff, PCR f/u':    ('Weekly antigen for teachers & staff, PCR f/u',    reds(1/5)),
    'Antigen every 2w, no f/u':                 ('Fortnightly antigen, no f/u',                     reds(2/5)),
    'Antigen every 2w, PCR f/u':                ('Fortnightly antigen, PCR f/u',                    reds(3/5)),
    'PCR every 2w':                             ('Fortnightly PCR, 1d delay',                       blues(3/6)),
    'Antigen every 1w, PCR f/u':                ('Weekly antigen, PCR f/u',                         reds(4/5)),
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
    #ret['introductions'] = np.zeros(10)
    ret['introductions'] = []
    ret['introductions_per_100_students'] = []
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
        ret['introductions_per_100_students'].append( len(stats['outbreaks'])/stats['num']['students'] * 100 )
        ret['outbreak_size'] += [ob['Infected Student'] + ob['Infected Teacher'] + ob['Infected Staff'] for ob in stats['outbreaks']]

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

print('WARNING: Removing 0.1%')
df = df.loc[df['key3']>0.001]

################
#%% PLOT RESULTS
################

#msim = cv.MultiSim(sims)
#msim.plot(to_plot={'Prevalence':['n_exposed']}, do_show=False)
#cv.savefig(os.path.join(imgdir, f'Prevalence.png'), dpi=300)

def get_zgrid(zlab, xlab, ylab, df):
    from sklearn import preprocessing
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern

    x = df[xlab].values.ravel()
    y = df[ylab].values.ravel()
    z = df[zlab].values.ravel()
    xi, yi = np.meshgrid(np.linspace(0, 0.015, 20), np.linspace(0, 12000, 20))
    X = np.vstack((x, y, z)).T
    scaler = preprocessing.StandardScaler().fit(X)
    sX = scaler.transform(X)
    #kernel = 1.0 * RBF(length_scale=[0.1,10], length_scale_bounds=(1e-2, 1e6)) \
    #         + WhiteKernel(noise_level_bounds=(1e-10, 1e+1))
    kernel = 1.0 * Matern(nu=1.5,length_scale=[0.1,10], length_scale_bounds=(1e-2, 1e6)) + WhiteKernel(noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0)
    #print('PARS:', gp.get_params())
    gp.fit(sX[:, :2], sX[:, -1])
    print('PARS:', gp.kernel_)
    sXi = scaler.transform(np.vstack((xi.flatten(), yi.flatten(), np.zeros(xi.shape).flatten())).T)
    szi_lin = gp.predict(sXi[:, :2])
    zi_lin = scaler.inverse_transform(np.vstack((np.ones((2, szi_lin.size)), szi_lin)).T)
    zi_lin = zi_lin[:, -1].reshape(xi.shape)

    return xi, yi, zi_lin


'''
import seaborn as sns
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")
print(tips)
exit()
'''

d = df[['key1', 'key2', 'key3', 'outbreak_size']]
l = []
for k,dat in d.groupby(['key1', 'key2', 'key3']):
    for entry in np.hstack(dat['outbreak_size']):
        l.append([k[0], k[1], k[2], entry])
osdf = pd.DataFrame(l, columns=['key1', 'key2', 'key3', 'outbreak_size'])

d = df[['key1', 'key2', 'key3', 'introductions']]
l = []
for k,dat in d.groupby(['key1', 'key2', 'key3']):
    for entry in np.hstack(dat['introductions']):
        l.append([k[0], k[1], k[2], entry])
idf = pd.DataFrame(l, columns=['key1', 'key2', 'key3', 'introductions'])


d = df[['key1', 'key2', 'key3', 'introductions_per_100_students']]
l = []
for k,dat in d.groupby(['key1', 'key2', 'key3']):
    for entry in np.hstack(dat['introductions_per_100_students']):
        l.append([k[0], k[1], k[2], entry])
isdf = pd.DataFrame(l, columns=['key1', 'key2', 'key3', 'introductions_per_100_students'])

####################

#mat = pd.crosstab(isdf['key3'], isdf['introductions_per_100_students']) # All testing scenarios ...

norm = mplt.colors.LogNorm()#vmin=1, vmax=50)
h, xe, ye, img = plt.hist2d(isdf['key3'], isdf['introductions_per_100_students'], bins=20, norm=norm, cmap='inferno')
plt.show()
exit()

'''
xx,yy = np.meshgrid(xe[:-1], ye[:-1])
#tmp = pd.DataFrame({'X':xx.ravel(), 'Y':yy.ravel(), 'Z':np.log10(1+h.T.ravel())})
tmp = pd.DataFrame({'X':xx.ravel(), 'Y':yy.ravel(), 'Z':h.T.ravel()})
xi, yi, zi_lin = get_zgrid('Z', 'X', 'Y', tmp)
fig, ax = plt.subplots(figsize=(8,5))
from matplotlib import ticker, cm
cs = ax.contourf(xi, yi, zi_lin)#, levels=levels, colors=['k']) , locator=ticker.LogLocator()

plt.figure()
xi, yi, zi_lin = get_zgrid('Z', 'X', 'Y', tmp)
fig, ax = plt.subplots(figsize=(8,5))
cs = ax.contour(xi, yi, zi_lin)#, levels=levels, colors=['k'])

plt.figure()
#plt.scatter(isdf['key3'], isdf['introductions'])#, c=[float(x) for x in df['key3']])
X,Y = np.meshgrid(isdf['key3'].unique(), isdf['introductions_per_100_students'].unique())
plt.contourf(X,Y,np.log(mat).T)

plt.figure()
sns.heatmap(np.log(1+mat))
'''




#ho = ['No diagnostic screening', 'Antigen every 4w, PCR f/u', 'Fortnightly antigen, PCR f/u', 'Weekly antigen, PCR f/u']
ho = [test_names[x][0] for x in ['None', 'PCR every 4w', 'PCR every 2w', 'PCR every 1w']]
#osdf['log10_outbreak_size'] = np.log10(osdf['outbreak_size'])
osdf = osdf[osdf['outbreak_size'] > 1]
g = sns.catplot(data=osdf, x='outbreak_size', y='key2', order=ho, col='key3', orient='h', kind='boxen', legend=False)
#g = sns.catplot(data=osdf, x='outbreak_size', y='key2', order=ho, col='key3', orient='h', kind='violin', area='area', inner='quartile', legend=False)
g.set_titles(col_template='{col_name}')
for ax in g.axes.flat:
    ax.set_title(f'{100*float(ax.get_title()):.1f}%')
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.figtext(0.6,0.02,'Outbreak size', ha='center')
plt.tight_layout()
cv.savefig(os.path.join(imgdir, f'OutbreakSize.png'), dpi=300)

g = sns.catplot(data=isdf, x='introductions_per_100_students', y='key2', order=ho, col='key3', orient='h', kind='boxen', legend=False)
g.set_titles(col_template='{col_name}')
for ax in g.axes.flat:
    ax.set_title(f'{100*float(ax.get_title()):.1f}%')
    ax.set_ylabel('')
    ax.set_xlabel('')
plt.figtext(0.6,0.02,'Number of unique introductions per school over 3-months', ha='center')
plt.tight_layout()
cv.savefig(os.path.join(imgdir, f'Introductions.png'), dpi=300)


#sns.displot(idf, x='introductions', hue='key3', col='key2', stat='probability', common_norm=False)
#idf_subset = idf.loc[idf['key2']=='No diagnostic screening']
fig, ax = plt.subplots(figsize=(8,5))
idf.loc[idf['introductions']>40, 'introductions'] = 40
cmap = {
    0.01:(0.9603888539940703, 0.3814317878772117, 0.8683117650835491),
    0.005: (0.6423044349219739, 0.5497680051256467, 0.9582651433656727),
    0.002: (0.22335772267769388, 0.6565792317435265, 0.8171355503265633)
}
sns.histplot(idf, x='introductions', hue='key3', stat='probability', binwidth=1, common_norm=False, element='step', palette=cmap, legend=True, linewidth=2, ax=ax)
#sns.histplot(idf, x='introductions', hue='key3', stat='probability', binwidth=1, common_norm=False, discrete=True, palette='tab10', legend=True, linewidth=2, ax=ax)
ax.set_xlim(0,40)
ax.set_xlabel('Number of unique introductions over 3-months')
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
    zzz = df[['key3', 'rep', 'introductions']].apply(f, axis=1)
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

'''
d = pd.melt(df, id_vars=['key1', 'key2', 'key3', 'rep'], value_vars=['introductions'], var_name='Group', value_name='introductions')
g = sns.FacetGrid(data=d, col='key1', row='key2', hue='key3', height=5, aspect=1.4, legend_out=True) # col='key1', 
g.map_dataframe( plot_distrib )#, hue_order=test_order, order=so, palette=test_hue)
g.add_legend()
cv.savefig(os.path.join(imgdir, f'FG_no_leg.png'), dpi=300)
'''

#plt.plot(df['introductions'])

l = []
for sim in sims:
    t = sim.results['t'] #pd.to_datetime(sim.results['date'])
    y = 100 * sim.results['n_exposed'].values / sim.pars['pop_size']
    d = pd.DataFrame({'Prevalence': y}, index=pd.Index(data=t, name='Date'))
    d['Prevalence Target'] = f'{100 * sim.key3}%'
    d['Scenario'] = f'{sim.key1} + {sim.key2}'
    d['Rep'] = sim.eidx
    l.append( d )
d = pd.concat(l).reset_index()
fig, ax = plt.subplots(figsize=(16,10))
sns.lineplot(data=d, x='Date', y='Prevalence', hue='Prevalence Target', style='Scenario', palette='Set1', ax=ax)
cv.savefig(os.path.join(imgdir, f'Prevalence.png'), dpi=300)


# Wrap up
sc.toc(T)
print('Done.')
