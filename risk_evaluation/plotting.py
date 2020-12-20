import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np

# Global plotting styles
font_size = 18
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style


#Plotting
def plot_tree(tree, stats, n_days, do_show=False):
    fig, ax = plt.subplots(figsize=(16,10))
    date_range = [n_days, 0]

    # TODO: move tree plotting to a function
    #print(f'Tree {i}', sid, sim.key1, sim.key2, sim.key2)
    #for u,v,w in tree.edges.data():
        #print('\tEDGE', u,v,w)
    #print(f'N{i}', sid, sim.key1, sim.key2, sim.key2, tree.nodes.data())
    for j, (u,v) in enumerate(tree.nodes.data()):
        #print('\tNODE', u,v)
        recovered = n_days if np.isnan(v['date_recovered']) else v['date_recovered']
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
                    plt.plot(t, j, marker='x', color='red', ms=10, lw=2)
                elif int(u) in outcomes['Negative']:
                    plt.plot(t, j, marker='x', color='green', ms=10, lw=2)

    for t, r in stats['testing'].items():
        ax.axvline(x=t, zorder=-100)
    date_range[1] = min(date_range[1], n_days)
    ax.set_xlim(date_range)
    ax.set_xticks(range(int(date_range[0]), int(date_range[1])))
    ax.set_yticks(range(0, len(tree.nodes)))
    ax.set_yticklabels([f'{int(u)}: {v["type"]}, age {v["age"]}' for u,v in tree.nodes.data()])
    #ax.set_title(f'School {sid}, Tree {i}')

    if do_show:
        plt.show()
    else:
        return fig
