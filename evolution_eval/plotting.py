import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TABLEAU_COLORS


from helpers import parse_run_names_df, \
    combine_means_sds, load_perplexity_mean, load_perplexity_stdev, \
    load_versions, \
    macro_perplexity, load_versions_plot, load_metric_files, load_args_defaults

# If using the serif variant (Libertinus Serif)
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Linux Libertine', ]
plt.rcParams.update({'font.size': 18})


def plot_perplexity_paper(args):
    runs_d = load_name_d(args)
    for model_name, runs in runs_d.items():
        plot_perplexity(args, model_name=model_name, runs=runs)

def plot_perplexity(args, runs, model_name, number_loops=10, epochs=5):
    df = load_perplexity_mean(runs, args)
    df_s = load_perplexity_stdev(runs, args)
    df = parse_run_names_df(df)
    df_s = parse_run_names_df(df_s)
    cols_num = load_versions(number_loops, epochs)
    xvalues = load_versions_plot(number_loops)
    assert len(xvalues) == len(cols_num)
    test_samples_wikitext2 = 2209
    # Set the figure size
    plt.figure(figsize=(12, 6))
    df_groups = df.groupby('rundate').size()
    colors = load_colors_paper(df_groups.shape[0])
    for i, (rundate, nr_models) in enumerate(df_groups.sort_values().items()):
        df_means = df[df['rundate'] == rundate][cols_num]
        df_std = df_s[df_s['rundate'] == rundate][cols_num]
        mean_combined, std_combined = combine_means_sds(df_means, df_std, test_samples_wikitext2)
        color = colors[i]
        model_name_i = rundate if model_name is None else model_name
        # Plot the data
        plt.fill_between(mean_combined.index, mean_combined.values - std_combined,
                         mean_combined.values + std_combined, alpha=0.05, color=color)
        plt.plot(mean_combined.index, mean_combined.values, label=f'M={nr_models}, {model_name_i}',
                 marker='x', linestyle='--', color=color)


    plt.xlabel('Iterations t')
    plt.ylabel('Perplexity µ ± σ') # combined mean perplexity
    plt.xticks(cols_num, xvalues)

    # Show the legend in two columns
    plt.legend(ncol=2)

    # Show the plot
    plt.tight_layout()
    plt.show()




def load_colors_paper(number_plots=None):
    colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854']
    if number_plots and number_plots > len(colors):
        colors = list(TABLEAU_COLORS.values())  # ['blue', 'orange'])
    return colors

def macro_perplexity_numbers(args):
    def set_titles():
        plt.xlabel('Hill-Shannon Diversity D')
        plt.ylabel('Aggregated Perplexity µ')
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.show()

    model_markers = ['v', 'o'] # 'o', 's']
    colors = load_colors_paper()
    cols = load_versions()[1:]  # without version_0_0
    runs_d = load_name_d(args)
    plt.figure(figsize=(12, 6))
    for i, (model_name, prio_runs) in enumerate(runs_d.items()):
        df = macro_perplexity(args, prio_runs)

        df.sort_values(by=['number_models'], inplace=True)
        pp = df[cols].mean(axis=1)
        x = df['number_models'].values
        pp.name = 'perplexity'
        color = colors[i]
        marker = model_markers[i]
        # Plot the data
        plt.scatter(x, pp,   color=colors[:len(x)],  marker=marker, s=120, label=f'{model_name}') # , alpha=0.7
        plt.plot(x, pp, linestyle='--',   color='black', alpha=0.5)
    set_titles()

    plt.figure(figsize=(12, 6))
    num_lines = len(cols)
    #colors = ['blue', 'orange']
    for k, model_name in enumerate(df.model_name.unique()):
        dfm = df[df['model_name'] == model_name].copy()
        dfm.sort_values(by=['number_models'], inplace=True)
        line_color = colors[k]
        for i, c in enumerate(cols):
            pp = dfm[c]
            pp.name = 'perplexity'
            # Plot the data
            x = dfm['number_models'].values
            # Get the color from the colormap
            if i == 0:
                plt.plot(x, pp, linestyle='--', label=f'{model_name}', color=line_color, marker='o', alpha=0.3)
            else:
                plt.plot(x, pp, linestyle='--', color=line_color, marker='o', alpha=0.3)


def macro_optimum_iterations_variations(args):
    offset = True
    model_markers = [4, 5, 6, 7]
    markersize = 12 if offset else 8
    runs_d = load_name_d(args)
    nr_runs = len(runs_d[list(runs_d.keys())[0]])
    nr_colors = nr_runs
    colors = load_colors_paper(nr_colors)
    model_numbers = [1, 2, 4, 8, 16] if nr_runs == 5 else [1, 2, 4, 16][:nr_runs]
    plt.figure(figsize=(12, 6))
    number_loops = 10
    cols_num = load_versions(number_loops=number_loops)
    x = load_versions_plot(number_loops=number_loops)[1:]
    x = [str(xi + 1) for xi in x]
    max_d = 1
    for i, (model_name, prio_runs) in enumerate(runs_d.items()):
        df = macro_perplexity(args, prio_runs,)
        df.index = df['number_models']
        if i == 1:
            df.sort_index(inplace=True, ascending=False)
        y = []
        colors_t, markers_t = [], []
        for t in range(number_loops):
            dft = df[cols_num[1:t+2]]
            opt_d = dft.mean(axis=1).idxmin()
            y.append(opt_d)
            max_d = max(max_d, opt_d)
            index = model_numbers.index(opt_d)
            colors_t.append(colors[index])
        marker = model_markers[i % len(colors)]
        plt.scatter(x, y, color=colors_t, marker=marker, s=150, label=f'{model_name}', linewidths=2)  # , alpha=0.7
        plt.plot(x, y, linestyle='--', color='black', alpha=0.3)

    plt.xlabel('Number of Iterations $T$')
    plt.ylabel('Optimum Level of Diversity ')
    specific_integers = model_numbers[:model_numbers.index(max_d)+1]

    # Set the y-axis tick locations
    plt.yticks(specific_integers)
    if max_d == 16:
        plt.ylim(0, max_d+2)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()

def main():
    args = load_args_defaults()
    load_metric_files(args)
    if args.plot == 'perplexity':
        plot_perplexity_paper(args)
    elif args.plot == 'optimal_diversity':
        macro_optimum_iterations_variations(args)
    elif args.plot == 'perplexity_diversity':
        macro_perplexity_numbers(args)
    else:
        raise ValueError('Invalid arguments')

def load_name_d(args):
    model_tag = args.model_tag
    d = {model_tag: args.run_names}
    return d


if __name__ == '__main__':
    main()
