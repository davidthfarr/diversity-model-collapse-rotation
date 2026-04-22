import argparse
import os
import re

import numpy as np
import pandas as pd


def arguments_defaults():
    arguments = {
        ('-main_dir', '--main_dir'): {
            'type': str,
            'default': 'versions',
            'help': 'Name of the saved model to save.'
        },
        ('-run_names', '--run_names'): {
            'nargs': '+',
            'type': str,
            'default': None,
            'help': 'Run names to use. Use it like this: --run_names run1 run2. ',
        },
        ('-plot', '--plot'): {
            'type': str,
            'default': 'perplexity',  # optimal_diversity
            'help': 'Types of plot: perplexity, optimal_diversity, perplexity_diversity.'
        },
        ('-model_tag', '--model_tag'): {
            'type': str,
            'default': 'Model1',
            'help': 'Model family name for the evaluated models',
        },


    }
    return arguments




def load_args_defaults():
    arguments = arguments_defaults()
    p = argparse.ArgumentParser(description='Loop')
    for k, v in arguments.items():
        p.add_argument(*k, **v)
    a = p.parse_args()
    return a


def load_metric_files(args):
    def save_df(col_n, file_name):
        dfk = pd.DataFrame([s[col_n] for s in combined_data])  # pd.DataFrame(combined_data)

        dfk.index = names
        dfk.index.name = col_n
        # sort the dataframe by the column name and then by the 'prio' column
        cols_num = load_versions(20)
        cols_num_e10 = load_versions(10, 10)
        dfk = dfk[cols_num + cols_num_e10[1:]]  # avoid duplicate version_0_0
        dfk.sort_index(inplace=True)# avoid duplicate version_0_0

        # Write the final dataframe to a new CSV file
        fp = os.path.join(args.main_dir, file_name)
        dfk.to_csv(fp)
        print(f'CSV files have been combined into {fp}')
    combined_data = []

    col_name = 'test_mean_perp'
    col_std = 'std_perps'
    cols = [col_name, col_std]
    # Loop through the subfolders (version_0, version_1, etc.)
    for file_name in os.listdir(args.main_dir):
        fp = os.path.join(args.main_dir, file_name)

        # Only process subfolders that start with 'version_'
        if os.path.isfile(fp) and file_name.endswith('metrics.csv'):
            print(f'Processing {file_name}')

            df = pd.read_csv(fp)

            dfi = df[~df[col_name].isna()]
            dfi.index = ['_'.join([str(v), str(e)]) for v, e in zip(dfi['version'], dfi['epoch'])]
            s = dfi[cols]
            s.name = '__'.join([df['run_name'].iloc[0],  df['model_tag'].iloc[0]])
            # Append the dataframe to the list
            combined_data.append(s)

    # Concatenate all dataframes into one
    names = [s.name for s in combined_data]
    save_df(col_name, f'perplexity_all.csv')
    save_df(col_std, f'perplexity_all_stdev.csv')





def parse_run_names_df(df):


    def name_correction(name_str):
        base_str, model_name = re.split(r'__', name_str)
        return base_str + '__' + model_name

    df['name'] = df.index
    df['name'] = df.name.apply(lambda x: name_correction(x))
    df['runname'] = df.name.apply(lambda x: re.split(r'_\d+', x)[0])

    df['rundate'] = df.name.apply(lambda x: re.split(r'__', x)[0])
    df['model'] = df.name.apply(lambda x: re.split(r'__', x)[1])
    df['model_name'] = df.model.apply(lambda x: re.split(r'-', x)[0])

    df['model_tag'] = df['model']

    return df






def combine_means_sds(means_df, sds_df, sample_size):
    """
    Returns: (combined_means: pd.Series, combined_sds: pd.Series)
    """
    nr_groups = means_df.shape[0]  # number of models
    n = sample_size
    n_total = nr_groups * n
    mu = means_df.mean(axis=0)  # equal n → simple average

    # Sum of squares within groups (uses sample SDs)
    sds_within = ((n - 1) * (sds_df ** 2)).sum(axis=0)

    # Sum of squares between groups
    sds_between = (n * (means_df.sub(mu, axis=1) ** 2)).sum(axis=0)

    var = (sds_within + sds_between) / (n_total - 1)  # unbiased sample variance
    sd = np.sqrt(var)

    return mu, sd




def load_perplexity_mean(prio_runs, args):

    df = load_perplexity_vals(prio_runs, args, file_name='perplexity_all.csv', col_name='test_mean_perp', )
    return df


def load_perplexity_stdev(prio_runs, args):
    df = load_perplexity_vals(prio_runs, args, file_name='perplexity_all_stdev.csv', col_name='std_perps', )
    return df


def load_perplexity_vals(prio_runs, args,
                         file_name='perplexity_all.csv', col_name='test_mean_perp', ):

    df = pd.read_csv(os.path.join(args.main_dir, file_name), index_col=0)
    # exlude post hoc evalution if not explicitly requested
    prio_runs_filter = [f'{run}__' for run in prio_runs]
    df = df[df.index.astype(str).str.contains('|'.join(prio_runs_filter))]
    # remove the prio column

    df.sort_values(by=[col_name], inplace=True)
    # fill NaN values with previous values
    df.ffill(axis=1, inplace=True)

    return df


def load_versions(number_loops=10, epochs=5):
    cols_num = ['version_0_0'] + [f'version_{i}_{epochs}' for i in range(number_loops)]
    return cols_num

def load_versions_plot(number_loops=10):
    cols_num = [-1] + list(range(number_loops))
    return cols_num


def load_versions_diversity(number_loops=10):
    cols_num = [f'version_{i}' for i in range(number_loops)]
    return cols_num



def macro_perplexity(args, runs, epochs=5):
    df = load_perplexity_mean(runs, args)
    df = parse_run_names_df(df)
    cols_num = load_versions(epochs=epochs)  # without version_0_0
    index, values = [], []
    for model_name in df.model_name.unique():

        dfm = df[df['model_name'] == model_name]
        for rundate in dfm.rundate.unique():
            dfi = dfm[dfm['rundate'] == rundate]
            s = dfi[cols_num].mean(axis=0)
            nr_models = dfi.shape[0]
            index.append(rundate)
            values.append(s.values.tolist() + [nr_models, model_name, rundate])
    df = pd.DataFrame(values, columns= cols_num + ['number_models', 'model_name', 'rundate'], index=index)
    return df



