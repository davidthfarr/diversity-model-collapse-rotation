"""Summarizes test perplexity by arm and iteration from checkpoint_metrics.json files.

Works while runs are in progress. D=8 values average over the 8 models and are shown only for iterations
that all 8 models have finished.

Example:
    python experiments/summarize_rotation.py --main-dir versions/rotation_full --plot rotation.png
"""
import argparse
import glob
import json
import os
import re
import statistics
from collections import defaultdict

ARM_ORDER = ['d1_baseline', 'd8_baseline', 'fixed_subset', 'rotated_own', 'rotated_real']
METRICS = {'final': 'reported_test_perp', 'best': 'best_checkpoint_test_perp'}


def load(main_dir, metric):
    per_model = defaultdict(list)  # (arm, seed, iteration) -> values across models
    models = defaultdict(set)  # (arm, seed) -> model folders
    for path in glob.glob(os.path.join(main_dir, '*', '*', 'version_*', 'checkpoint_metrics.json')):
        run, model = path.split(os.sep)[-4:-2]
        match = re.fullmatch(r'(.+)_seed(\d+)', run)
        if not match:
            continue
        arm, seed = match.group(1), int(match.group(2))
        with open(path) as f:
            row = json.load(f)
        per_model[(arm, seed, row['iteration'])].append(row[METRICS[metric]])
        models[(arm, seed)].add(model)
    result = defaultdict(dict)  # arm -> iteration -> {seed: value}
    for (arm, seed, iteration), values in per_model.items():
        if len(values) == len(models[(arm, seed)]):  # skip partly finished D=8 iterations
            result[arm].setdefault(iteration, {})[seed] = statistics.mean(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main-dir', default='versions/rotation_full')
    parser.add_argument('--metric', choices=list(METRICS), default='final',
                        help='final: final-epoch weights, as the paper reports. best: best validation checkpoint.')
    parser.add_argument('--plot', help='Optional path for a PNG of the curves.')
    args = parser.parse_args()

    data = load(args.main_dir, args.metric)
    if not data:
        raise SystemExit(f'No checkpoint_metrics.json files found in {args.main_dir}')
    arms = [a for a in ARM_ORDER if a in data] + sorted(set(data) - set(ARM_ORDER))
    iterations = sorted({i for arm in arms for i in data[arm]})
    print(f'Test perplexity ({args.metric}), mean ± SD across seeds, [number of seeds]')
    print(f'{"arm":<14}' + ''.join(f'{"t=" + str(i):>18}' for i in iterations))
    for arm in arms:
        cells = []
        for i in iterations:
            values = list(data[arm].get(i, {}).values())
            if not values:
                cells.append('')
            elif len(values) == 1:
                cells.append(f'{values[0]:.1f} [1]')
            else:
                cells.append(f'{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f} [{len(values)}]')
        print(f'{arm:<14}' + ''.join(f'{c:>18}' for c in cells))

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm in arms:
            points = sorted((i, statistics.mean(v.values())) for i, v in data[arm].items())
            ax.plot(*zip(*points), marker='o', label=arm)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'Test perplexity ({args.metric})')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f'Saved {args.plot}')


if __name__ == '__main__':
    main()
