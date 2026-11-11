"""Checks each arm's data flow and that default behavior matches the upstream commit.

Usage: python tests/test_rotation_arms.py [UPSTREAM_CHECKOUT]
Uses a tiny random model and synthetic tokens, so it tests plumbing, not scientific results.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ['--model_tag', 'opt125', '-s', 'auto', '-w', '0', '-b', '4', '-m', '2', '-blsz', '8', '-nl', '3']
tmp = tempfile.mkdtemp()
counter = 0


def run(*flags, repo=ROOT):
    global counter
    counter += 1
    output = os.path.join(tmp, f'{counter}.json')
    subprocess.run([sys.executable, os.path.join(ROOT, 'tests', 'harness.py'), repo, output, *BASE,
                    '--main_dir', os.path.join(tmp, 'runs'), '--run_name', f'run{counter}', *flags],
                   check=True, cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(output) as f:
        result = json.load(f)
    for key in ('train', 'prompts'):
        result[key] = [{tuple(r) for r in rows} if isinstance(rows, list) else rows for rows in result[key]]
    result['generated'] = {k: {tuple(r) for r in v} for k, v in result['generated'].items()}
    return result


def check(name, condition):
    print(('PASS ' if condition else 'FAIL ') + name)
    if not condition:
        raise SystemExit(1)


one = ['--number_models', '1']
baseline = run(*one)
ecosystem = run('--number_models', '8', '-nl', '1')
segment = ecosystem['train'][0]
pool = baseline['train'][0]
check('N is 64 synthetic blocks and D=8 model 0 has 8', baseline['train_counts'][0] == 64 and ecosystem['train_counts'][0] == 8 and segment <= pool)
check('baseline prompts cover N every iteration', all(p == pool for p in baseline['prompts']))
check('baseline trains on its previous outputs', baseline['train'][1] == baseline['generated']['opt125-s0/0'])

fixed = run(*one, '--arm', 'fixed_subset')
check('fixed_subset starts on the D=8 segment', fixed['train'][0] == segment)
check('fixed_subset always prompts from that segment', all(p == segment for p in fixed['prompts']))
check('fixed_subset trains on all its previous outputs',
      fixed['train'][1] == fixed['generated']['opt125-s0/0'] and fixed['train_counts'][1] == 8)

own = run(*one, '--arm', 'rotated_own')
check('rotated_own starts on the D=8 segment', own['train'][0] == segment)
check('rotated_own prompts from all of N every iteration', all(p == pool for p in own['prompts']))
for k in (1, 2):
    check(f'rotated_own t={k} trains on 8 of its own t={k - 1} outputs',
          own['train_counts'][k] == 8 and own['train'][k] <= own['generated'][f'opt125-s0/{k - 1}'])
check('rotated_own has no real blocks after t=0', not (own['train'][1] & pool) and not (own['train'][2] & pool))
check('rotated_own draws differ across iterations', own['train'][1] != own['train'][2])

real = run(*one, '--arm', 'rotated_real')
check('rotated_real starts on the D=8 segment', real['train'][0] == segment)
for k in (1, 2):
    check(f'rotated_real t={k} trains on 8 real blocks from N', real['train_counts'][k] == 8 and real['train'][k] <= pool)
check('rotated_real draws differ across iterations', real['train'][1] != real['train'][2])
check('rotated_real skips generation', not real['generated'] and not any(real['prompts']))

reseeded = run('--number_models', '8', '-nl', '1', '--partition_seed', '1')
check('partition_seed changes the segment', reseeded['train'][0] != segment)
check('seed changes the rotation draw', run(*one, '--arm', 'rotated_real', '--partition_seed', '1')['train'][1] != real['train'][1])

logged = run(*one, '--log_checkpoint_metrics')
check('logging leaves generated data unchanged', logged['generated_sha256'] == baseline['generated_sha256'])
metric = logged['metrics']['opt125-s0/2']
check('metrics report both model states', {'best_epoch', 'best_checkpoint_test_perp', 'final_epoch_test_perp'} <= set(metric))
check('final-epoch evaluation matches the reported value',
      abs(metric['final_epoch_test_perp'] - metric['reported_test_perp']) < 1e-2 * metric['reported_test_perp'])

skipped = run(*one, '--log_checkpoint_metrics', '--skip_final_generate')
check('skip_final_generate drops only the last dataset',
      set(skipped['generated']) == {'opt125-s0/0', 'opt125-s0/1'} and
      all(skipped['generated'][k] == logged['generated'][k] for k in skipped['generated']))
check('skip_final_generate leaves metrics unchanged', all(
    abs(skipped['metrics'][k]['reported_test_perp'] - logged['metrics'][k]['reported_test_perp']) < 1e-6
    for k in logged['metrics']))
eco_skip = run('--number_models', '2', '--skip_final_generate')
check('skip_final_generate works for ecosystems', len(eco_skip['generated']) == 4)

greedy = run(*one, '--num_beams', '1', '-nl', '1')
check('num_beams changes generation', greedy['generated_sha256'] != run(*one, '-nl', '1')['generated_sha256'])

if len(sys.argv) > 1:
    upstream = sys.argv[1]
    for flags in (one, ['--number_models', '2'], ['--number_models', '8'], [*one, '--train_only'],
                  [*one, '--generate_percentage', '.1'], [*one, '--temperature', '1']):
        a, b = run(*flags), run(*flags, repo=upstream)
        same = a['generated_sha256'] == b['generated_sha256'] and a['train'] == b['train'] and a['prompts'] == b['prompts']
        check(f'matches upstream: {" ".join(flags)}', same)
print('All checks passed')
