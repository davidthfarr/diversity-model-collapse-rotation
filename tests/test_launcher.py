"""Runs the launcher and summary end to end with the offline harness instead of real models.

Usage: python tests/test_launcher.py
"""
import glob
import os
import shlex
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp = tempfile.mkdtemp()
main_dir = os.path.join(tmp, 'runs')
command = f'{shlex.quote(sys.executable)} tests/harness.py . {os.devnull}'
launch = [sys.executable, 'experiments/run_rotation.py', '--seeds', '0', '1', '--iterations', '3',
          '--gpus', '0', '1', '--jobs-per-gpu', '2', '--main-dir', main_dir, '--command', command,
          '-s', 'auto', '-w', '0', '-b', '4', '-m', '2', '-blsz', '8']


def check(name, condition):
    print(('PASS ' if condition else 'FAIL ') + name)
    if not condition:
        raise SystemExit(1)


result = subprocess.run(launch, cwd=ROOT, capture_output=True, text=True)
print(result.stdout[-1500:], result.stderr[-1500:])
check('all 10 jobs finish', result.returncode == 0 and len(glob.glob(f'{main_dir}/*/.done')) == 10)
starts = [line for line in result.stdout.splitlines() if ' start ' in line]
check('seed 0 jobs start first', all('seed0' in line for line in starts[:5]))
check('both GPUs are used', {'GPU 0', 'GPU 1'} <= {line.split(' on ')[-1] for line in starts})
check('no dataset after the final iteration', not glob.glob(f'{main_dir}/*/*/version_2/dataset_version_2.pkl'))
check('D=8 writes metrics for every model',
      len(glob.glob(f'{main_dir}/d8_baseline_seed0/*/version_2/checkpoint_metrics.json')) == 8)
again = subprocess.run(launch, cwd=ROOT, capture_output=True, text=True)
check('a rerun skips finished jobs', again.returncode == 0 and again.stdout.count('already done') == 10)

summary = subprocess.run([sys.executable, 'experiments/summarize_rotation.py', '--main-dir', main_dir,
                          '--plot', os.path.join(tmp, 'curves.png')], cwd=ROOT, capture_output=True, text=True)
print(summary.stdout, summary.stderr[-800:])
check('summary covers 5 arms and 2 seeds', summary.returncode == 0 and summary.stdout.count('[2]') == 15)
check('summary plot is written', os.path.exists(os.path.join(tmp, 'curves.png')))
print('All checks passed')
