"""Runs the rotation experiment as a queue of jobs spread across GPUs.

Jobs run seed by seed, so seed 0 finishes first and can be read as a pilot while later seeds continue.
Within a seed, the longest arms start first. Each job writes a log to MAIN_DIR/logs/ and a .done marker
to its run folder when it finishes, so rerunning the same command skips completed jobs.

Example:
    python experiments/run_rotation.py --gpus 0 1 --jobs-per-gpu 2

Arguments not listed here are passed to every job, for example --num_beams 1.
"""
import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARMS = {  # longest first
    'd8_baseline': ['--number_models', '8'],
    'd1_baseline': ['--number_models', '1'],
    'rotated_own': ['--number_models', '1', '--arm', 'rotated_own'],
    'fixed_subset': ['--number_models', '1', '--arm', 'fixed_subset'],
    'rotated_real': ['--number_models', '1', '--arm', 'rotated_real'],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--model', default='opt125')
    parser.add_argument('--gpus', nargs='+', default=['0'], help='GPU indices to use.')
    parser.add_argument('--jobs-per-gpu', type=int, default=1)
    parser.add_argument('--main-dir', default='versions/rotation_full')
    parser.add_argument('--dry-run', action='store_true', help='Print the jobs without running them.')
    parser.add_argument('--command', default=f'{shlex.quote(sys.executable)} evolution/train_gen_loop.py',
                        help=argparse.SUPPRESS)  # replaced in tests
    return parser.parse_known_args()


def build_jobs(args, extra):
    jobs = []
    for seed in args.seeds:
        for arm in args.arms:
            name = f'{arm}_seed{seed}'
            command = shlex.split(args.command) + [
                '--run_name', name, '--main_dir', args.main_dir, '--model_tag', args.model,
                '--number_loops', str(args.iterations), '--seed', str(seed), '--partition_seed', str(seed),
                '--log_checkpoint_metrics', '--skip_final_generate', *ARMS[arm], *extra]
            jobs.append((name, command))
    return jobs


def main():
    args, extra = parse_args()
    os.chdir(ROOT)
    pending = queue.Queue()
    for name, command in build_jobs(args, extra):
        run_dir = os.path.join(args.main_dir, name)
        if os.path.exists(os.path.join(run_dir, '.done')):
            print(f'skip {name}: already done')
        elif os.path.exists(run_dir) and not args.dry_run:
            # A partial run would load stale checkpoints, so it has to be removed first.
            raise SystemExit(f'{run_dir} exists from an unfinished run. Delete it or use another --main-dir.')
        else:
            pending.put((name, command))
    if args.dry_run:
        while not pending.empty():
            print(shlex.join(pending.get()[1]))
        return

    log_dir = os.path.join(args.main_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    failed, lock = [], threading.Lock()

    def report(message):
        with lock:
            print(f'{time.strftime("%H:%M:%S")} {message}', flush=True)

    def worker(gpu):
        while True:
            try:
                name, command = pending.get_nowait()
            except queue.Empty:
                return
            report(f'start {name} on GPU {gpu}')
            start = time.time()
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu}
            with open(os.path.join(log_dir, f'{name}.log'), 'w') as log:
                code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            minutes = (time.time() - start) / 60
            if code == 0:
                open(os.path.join(args.main_dir, name, '.done'), 'w').close()
                report(f'done  {name} in {minutes:.0f} min')
            else:
                with lock:
                    failed.append(name)
                report(f'FAIL  {name} after {minutes:.0f} min (exit {code}), see {log_dir}/{name}.log')

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus for _ in range(args.jobs_per_gpu)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failed:
        raise SystemExit(f'{len(failed)} job(s) failed: {", ".join(failed)}')
    print('All jobs finished')


if __name__ == '__main__':
    main()
