# diversity-model-collapse

This repository is based on [hodeld/diversity-model-collapse](https://github.com/hodeld/diversity-model-collapse) at commit `cb73c24`. That repository has no license file, so its code remains the original authors' work and is not covered by this repository's `LICENSE`.

Code for the paper "Epistemic diversity across models mitigates knowledge collapse"

## Minimal setup
Install dependencies using  `pip install -r requirements.txt`

## Run code

### Model evolution
To conduct the experiments in the paper, run `evolution/train_gen_loop.py` using the following basic arguments: 
* `--run_name` to define the name of the run, 
* `--main_dir` to define the directory for storing the model iterations,
* `--model_tag` to define the model (`opt125`, `opt350`, `gpt2`, `gpt2m`), 
* `--number_models` to define the number of models per ecosystem (1, 2, 4, 16).  

For example, to run the evolution of two opt125 models (D=M=2), use the following parameters:

`python evolution/train_gen_loop.py --run_name RUN_NAME --main_dir MAIN_DIR --model_tag opt125  --number_models 2`

To run experimental variations involving larger model size and dataset size, use the following arguments and parameters:
* `--model_tag` to define a larger model size (`opt350`, `gpt2m`), 
* `--dataset_fraction 0.5` to use the entire Wikitext2 dataset (dataset_fraction refers to 1/2 of the total dataset fraction per ecosystem)

To run experimental variations involving real training data and temperature sampling, use the following arguments:
* `--generate_percentage` to define the fraction of real data at each iteration step. For example,  `--generate_percentage .1` sets the fraction to  10%.
* `--temperature`: To define the sampling temperature. For example, `--temperature .5` sets the sampling temperature to 0.5.

### Plotting
To plot the model perplexities, run `evolution_eval/plotting.py` using the following basic arguments: 
* `--run_names` to define the names of the runs, 
* `--main_dir` to define the directory for storing the model iterations,
* `--model_tag` to define the model label, 
* `--plot` to define the plot type:
  * `perplexity`: Perplexity vs. Number of iterations
  * `optimal_diversity`: Optimal level of diversity vs. Number of iterations
  * `perplexity_diversity`: Aggregated perplexity (T=10) vs. Diversity D

## Data-rotation experiment
This fork adds single-model conditions that test whether data rotation accounts for the benefit of partitioning across models. All new options default to the original behavior.

| `--arm` | Training data at t=0 | Prompts | Training data at t>0 |
|---|---|---|---|
| `baseline` | N real blocks | All N | All of the model's previous outputs |
| `fixed_subset` | N/M real blocks | The same N/M | All of the model's previous outputs |
| `rotated_own` | N/M real blocks | All N | A fresh N/M of the model's previous outputs |
| `rotated_real` | N/M real blocks | None | A fresh N/M real blocks from N |

M is `--equivalent_models` (default 8), and the N/M segment is model 0's segment in the D=M ecosystem.

Other options:
* `--partition_seed`: varies data segments and rotation draws (default 0, the original partition).
* `--num_beams`: sets generation beams (default 5). Use `--num_beams 1 --temperature 1` for plain sampling.
* `--log_checkpoint_metrics`: writes `checkpoint_metrics.json` in each version folder, with test perplexity for the final-epoch and best-checkpoint weights.

* `--skip_final_generate`: skips generation after the last iteration, whose output is never used.

### Running
`experiments/run_rotation.py` queues every arm and seed and runs them across GPUs:

```bash
python experiments/run_rotation.py --gpus 0 1 --jobs-per-gpu 2
```

Seeds run in order, so seed 0 finishes first and works as a pilot while later seeds continue. Each job logs to `MAIN_DIR/logs/` and marks its folder `.done` when it finishes. Rerunning the same command skips finished jobs. A folder from an unfinished job must be deleted first. Use `--dry-run` to print the commands. Other arguments, such as `--num_beams 1`, are passed to every job.

Summarize results at any point, including while jobs are running:

```bash
python experiments/summarize_rotation.py --plot rotation.png
python experiments/summarize_rotation.py --metric best
```

Check the data flow with `python tests/test_rotation_arms.py [UPSTREAM_CHECKOUT]` and the launcher with `python tests/test_launcher.py`. These use a tiny model and synthetic tokens, so it tests plumbing rather than results.
