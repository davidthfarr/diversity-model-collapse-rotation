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