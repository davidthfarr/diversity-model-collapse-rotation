import argparse
import copy
import os
import pickle
from itertools import chain

import lightning as L
import pandas as pd

import torch
from datasets import load_from_disk
from datasets import load_dataset as load_dataset_hf, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset, Subset
from transformers import AutoTokenizer


def arguments_defaults():
    arguments = {
        ('-tag', '--model_tag'): {
            'type': str,
            'default': None,
            'help': 'Model tag for the model to use. If not specified, the first model in model_tags will be used.',
        },

        ('-tags', '--model_tags'): {
            'nargs': '+',
            'type': str,
            'default': None,
            'help': 'Model tags for the models to use. Use it like this: --model_tags gpt2 gpt2-left. '
                    'This argument takes precedence over model_tag.'
                    'If not specified, the list will contain only model_tag.',
        },
        ('-nl', '--number_loops'): {
            'type': int,
            'default': 10,
            'help': 'Number of generation-training loops.',
        },

        ('-starting_loop', '--starting_loop'): {
            'type': int,
            'default': 0,
            'help': 'Starting loop number (version) to start with.',
        },

        # checkpoint
        ('-load', '--load_name'): {
            'type': str,
            'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('-save', '--save_name'): {
            'type': str,
            'default': None,
            'help': 'Name of the saved model to save.'
        },
        ('-main_dir', '--main_dir'): {
            'type': str,
            'default': 'versions',
            'help': 'Name of the saved model to save.'
        },
        ('-save_last', '--save_last'): {
            'action': argparse.BooleanOptionalAction,
            'default': False,
            'help': 'Whether to save last checkpoint of training.'
        },

        # debug control
        ('-dev', '--development'): {
            'action': argparse.BooleanOptionalAction,
            'default': False,
            'help': 'Use a small dataset. Use --no-development for full dataset.',
        },

        ('-small_dataset', '--small_dataset'): {
            'action': argparse.BooleanOptionalAction,
            'default': True,
            'help': 'Use a small dataset(20%) . Use --no-small_dataset for full dataset.',
        },
        ('-small_dataset_train_only', '--small_dataset_train_only'): {
            'action': argparse.BooleanOptionalAction,
            'default': True,
            'help': 'Use a small dataset (20%) on train only. Use --no-small_dataset_train_only for small test and val dataset.',
        },

        ('-slice_number', '--slice_number'): {
            'default': '0',
            'help': 'Which slice of the small dataset (20%) to use. 0 refers to the first one.'
                    'Use it like this: --slice_number 0. Or use --slice_number 0_1 to load two slices.'
        },
        ('-slice_numbers', '--slice_numbers'): {
            'nargs': '+',
            'default': ['0', ],
            'help': 'Which slices of the small dataset (20%) to use. 0 refers to the first one.'
                    'Use it like this: --slice_numbers 0 1. Or use --slice_numbers 0_1 2_3 to load two slices per model.'
                    'Or use --slice_numbers 0.0, 0.5, 1.0, 1.5 to split slices 0_1 into four subsets.'
        },

        ('-number_models', '--number_models'): {
            'type': int,
            'default': None,
            'help': 'Number of models to train in parallel. If None, it will be equal to the number of model_tags.'
        },

        ('-delete_models', '--delete_models'): {
            'action': argparse.BooleanOptionalAction,
            'default': False,
            'help': 'Whether to delete models after loop in case of using number_models argument. '
                    'Use --no-delete_models to keep all models.'
        },

        ('-dataset', '--dataset'): {
            'type': str,
            'default': 'wikitext2',
            'help': 'Pick a dataset.',
        },

        ('-dataset_fraction', '--dataset_fraction'): {
            'type': float,
            'default': None,
            'help': 'Fraction of dataset if small_dataset is set to True.',
        },

        ('-train_only', '--train_only'): {
            'action': argparse.BooleanOptionalAction,
            'default': False,
            'help': 'Training only without text generation.',
        },

        # common training args
        ('-opt', '--optimizer'): {
            'type': str,
            'default': 'adamw',
            'help': 'Pick an optimizer.',
        },

        ('-lr', '--learning-rate'): {
            'type': float,
            'default': 1e-5,
            'help': 'Initial learning rate.',
        },
        ('-m', '--max-epochs'): {
            'type': int,
            # 'default': 30,#100,
            'default': 5,  # 100,
            'help': 'Maximum number of epochs for training.',
        },
        ('-b', '--batch_size'): {
            'type': int,
            'default': 32,  # in Nature paper: 132
            'help': 'Batch size for training and evaluation.',
        },

        ('-blsz', '--block_size'): {
            'type': int,
            'default': 128,
            'help': 'Block size for training, generation, and evaluation.',
        },
        # text generation
        ('-repe', '--repetition_penalty'): {
            'type': int,
            'default': 1.0,
            'help': 'Repetition penalty in text generation. HF default is 1.0.',
        },

        ('-do_sample', '--do_sample'): {
            'action': argparse.BooleanOptionalAction,
            'default': False,
            'help': 'Sets do_sample in text generation to True. Uses default sampling parameters.'
                    '--temperature can override default temperature .',
        },
        ('-temperature', '--temperature'): {
            'type': float,
            'default': None,
            'help': 'Sets temperature parameter in text generation and sets do_sample to True.',
        },

        # debug control in training 'fast_dev_run'
        ('-d', '--debug'): {
            'action': 'store_true',
            'help': 'Verbose debug',
        },
        ('-seed', '--seed'): {
            'type': int,
            'default': 0,
            'help': 'Number of steps for model optimisation',
        },

        # cpu gpu setup for lightning
        ('-w', '--num_workers'): {
            # multiprocessing fail with too many works
            'type': int,
            'default': 1,
            'help': 'Number of CPU workers.',
        },
        ('-n', '--num_devices'): {
            'type': int,
            'default': 1,
            'help': 'Number of GPU devices.',
        },
        ('-cuda', '--cuda'): {
            'type': bool,
            'default': torch.cuda.is_available(),
            'help': 'Accelerator style.',
        },
        ('-a', '--accelerator'): {
            'type': str,
            'default': 'cuda',
            'help': 'Accelerator style.',
        },
        ('-s', '--strategy'): {
            'type': str,
            'default': 'ddp',
            # 'default': 'ddp_find_unused_parameters_false',
            'help': 'Strategy style.',
        },

        ('-p', '--pretrained'): {
            'action': 'store_true',
            'help': 'Load a pretrained network from Huggingface',
        },

        ('-version_name', '--version_name'): {
            'type': str,
            'default': None,
            'help': 'Version name.',
        },
        ('-rn', '--run_name'): {
            'type': str,
            'default': 'test',
            'help': 'Run name.',
        },

        # custom dataset generation and loading
        ('-gen', '--generate'): {
            'type': str,
            'default': None,
            'help': 'The file name to store the generated dataset.',
        },
        ('-load_gen', '--load_generate'): {
            'type': str,
            'default': None,
            'help': 'The file name to load the generated dataset.',
        },
        ('-gen_percent', '--generate_percentage'): {
            'type': float,
            'default': 0.0,
            'help': 'How much to mix from the real dataset',
        },
        ('-train_on_real', '--train_on_real'): {
            'action': argparse.BooleanOptionalAction,
            'default': True,
            'help': 'Train on real dataset in first loop. Use --no-train_on_real to skip training on real dataset.',
        },

        # data-rotation experiment
        ('-arm', '--arm'): {
            'type': str,
            'choices': ['baseline', 'fixed_subset', 'rotated_own', 'rotated_real'],
            'default': 'baseline',
            'help': 'Single-model data condition. baseline keeps the original behavior. See README.',
        },
        ('-equivalent_models', '--equivalent_models'): {
            'type': int,
            'default': 8,
            'help': 'Non-baseline arms train on N/equivalent_models blocks per iteration, matching D=M per-model size.',
        },
        ('-partition_seed', '--partition_seed'): {
            'type': int,
            'default': 0,
            'help': 'Seed for data segments and rotation draws. The default reproduces the original fixed partition.',
        },
        ('-num_beams', '--num_beams'): {
            'type': int,
            'default': 5,
            'help': 'Number of beams in generation. Use 1 with --temperature for plain sampling.',
        },
        ('-log_checkpoint_metrics', '--log_checkpoint_metrics'): {
            'action': argparse.BooleanOptionalAction,
            'default': False,
            'help': 'Write test perplexity for final-epoch and best-checkpoint weights each iteration.',
        },

    }
    return arguments


def evolution_args_preparation(args):
    if args.model_tags is None:
        if args.model_tag is None:
            raise ValueError('Either model_tag or model_tags must be specified.')
        args.model_tags = [args.model_tag, ]
    else:
        if args.model_tag is None:
            args.model_tag = args.model_tags[0]
        else:
            if args.model_tag != args.model_tags[0]:
                raise ValueError('model_tag is different from the first element of model_tags.')
    if args.arm != 'baseline':
        if args.number_models != 1:
            raise ValueError('Non-baseline arms require --number_models 1.')
        if args.equivalent_models < 2 or args.equivalent_models % 2 != 0:
            raise ValueError('--equivalent_models must be an even number of at least 2.')
        if args.generate_percentage > 0 or not args.small_dataset or not args.train_on_real:
            raise ValueError('Non-baseline arms require the primary setting: no real-data mix, '
                             'the small dataset, and real-data training at t=0.')
        if args.train_only and args.arm != 'rotated_real':
            raise ValueError('--train_only is only compatible with --arm rotated_real.')
        if args.arm == 'rotated_real':  # real data needs no generation step
            args.train_only = True


def load_args_defaults():
    arguments = arguments_defaults()
    p = argparse.ArgumentParser(description='Loop')
    for k, v in arguments.items():
        p.add_argument(*k, **v)
    a = p.parse_args()
    print('modelt', a.model_tag)
    evolution_args_preparation(a)
    return a


def load_base_model_name(args):
    path_dict = {
    }
    base_model_path = path_dict.get(args.model_tag, None)
    return base_model_path


def load_hf_model_name(model_tag):
    if 'gpt2-s' in model_tag:  # for gpt2-s1, ..., gpt2-sN
        base_model_name = 'gpt2'
    elif 'opt125-s' in model_tag:  # for opt125-s1, ..., opt125-sN
        base_model_name = 'facebook/opt-125m'
    elif 'gpt2m-s' in model_tag:  # for gpt2l-s1, ..., gpt2-sN
        base_model_name = 'gpt2-medium'
    elif 'opt350-s' in model_tag:  # for gpt2l-s1, ..., gpt2-sN
        base_model_name = 'facebook/opt-350m'
    else:
        raise ValueError('Unknown model tag')
    return base_model_name


def load_tokenizer(hf_model_name):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name,
                                              cache_dir='./model_cache_dir',
                                              return_dict=True)

    return tokenizer


def plt_model_load(model, checkpoint, args):
    if args.cuda:
        state_dict = torch.load(checkpoint)['state_dict']
    else:
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))['state_dict']

    model.load_state_dict(state_dict)
    return model


def load_model(load_path, plt_model, args):
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if os.path.isdir(load_path):
                checkpoint = os.path.join(load_path, "best.ckpt")
            else:
                raise ValueError(
                    f"if it is a directory, if must end with /; if it is a file, it must end with .ckpt: {load_path}"
                )
        plt_model = plt_model_load(plt_model, checkpoint, args)
        print(f"Loaded model from {checkpoint}")
    return plt_model


def paths_definitions(args):
    main_dir = args.main_dir
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    run_name = args.run_name  # f"{datetime.utcnow().strftime('%Y%m%d')}_"
    par_dir = os.path.join(main_dir, run_name)

    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    return par_dir


def load_version_file_paths(base_directory, save_suffix):
    # List to hold all dataframes
    file_paths = []
    versions = []
    # Loop through the subfolders (version_0, version_1, etc.)
    for subfolder in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, subfolder)

        # Only process subfolders that start with 'version_'
        if os.path.isdir(subfolder_path) and subfolder.startswith('version_'):
            print(f'Processing {subfolder}')
            # Loop through the CSV files in the subfolder
            version = subfolder
            name_ending = save_suffix.format(version=version)
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if file_name.endswith(name_ending):
                    versions.append(version)
                    file_paths.append(file_path)
    return versions, file_paths


def combine_metrics_files(base_directory):
    versions, file_paths = load_version_file_paths(base_directory, 'metrics.csv')
    if len(versions) == 0:
        return
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    for df, version in zip(dfs, versions):
        df['version'] = version
    # Concatenate all dataframes into one
    df = pd.concat(dfs, ignore_index=True)

    col_fix = ['version', 'epoch', 'step']
    df.sort_values(by=col_fix, inplace=True)
    new_order = col_fix + [i for i in df.columns if i not in col_fix]
    df = df.reindex(columns=new_order)
    save_combined_df(base_directory, df, 'metrics.csv')


def save_combined_df(base_directory, df, save_suffix):
    components = run_name_from_path(base_directory)
    prefix = '_'.join(components)

    if len(components) == 2:
        df['run_name'] = components[0]
        df['model_tag'] = components[1]

    # Write the final dataframe to a new CSV file
    fp = os.path.join(base_directory, f'{prefix}_{save_suffix}')
    df.to_csv(fp, index=False)
    print(f'CSV files have been combined into {fp}')


def run_name_from_path(base_directory):
    components = []
    head = base_directory
    for i in range(2):
        head, tail = os.path.split(head)
        if len(tail) > 0:
            components.append(tail)

    components.reverse()
    return components


### Datasets


def prepare_process_wikitext2(args):
    raw_dataset = prepare_data_wikitext2()
    dataset = preprocess_datasets(
        raw_dataset, args.tokenizer, block_size=args.block_size)
    return dataset


def combine_hf_datasets(datasets):
    train_ds = concatenate_datasets([ds['train'] for ds in datasets])
    val_ds = concatenate_datasets([ds['validation'] for ds in datasets])
    test_ds = concatenate_datasets([ds['test'] for ds in datasets])
    dataset = DatasetDict({
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds,
    })
    return dataset


def prepare_data_wikitext2(path='./data/wikitext2'):
    if (path is not None) and (not os.path.isdir(path)):
        print("Downloading and processing dataset...")
        dataset = load_dataset_hf('wikitext', 'wikitext-2-raw-v1')
        dataset.save_to_disk(path)
    else:
        print("Dataset already downloaded and processed")
        dataset = load_from_disk(path)
    return dataset


def preprocess_datasets(
        raw_dataset,
        tokenizer,
        block_size,
        overwrite_cache=False,
        preprocessing_num_workers=4):
    column_names = raw_dataset['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    print('block size: ', block_size)
    if block_size is None:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name],
                         add_special_tokens=False)  # gpt2 doesn't add but 'facebook/opt-125m' adds eos_token at every eos

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
        keep_in_memory=True,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
        keep_in_memory=True
    )
    return dataset


class MyDataLoader(L.LightningDataModule):

    def __init__(
            self, dataset_name, workers, train_dataset, val_dataset,
            test_dataset, batch_size):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.batch_size = batch_size
        self.num_workers = workers

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers)

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)

        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers)


class WikiText2Dataset(Dataset):
    path = './data/wikitext2'

    def __init__(
            self,
            dataset,
            partition,
            tokenizer=None,
            max_token_count=512):
        self.setup_tokenizer(tokenizer, max_token_count)
        self.dataset = dataset[partition]

    def setup_tokenizer(self, tokenizer, max_token_count):
        self.tokenizer = tokenizer
        self.max_token_count = max_token_count

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, index):
        data_row = self.dataset[index]
        return dict(
            input_ids=torch.tensor(data_row['input_ids']),
            attention_mask=torch.tensor(data_row['attention_mask']),
            labels=torch.tensor(data_row['labels']))


class GeneratedDataset(Dataset):
    path = './data/wikitext2'

    def __init__(
            self,
            my_file,
            partition=None,
            tokenizer=None,
            max_token_count=512):
        with open(my_file, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_row = self.dataset[index]
        return dict(
            input_ids=data_row['input_ids'],
            attention_mask=data_row['attention_mask'],
            labels=data_row['labels'])


def load_dataset(args):
    if args.dataset == 'wikitext2':
        dataset = prepare_process_wikitext2(args)
    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")
    tokenizer = args.tokenizer

    train_dataset = WikiText2Dataset(dataset=dataset, partition='train', tokenizer=tokenizer)

    val_dataset = WikiText2Dataset(dataset=dataset, partition='validation', tokenizer=tokenizer)
    test_dataset = WikiText2Dataset(dataset=dataset, partition='test', tokenizer=tokenizer)
    print('lengths original of train, val, test: ', len(train_dataset), len(val_dataset), len(test_dataset))
    if args.small_dataset and (args.load_generate is None or args.small_dataset_train_only is False):
        full_train_dataset = train_dataset
        train_dataset, new_val_dataset, new_test_dataset = load_small_dataset(train_dataset, val_dataset, test_dataset,
                                                                              args)
        train_dataset = select_real_data_for_arm(train_dataset, full_train_dataset, val_dataset, test_dataset, args)

        if args.small_dataset_train_only is False:
            val_dataset = new_val_dataset
            test_dataset = new_test_dataset
    if args.load_generate is not None:
        with open(args.load_generate, 'rb') as f:
            train_dataset_loaded = pickle.load(f)
        print(f"Loaded dataset from {args.load_generate}")
        assert str(train_dataset_loaded[0]['input_ids'].device) == str(
            train_dataset_loaded[-1]['input_ids'].device) == 'cpu'
        if args.generate_percentage > 0:
            print('Adding real dataset with percentage: ', args.generate_percentage)
            len_loaded_orig = len(train_dataset_loaded)
            len_real_orig = len(train_dataset)
            len_real = int(len_loaded_orig * args.generate_percentage)
            len_loaded = len_loaded_orig - len_real
            generator1 = torch.Generator().manual_seed(args.loop_number)
            train_dataset_loaded = \
            random_split(train_dataset_loaded, [len_loaded, len_loaded_orig - len_loaded], generator=generator1)[0]
            train_dataset = random_split(train_dataset, [len_real, len_real_orig - len_real], generator=generator1)[0]

            train_dataset = ConcatDataset([train_dataset, train_dataset_loaded])
            assert len(train_dataset) == len_loaded_orig
        elif args.arm == 'rotated_own':
            train_dataset = draw_rotating_subset(train_dataset_loaded, args)
        else:
            train_dataset = train_dataset_loaded
    print('lengths of train, val, test: ', len(train_dataset), len(val_dataset), len(test_dataset))
    return train_dataset, val_dataset, test_dataset


def draw_rotating_subset(pool, args):
    """Draws a fresh N/equivalent_models subset of the pool for the current iteration."""
    size = round(len(pool) / args.equivalent_models)
    generator = torch.Generator().manual_seed(args.partition_seed * 100003 + args.loop_number)
    indices = torch.randperm(len(pool), generator=generator)[:size].tolist()
    return Subset(pool, indices)


def select_real_data_for_arm(pool, full_train_dataset, val_dataset, test_dataset, args):
    """Selects the real blocks used for training or generation prompts in the current arm.

    fixed_subset:  the same segment at t=0 and for all prompts.
    rotated_own:   the segment at t=0, then prompts from the whole pool at every iteration.
    rotated_real:  the segment at t=0, then a fresh draw from the pool at every iteration.
    """
    if args.arm == 'baseline':
        return pool
    if args.arm == 'rotated_own' and args.data_role == 'generate':
        return pool
    if args.arm == 'rotated_real' and args.loop_number > 0:
        return draw_rotating_subset(pool, args)
    # Same blocks as model 0 of the D=equivalent_models ecosystem with this partition seed.
    segment_args = copy.copy(args)
    segment_args.number_models = args.equivalent_models
    segment_args.slice_number = '0'
    return load_small_dataset(full_train_dataset, val_dataset, test_dataset, segment_args)[0]


def load_small_dataset(train_dataset, val_dataset, test_dataset, args):
    if args.dataset_fraction is not None:
        fraction = args.dataset_fraction
    elif args.dataset == 'wikitext2':
        fraction = .2

    slices = [fraction] * int(1 / fraction)

    def load_slice(dataset_i, slice_number_i, slices_i=slices):
        generator1 = torch.Generator().manual_seed(args.partition_seed)
        new_ds = random_split(dataset_i, slices_i, generator=generator1)[slice_number_i]
        return new_ds

    def concat_slices(slice_numbers_i):
        print('using multiple slices: ', slice_numbers)
        new_train_ds_i = ConcatDataset([load_slice(train_dataset, i) for i in slice_numbers_i])
        new_val_ds_i = ConcatDataset([load_slice(val_dataset, i) for i in slice_numbers_i])
        new_test_ds_i = ConcatDataset([load_slice(test_dataset, i) for i in slice_numbers_i])
        return new_train_ds_i, new_val_ds_i, new_test_ds_i

    slice_number = args.slice_number
    print('using small dataset. Fraction: ', fraction)
    print('slice number(s): ', slice_number)
    if args.number_models and args.number_models == 1:
        assert slice_number == '0'
        slice_numbers = [0, 1]
        new_train_dataset, new_val_dataset, new_test_dataset = concat_slices(slice_numbers)
    elif args.number_models:
        slice_number = int(slice_number)
        first_slice = 0 if slice_number < args.number_models / 2 else 1
        print('first slice', first_slice, f'for slice number', slice_number, 'of', args.number_models)
        new_train_dataset = load_slice(train_dataset, first_slice)
        new_val_dataset = load_slice(val_dataset, first_slice)
        new_test_dataset = load_slice(test_dataset, first_slice)
        nr_fractions = args.number_models // 2
        if nr_fractions > 1:
            fraction_rest = 1 / nr_fractions
            slices_rest = [fraction_rest] * nr_fractions
            second_slice = slice_number if slice_number < nr_fractions else slice_number - nr_fractions  # e.g. slice_number 7 will is 3
            print('second slice', second_slice)
            new_train_dataset = load_slice(new_train_dataset, second_slice, slices_rest)
            new_val_dataset = load_slice(new_val_dataset, second_slice, slices_rest)
            new_test_dataset = load_slice(new_test_dataset, second_slice, slices_rest)
    return new_train_dataset, new_val_dataset, new_test_dataset


def merge_split_datasets_same_family(save_dirs, k_version):
    """Produces two random sets of equal size from two datasets of equal size
    containing 50% of each original dataset."""
    datasets = merge_split_datasets(save_dirs, k_version)
    return datasets


def merge_split_datasets(save_dirs, k_version):
    nr_splits = len(save_dirs)
    fraction = 1 / nr_splits
    slices = [fraction] * nr_splits  # [.5, .5] for 2
    new_datasets = {k: [] for k in range(nr_splits)}
    all_idx = list(range(nr_splits)) * 2  # [0, 1, 0, 1] for 2 datasets
    len_orig = []
    for i, fp in enumerate(save_dirs):
        dataset = GeneratedDataset(fp)
        len_orig.append(len(dataset))
        # ensure that the datasets sample from different data points in every loop
        generator1 = torch.Generator().manual_seed(k_version)
        datasets = random_split(dataset, slices, generator=generator1)
        # ensure equal length in case of uneven number of elements for both models

        idx = all_idx[i: i + nr_splits]  #
        subsets = [datasets[k] for k in idx]
        for k in range(nr_splits):
            new_datasets[k].append(subsets[k])

    # don't need to shuffle the datasets, as they will be shuffled in the dataloader
    merged_datasets = [ConcatDataset(new_datasets[k]) for k in range(nr_splits)]
    assert len(merged_datasets) == len(save_dirs)
    assert abs(len(merged_datasets[0]) - len(merged_datasets[1])) <= 1
    for i in range(nr_splits):
        assert abs(len_orig[i] - len(merged_datasets[i])) <= 1
    return merged_datasets
