import json
from datetime import datetime

import torch
import lightning as L
import pickle
from tqdm import tqdm
import os
import gc

from transformers import AutoModelForCausalLM
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from lning_defs import GenTrainModule
from helpers import load_model, load_args_defaults, paths_definitions, load_hf_model_name, load_base_model_name, \
    combine_metrics_files, MyDataLoader, load_dataset, merge_split_datasets_same_family, \
    load_tokenizer


#  Based on the code from https://zenodo.org/records/10866595
def evaluate_test_perplexity(plt_model, test_dataloader, device):
    """Matches GenTrainModule's test_mean_perp: exp of the unweighted mean batch loss."""
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    plt_model.to(device).eval()
    losses = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype):
        for batch in test_dataloader:
            loss, _ = plt_model(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                                batch['labels'].to(device))
            losses.append(loss.float())
    return torch.exp(torch.stack(losses).mean()).item()


def write_checkpoint_metrics(args, plt_model, trainer, checkpoint_callback, data_loader):
    """Records both model states in an iteration without changing the run itself.

    Test reporting and generation use the final-epoch weights, while the next iteration starts from the best
    validation checkpoint. Evaluating both leaves the weights and random state as they were.
    """
    device = torch.device('cuda' if args.cuda else 'cpu')
    final_state = {k: v.detach().cpu().clone() for k, v in plt_model.state_dict().items()}
    with torch.random.fork_rng():
        final_test_perp = evaluate_test_perplexity(plt_model, data_loader.test_dataloader, device)
        checkpoint = torch.load(checkpoint_callback.best_model_path, map_location='cpu', weights_only=False)
        plt_model.load_state_dict(checkpoint['state_dict'])
        best_test_perp = evaluate_test_perplexity(plt_model, data_loader.test_dataloader, device)
    plt_model.load_state_dict(final_state)
    metrics = {
        'arm': args.arm,
        'iteration': args.loop_number,
        'train_blocks': len(data_loader.train_dataset),
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint_callback.best_model_score.item(),
        'reported_test_perp': trainer.callback_metrics['test_mean_perp'].item(),
        'final_epoch_test_perp': final_test_perp,
        'best_checkpoint_test_perp': best_test_perp,
    }
    with open(os.path.join(args.save_dir, args.version_name, 'checkpoint_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)


def loop(args):
    args.data_role = 'train'
    hf_model_name = load_hf_model_name(args.model_tag)
    args.base_model_name = load_base_model_name(args)
    # when loaded via config file, the loss is much higher
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    print(f"Loaded model from HuggingFace")
    tokenizer = load_tokenizer(hf_model_name)

    args.tokenizer = tokenizer

    # load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(args)

    data_loader = MyDataLoader(
        'wikitext2',
        args.num_workers,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size)

    # print(f"Loaded dataset from HuggingFace")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best",
        dirpath=args.save_name,
        save_last=args.save_last,
    )

    # TensorBoardLogger is the default logger in Lightning
    # os.path.join(save_dir, name, version).
    # save_dir includes model_tag
    if args.version_name is not None:
        logger = CSVLogger(args.save_dir, name='', version=args.version_name)
    else:
        logger = CSVLogger(args.save_dir, name='')
    arg_dict = {k: v for k, v in args.__dict__.items() if k not in ['model', 'tokenizer']}
    with open(os.path.join(args.save_dir, 'commandline_args.json'), 'w') as f:
        json.dump(arg_dict, f, indent=2)
    plt_model = GenTrainModule(
        model,
        learning_rate=args.learning_rate,
        epochs=args.max_epochs
    )
    if args.load_name is None:
        plt_model = load_model(plt_model=plt_model, load_path=args.base_model_name, args=args)
    else:
        plt_model = load_model(plt_model=plt_model, load_path=args.load_name, args=args)
    # testm = MetricTracker()#MetricTracker(MetricCollection([CatMetric()]))

    trainer = L.Trainer(
        precision=16,
        max_epochs=args.max_epochs,
        devices=args.num_devices,
        accelerator=args.accelerator if args.cuda else 'cpu',
        strategy=args.strategy,
        fast_dev_run=args.debug,
        callbacks=[checkpoint_callback],  # , testm],
        logger=logger,
    )

    if args.cuda:
        plt_model = plt_model.cuda()
    if args.test_before_train:  # test untrained model
        trainer.test(plt_model, dataloaders=data_loader.test_dataloader)

    if args.train_model:
        trainer.fit(
            plt_model,
            train_dataloaders=data_loader.train_dataloader,
            val_dataloaders=data_loader.val_dataloader)

        trainer.test(plt_model, dataloaders=data_loader.test_dataloader)  # same as in log file
        if args.log_checkpoint_metrics:
            write_checkpoint_metrics(args, plt_model, trainer, checkpoint_callback, data_loader)

    if args.train_only is False and args.generate is not None:
        # load dataloader with original dataset in case loaded from generated for training
        if args.load_generate is not None or args.arm == 'rotated_own':
            args.load_generate = None
            args.data_role = 'generate'
            # load real dataset
            train_dataset, val_dataset, test_dataset = load_dataset(args)

            data_loader = MyDataLoader(
                'wikitext2',
                args.num_workers,
                train_dataset,
                val_dataset,
                test_dataset,
                batch_size=args.batch_size)

        with torch.no_grad():
            i = 0
            model = plt_model.model.eval()
            batches = []
            for batch in tqdm(data_loader.train_dataloader):
                if args.cuda:
                    input_ids = batch["input_ids"].cuda()
                    attention_mask = batch["attention_mask"].cuda()
                    labels = batch["labels"].cuda()
                    plt_model = plt_model.cuda()
                    model = model.cuda()
                else:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                    plt_model = plt_model
                    model = model

                batch_size, _ = input_ids.shape

                if i == 0:
                    print('generating batch 1 of size', batch_size)
                if args.do_sample is True or args.temperature:
                    do_sample = True
                    outputs = model.generate(input_ids, num_beams=args.num_beams, max_new_tokens=args.block_size,
                                             min_new_tokens=args.block_size,
                                             repetition_penalty=args.repetition_penalty,
                                             pad_token_id=tokenizer.eos_token_id,
                                             attention_mask=attention_mask,
                                             do_sample=do_sample, temperature=args.temperature)
                else:
                    outputs = model.generate(input_ids, num_beams=args.num_beams, max_new_tokens=args.block_size,
                                             min_new_tokens=args.block_size,
                                             repetition_penalty=args.repetition_penalty,
                                             pad_token_id=tokenizer.eos_token_id,
                                             attention_mask=attention_mask, )
                preds = outputs[:, args.block_size:]

                batch["input_ids"] = preds.cpu().detach()
                batch["labels"] = preds.cpu().detach()
                batch["attention_mask"] = attention_mask.cpu().detach()

                my_batch = []
                for j in range(batch_size):
                    my_batch.append(
                        {"input_ids": batch["input_ids"][j, :], "attention_mask": batch["attention_mask"][j, :],
                         "labels": batch["labels"][j, :]})
                batches += my_batch
                i += 1
            file_name = args.generate
            with open(file_name, "wb") as f:
                pickle.dump(batches, f)
            print(f"{file_name} generated and saved!")


def loop_wrapper(args):
    # define paths
    par_dir = paths_definitions(args)

    # sets seeds for numpy, torch and python.random.
    L.seed_everything(args.seed, workers=True)
    if args.number_models:
        args.model_tags = [f'{args.model_tag}-s{i}' for i in range(args.number_models)]
        args.slice_numbers = [str(i) for i in range(args.number_models)]

    print('model_tags', args.model_tags, 'slice_numbers', args.slice_numbers)
    model_tags = args.model_tags

    save_ds_name = 'dataset'
    if len(model_tags) > 1:
        load_ds_name = f'load_{save_ds_name}'
    else:
        load_ds_name = save_ds_name
    if len(model_tags) > len(args.slice_numbers):
        args.slice_numbers = [args.slice_number] * len(model_tags)
        print('Using slice_number', args.slice_number, 'for all models:', model_tags)
    for k in range(args.starting_loop, args.number_loops):
        print('version', k)
        vrs_load = f"version_{k - 1}"
        vrs_save = f"version_{k}"

        args.version_name = vrs_save
        args.loop_number = k
        ds_save_dirs = []
        for model_tag, slice_number in zip(model_tags, args.slice_numbers):
            print('model_tag', model_tag, 'slice_number', slice_number, )
            save_dir = os.path.join(par_dir, model_tag)
            args.save_dir = save_dir
            args.model_tag = model_tag
            args.slice_number = slice_number
            vrs_dir = os.path.join(save_dir, vrs_save)
            if not os.path.exists(vrs_dir):
                os.makedirs(vrs_dir)
            args.train_model = True
            args.test_before_train = False
            if k == 0 and args.train_on_real:  # load base model and train on real and generate
                args.load_name = None  # load  base model
                args.test_before_train = True  # test untrained model
            elif k == 0 and args.train_on_real is False:  # load base model and generate
                args.load_name = None
                args.test_before_train = True  # test untrained model
                args.train_model = False
            elif k == 1 and args.train_on_real is False:  # load base model and train on generated and generate
                args.load_name = None  # load  base model
                args.load_generate = os.path.join(save_dir, vrs_load, f'{load_ds_name}_{vrs_load}.pkl')
            else:  # load trained model, train on generated dataset and generate
                args.load_name = os.path.join(save_dir, vrs_load, 'checkpoints', 'best')  # path to load model
                args.load_generate = os.path.join(save_dir, vrs_load, f'{load_ds_name}_{vrs_load}.pkl')

            if args.train_only:
                args.load_generate = None
            args.generate = os.path.join(save_dir, vrs_save,
                                         f'{save_ds_name}_{vrs_save}.pkl')  # path to save generated dataset
            ds_save_dirs += [args.generate]
            # set name of the saved model to save
            args.save_name = os.path.join(save_dir, vrs_save, 'checkpoints', 'best')
            loop(args)

            if args.number_models:
                gc.collect()  # free memory
                torch.cuda.empty_cache()  # free memory

        if len(model_tags) > 1 and args.train_only is False:
            # merge and split datasets
            datasets = merge_split_datasets_same_family(ds_save_dirs, k + args.partition_seed * 100003)

            # save datasets to load from
            for model_tag, ds in zip(model_tags, datasets):
                file_name = os.path.join(par_dir, model_tag, vrs_save, f'{load_ds_name}_{vrs_save}.pkl')
                with open(file_name, "wb") as f:
                    pickle.dump(ds, f)
        if args.delete_models is True and k > 0:
            print(f'Deleting model from {vrs_load}')
            for model_tag in model_tags[1:]:  # keep the first model for further analysis
                path_dir = os.path.join(par_dir, model_tag, vrs_load, 'checkpoints')
                os.system(f'rm -r {path_dir}')
                file_path = os.path.join(par_dir, model_tag, vrs_load, f'load_dataset_{vrs_load}.pkl')
                os.remove(file_path)

        gc.collect()  # free memory
        torch.cuda.empty_cache()  # free memory

    for model_tag in model_tags:
        save_dir = os.path.join(par_dir, model_tag)
        combine_metrics_files(save_dir)


def main():
    args = load_args_defaults()
    loop_wrapper(args)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
