import argparse
import os
import torch
import wandb
import pytorch_lightning as L

from datetime import datetime
from glue_transfomer import GLUETransformer
from data_module import GLUEDataModule
from wandb_utils import setup_wandb

DEFAULT_HYPERPARAMS = {
    'model_name': 'distilbert-base-uncased',
    'task_name': 'mrpc',
    'project_name': 'mlops',
    'epochs': 3,
    'learning_rate': 1e-5,
    'warmup_steps': 0,
    'weight_decay': 0.0,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'use_cyclic_lr': False,
    'base_lr': 1e-5,
    'max_lr': 1e-1,
    'step_size_up': 100000,
    'step_size_down': 100000
}
def parse_args():
    """
    Parse command-line arguments for hyperparameters and paths, using defaults from DEFAULT_HYPERPARAMS.
    """
    parser = argparse.ArgumentParser(description="Training script for GLUE tasks")
    
    for key, value in DEFAULT_HYPERPARAMS.items():
        arg_type = type(value)
        parser.add_argument(f'--{key}', type=arg_type, default=value, help=f'{key.replace("_", " ").capitalize()} (default: {value})')

    parser.add_argument('--output_dir', type=str, default='../logs', help='Path to directory for logs and checkpoints')
    args = parser.parse_args()
    return args

def filter_hyperparams(args):
    """
    Compare parsed hyperparameters with the default ones.
    Return only those that are considered hyperparameters and differ from the default values.
    """
    excluded_keys = {'output_dir', 'model_name', 'task_name', 'project_name'}
    hyperparams = {}

    for key, value in vars(args).items():
        if key in excluded_keys:
            continue
        if key not in DEFAULT_HYPERPARAMS or value != DEFAULT_HYPERPARAMS[key]:
            hyperparams[key] = value

    return hyperparams


def main():
    args = parse_args()

    wandb.login()

    filtered_hyperparams = filter_hyperparams(args)
    full_hyperparams = vars(args)

    model_name = args.model_name
    task_name = args.task_name
    project_name = args.project_name
    epochs = args.epochs
    output_dir = args.output_dir

    wandb_logger, checkpoint_callback, log_dir, checkpoint_dir = setup_wandb(full_hyperparams, 
                                                                            filtered_hyperparams, 
                                                                            model_name, task_name, 
                                                                            project_name, 
                                                                            output_dir)

    L.seed_everything(42)

    dm = GLUEDataModule(
        model_name_or_path=model_name,
        task_name=task_name,
    )
    dm.setup("fit")

    model = GLUETransformer(
        model_name_or_path=model_name,
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        **filtered_hyperparams
    )

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=dm)

    wandb.finish()

if __name__ == '__main__':
    main()
