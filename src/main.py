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
    Parse command-line arguments for hyperparameters and paths.
    """
    parser = argparse.ArgumentParser(description="Training script for GLUE tasks")
    
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--use_cyclic_lr', type=bool, default=False, help='Whether to use cyclic learning rate')
    parser.add_argument('--base_lr', type=float, default=1e-5, help='Base learning rate for cyclic lr')
    parser.add_argument('--max_lr', type=float, default=1e-1, help='Maximum learning rate for cyclic lr')
    parser.add_argument('--step_size_up', type=int, default=100000, help='Step size up for cyclic lr')
    parser.add_argument('--step_size_down', type=int, default=100000, help='Step size down for cyclic lr')

    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased', help='Pretrained model name')
    parser.add_argument('--task_name', type=str, default='mrpc', help='Task name')
    parser.add_argument('--project_name', type=str, default='mlops', help='WandB project name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    
    parser.add_argument('--output_dir', type=str, default='../logs', help='Path to directory for logs and checkpoints')
    
    args = parser.parse_args()
    
    return args

def filter_hyperparams(args):
    """
    Compare parsed hyperparameters with the default ones.
    Return only those that are hyperparameters and differ from the default values.
    """
    hyperparams = {}
    for key, value in vars(args).items():
        if (key == 'output_dir') | (key == 'model_name') | (key == 'task_name')| (key == 'project_name'):  
            continue
        if key in DEFAULT_HYPERPARAMS and value != DEFAULT_HYPERPARAMS[key]:
            hyperparams[key] = value
        elif key not in DEFAULT_HYPERPARAMS:  
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
