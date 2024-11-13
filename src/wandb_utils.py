import os
import wandb
from datetime import datetime
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def create_directories(path):
    """
    Create directories for logging and checkpoints if they do not exist.
    """
    os.makedirs(path, exist_ok=True)
    return path

def construct_experiment_name(hyperparams, full_hyperparams):
    """
    Constructs a unique experiment name based on hyperparameters and current timestamp.
    """
    hyperparam_names = "-".join(hyperparams.keys())
    hyperparam_values = "-".join([str(value) for value in hyperparams.values()])
    hyperparam_string = f"{hyperparam_names}_{hyperparam_values}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{full_hyperparams['model_name']}_{full_hyperparams['task_name']}_{hyperparam_string}"

def setup_wandb(full_hyperparams, filtered_hyperparams, model_name, task_name, project_name, base_dir='mlops'):
    """
    Set up WandB logging and checkpointing for an experiment with the given hyperparameters.
    """
    
    experiment_name = construct_experiment_name(filtered_hyperparams, full_hyperparams)
    wandb_experiment_name = experiment_name.replace("/", "-")
    #wandb_hyperparam_string = "/".join([str(value) for value in filtered_hyperparams.values()]).replace("/", "-")
    
    folder_structure = os.path.join('../logs/', experiment_name)
    
    log_dir = create_directories(os.path.join(folder_structure, 'wandb_logs'))
    checkpoint_dir = create_directories(os.path.join(folder_structure, 'checkpoints'))

    wandb.init(
        project=project_name,
        name=wandb_experiment_name,
        config={  
            **full_hyperparams,  
            "model_name": model_name,
            "task_name": task_name,
        },
        tags=[task_name, model_name], 
        dir=log_dir,  
        id=wandb_experiment_name,  
    )

    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        log_model=True,  
        save_dir=log_dir, 
        id=wandb_experiment_name
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  
        dirpath=checkpoint_dir, 
        filename=f"{wandb_experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}",  
        save_top_k=1,  
        mode='min',  
    )

    return wandb_logger, checkpoint_callback, log_dir, checkpoint_dir
