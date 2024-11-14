# MLOPS Project 2 Containerization

This repository contains the code and a dockerized setup to train distilbert-base-uncased on MRPC for paraphrase detection and logg metrics with wandb.

## Requirements

- **Docker** must be installed on your system. [docker](https://www.docker.com)
- **Wandb** is used for logging training metrics. You need a wandb account and API key. [wandb](https://wandb.ai/home)

## Setup Instructions

1. **Clone repository**

   ```bash
   git clone <repository-url>
   cd mlops  
   ```

2. **Build Docker image**  

   ```bash
   docker build -t mlops-container .
   ```

3. **Run container**  
Run the container and passing in your wandb API key. This command maps the src and logs directories to the container, allowing the model code to be accessible and logs to be saved.

   ```bash
   docker run --rm -v $(pwd)/src:/code/src -v $(pwd)/logs:/code/logs -e WANDB_API_KEY=<YOUR_API_KEY> mlops-container
   ```

## Default Training Hyperparameters

The training is preconfigured with default hyperparameters.

```python
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
    'max_lr': 1e-4,
    'step_size_up': 100,
    'step_size_down': 100

}
```

Parameters can be customized by passing them as arguments to main.py. For example, to change the learning rate, run:

```bash
python main.py --learning_rate 3e-5
```

## Custom Output Directory

The path for the output directory where wandb-logs and checkpoints are stored can be changed with the --output_dir argument. The default path is `../logs`.

```bash
python main.py --output_dir '/path/to/custom/directory'
```

## Predefined Docker Setup

The Docker container is configured to run main.py with specific hyperparameters.

```dockerfile
CMD ["python", "main.py", "--use_cyclic_lr", "True", "--max_lr", "1e-3", "--step_size_up", "100_000", "--step_size_down", "100_000"]
```

This setup activates a cyclic learning rate with specified max_lr and step sizes. You can adjust this command in the Dockerfile to modify the configuration.

## Dynamic Run Naming

Run names are dynamically generated based on current date and hyperparameters that differ from the defaults in DEFAULT_HYPERPARAMS.

