FROM continuumio/miniconda3:latest AS base

WORKDIR /code

COPY environment.yml /code/

RUN conda env create -f environment.yml

ENV PATH /opt/miniconda3/envs/mlops/bin:$PATH

SHELL ["conda", "run", "-n", "mlops", "/bin/bash", "-c"]

RUN conda list

WORKDIR /code/src

CMD ["conda", "run", "-n", "mlops", "python", "/code/src/main.py", "--use_cyclic_lr", "True", "--max_lr", "1e-3", "--step_size_up", "100_000", "--step_size_down", "100_000"]