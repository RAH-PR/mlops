FROM python:3.12

WORKDIR /code

COPY requirements.txt /code/

RUN pip install --upgrade pip 
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /code/requirements.txt 

WORKDIR /code/src

CMD ["python", "main.py", "--use_cyclic_lr", "True", "--max_lr", "1e-3", "--step_size_up", "100_000", "--step_size_down", "100_000"]