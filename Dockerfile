FROM robonet-base:latest

COPY requirements.txt /tmp/requirements.txt
RUN ~/myenv/bin/pip install -r /tmp/requirements.txt
ENV PYTHONPATH=${PYTHONPATH}:/home/robonet/code/Bridge-Torch

# modify packages to work with python 3.8 (ros noetic needs python 3.8)
# to avoid orbax checkpoint error, downgrade flax 
# RUN ~/myenv/bin/pip install flax==0.6.11
# to avoid typing errors, upgrade distrax
# RUN ~/myenv/bin/pip install distrax==0.1.3

# avoid git safe directory errors
RUN git config --global --add safe.directory /home/robonet/code/Bridge-Torch

WORKDIR /home/robonet/code/Bridge-Torch