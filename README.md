# Pytorch BC/RL Implementations for BridgeData V2

This repository provides distributed training code (based on Deepspeed) for [BridgeData V2](https://rail-berkeley.github.io/bridgedata/). Since most open-source LLMs are implemented with Pytorch and require memory-efficient distributed training frameworks to fine-tune on, we believe this repository can facilitate the application of Multimodal LLMs on this field.

We provide implementations for the following methods:

- Goal-conditioned BC
- Goal-condtioned IQL (TODO)

A novel generative VLM-based BC method is under development by us. Stay tuned.

## Environment

The dependencies for this codebase can be installed in a conda environment:

```
pip install -r requirements.txt
```

## Data

The raw dataset (comprised of JPEGs, PNGs, and pkl files) can be downloaded from the [website](https://rail-berkeley.github.io/bridgedata/). For training, first use `data_processing/bridgedata_raw_to_numpy.py` to convert the raw data into numpy files; then use `data_processing/bridgedata_numpy_split.py` to convert the numpy files into trajectory-level pickle files.

## Training

To start training run the command below.

```
deepspeed src/train.py \
    --steps 300000 \
    --warmup_steps 10000 \
    --save_dir gc_bc_save \
    --run_name gd_bc \
    --random_seed 42
```

Hyperparameters can be modified in `src/config.py`.
The training log is expected to look like `train.log`.

## Evaluation

First, set up the robot hardware according to the [official guide](https://docs.google.com/document/d/1si-6cTElTWTgflwcZRPfgHU7-UwfCUkEztkH3ge5CGc/edit?usp=sharing). Install the WidowX robot controller stack from the [official repo](https://github.com/rail-berkeley/bridge_data_robot). Then, run the command:

```
python src/eval.py \
    --num_timesteps NUM_TIMESTEPS \
    --video_save_path VIDEO_DIR \
    --checkpoint_path gc_bc_save/190000/mp_rank_00_model_states.pt \
    --config_path  gc_bc_save/config.json \
    --blocking
```

The script loads cofiguration of the checkpoint from a json file in save_dir (e.g., gc_bc_save) output by train.py.

We will be grateful if you can report your reproduced results on an issue.

## Provided Checkpoints

Checkpoints for GCBC are available [here](https://drive.google.com/drive/folders/11d6OPfqE51YHa28Rgwt26u849IaayqAs?usp=sharing).

## Acknowledgement

This code mimics the [official JAX implementation](https://github.com/rail-berkeley/bridge_data_v2), while being more compact and easier to follow.
