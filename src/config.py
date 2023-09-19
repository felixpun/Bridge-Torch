import argparse

DATASETS = [
    "/data/hyx/raw_icra_trajs",
    "/data/hyx/raw_flap_trajs",
    "/data/hyx/raw_bridge_data_v1_trajs",
    "/data/hyx/raw_rss_trajs",
    "/data/hyx/raw_bridge_data_v2_trajs",
    "/home/hyx/bridge_data_v2/bridge_torch/data_processing/scipted_trajs"
]

ACT_MEAN = [
    1.9296819e-04,
    1.3667766e-04,
    -1.4583133e-04,
    -1.8390431e-04,
    -3.0808983e-04,
    2.7425270e-04,
    5.9716219e-01,
]

ACT_STD = [
    0.00912848,
    0.0127196,
    0.01229497,
    0.02606696,
    0.02875283,
    0.07807977,
    0.48710242,
]

GOAL_RELABELING_KWARGS = dict(reached_proportion=0.0)

AUGMENT_KWARGS = dict(
    random_resized_crop=dict(
        size=[128, 128],
        scale=[0.8, 1.0],
        ratio=[0.9, 1.1],
        antialias=True
    ),
    color_jitter=dict(
        brightness=0.2,
        contrast=[0.8, 1.2],
        saturation=[0.8, 1.2],
        hue=0.1,
    ),
    augment_order=[
        "random_resized_crop",
        "color_jitter",
    ],
)

ENCODER_KWARGS = dict(
    pooling_method="avg",
    add_spatial_coordinates=True,
    act="SiLU",
    input_img_shape=[128, 128],
    input_channels=6
)

def get_args():
    def str2bool(v):
        return v.lower() in ('true')

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    
    data_arg = parser.add_argument_group('Data')
    data_arg.add_argument('--sample_weights', type=str, default='balance')
    data_arg.add_argument('--num_workers', type=int, default=8)
    data_arg.add_argument('--relabel_actions', type=str2bool, default=True)
    data_arg.add_argument('--goal_relabeling_strategy', type=str, default='uniform')
    data_arg.add_argument('--augment', type=str2bool, default=True)

    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--dtype', type=str, default='fp32')
    model_arg.add_argument('--encoder', type=str, default='resnetv1-34-bridge')

    learn_arg = parser.add_argument_group('Learning')
    learn_arg.add_argument('--save_dir', type=str, default='./save')
    learn_arg.add_argument('--ckpt_id', type=str, default=None)
    learn_arg.add_argument('--train_batch_size', type=int, default=256)
    learn_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learn_arg.add_argument('--eval_batch_size', type=int, default=256)
    learn_arg.add_argument('--max_lr', type=float, default=3e-4)
    learn_arg.add_argument('--min_lr', type=float, default=1e-5)
    learn_arg.add_argument('--weight_decay', type=float, default=1e-6)
    learn_arg.add_argument('--max_grad_norm', type=float, default=5.0)
    learn_arg.add_argument('--epochs', type=int, default=None)
    learn_arg.add_argument('--steps', type=int, default=500000)
    learn_arg.add_argument('--warmup_steps', type=int, default=10000)
    learn_arg.add_argument('--decay_steps', type=int, default=None)
    learn_arg.add_argument('--log_interval', type=int, default=5000)
    learn_arg.add_argument('--eval_interval', type=int, default=10000)
    learn_arg.add_argument('--save_interval', type=int, default=10000)
    learn_arg.add_argument('--save_best', type=str2bool, default=True)
    learn_arg.add_argument('--main_metric', type=str, default='log_probs')
    
    misc_arg = parser.add_argument_group('MISC')
    misc_arg.add_argument('--method', type=str, default='gc_bc')
    misc_arg.add_argument('--random_seed', type=int, default=42)

    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))

    args.datasets = DATASETS
    args.act_mean = ACT_MEAN
    args.act_std = ACT_STD
    args.goal_relabeling_kwargs = GOAL_RELABELING_KWARGS
    args.augment_kwargs = AUGMENT_KWARGS
    args.encoder_kwargs = ENCODER_KWARGS

    return args


def get_ds_config(args):
    ds_config = {
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": args.log_interval,
        "optimizer": {
            "type": "Adam",
            "params": {
                "weight_decay": args.weight_decay,
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.steps if args.steps else args.warmup_steps + args.decay_steps,
                "warmup_min_lr": args.min_lr,
                "warmup_max_lr": args.max_lr,
                "warmup_num_steps": args.warmup_steps,
            }
        },
        "gradient_clipping": args.max_grad_norm,
        "bf16": {
            "enabled": args.dtype == "bf16"
        },
        "fp16": {
            "enabled": args.dtype == "fp16",
            "fp16_master_weights_and_grads": False,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 15
        },
        # "zero_optimization": {
        #     "stage": 2,
        #     "contiguous_gradients": True,
        #     "overlap_comm": True,
        #     "reduce_scatter": True,
        # }
    }

    return ds_config