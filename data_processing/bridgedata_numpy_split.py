"""
Split data from the BridgeData numpy into trajectory per file.

Consider the following directory structure for the input data:

    bridgedata_numpy/
        rss/
            toykitchen2/
                set_table/
                    00/
                        train/
                            out.npy
                        val/
                            out.npy
        icra/
            ...

The --depth parameter controls how much of the data to process at the
--input_path; for example, if --depth=5, then --input_path should be
"bridgedata_numpy", and all data will be processed. If --depth=3, then
--input_path should be "bridgedata_numpy/rss/toykitchen2", and only data
under "toykitchen2" will be processed.

Take the second case as an example, the output will be written as
"{output_path}/train/set_table.00.{index}.npy", where index is the index
of corresponding trajectory

Written by Kevin Black (kvablack@berkeley.edu).
Adapted by Yuxin He (he.yuxin@qq.com).
"""
import os
import pickle
from multiprocessing import Pool

import numpy as np
import glob
import tqdm
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse. Looks for {input_path}/dir_1/dir_2/.../dir_{depth-1}/train/out.npy",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_integer("num_workers", 4, "Number of threads to use")


def process(path):

    if not os.path.exists(path):
        logging.info(f"Skipping {path}, does not exist")
        return
    
    with open(path, "rb") as f:
        arr = np.load(f, allow_pickle=True)

    if len(arr) == 0:
        logging.info(f"Skipping {path}, empty")
        return

    dirname = os.path.dirname(os.path.abspath(path))
    train_or_val = dirname.split(os.sep)[-1]
    # {output_path}/train or {output_path}/val
    outpath = os.path.join(FLAGS.output_path, train_or_val)
    os.makedirs(outpath, exist_ok=True)

    out_fn_prefix = '.'.join(dirname.split(os.sep)[-FLAGS.depth : -1])
    # outpath = f"{outpath}/out.tfrecord"

    for index, traj in enumerate(arr):
        truncates = np.zeros(len(traj["actions"]), dtype=np.bool_)
        truncates[-1] = True
        example = {
            "observations/images0": np.array(
                [o["images0"] for o in traj["observations"]],
                dtype=np.uint8,
            ),
            "observations/state": np.array(
                [o["state"] for o in traj["observations"]],
                dtype=np.float32,
            ),
            "next_observations/images0": np.array(
                [o["images0"] for o in traj["next_observations"]],
                dtype=np.uint8,
            ),
            "next_observations/state": np.array(
                [o["state"] for o in traj["next_observations"]],
                dtype=np.float32,
            ),
            "language": traj["language"],
            "actions": np.array(traj["actions"], dtype=np.float32),
            "terminals": np.zeros(len(traj["actions"]), dtype=np.bool_),
            "truncates": truncates,
        }
        out_fn = out_fn_prefix + f'.{index}'
        out_fn = os.path.join(outpath, out_fn)
        with open(out_fn, 'wb') as f:
            pickle.dump(example, f)


def main(_):
    assert FLAGS.depth >= 1

    paths = glob.glob(
        os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1)))
    )
    paths = [f"{p}/train/out.npy" for p in paths] + [f"{p}/val/out.npy" for p in paths]

    with Pool(FLAGS.num_workers) as p:
        list(tqdm.tqdm(p.imap(process, paths), total=len(paths)))


if __name__ == "__main__":
    app.run(main)
