import os
import math
import pickle
import torch
import numpy as np
from torch.utils.data import IterableDataset
from .goal_relabeling import GOAL_RELABELING_FUNCTIONS
from .augmentations import get_augment_transform

def repeat_by_weights(list_data_filenames, sample_weights):
    if sample_weights == None:
        all_data_filenames = sum(list_data_filenames, [])

    elif sample_weights == 'balance':
        max_samples_per_set = max([len(x) for x in list_data_filenames])
        all_data_filenames = []
        for data_filenames in list_data_filenames:
            repeat_times = max_samples_per_set // len(data_filenames)
            all_data_filenames.extend(data_filenames * repeat_times)

    else:
        sample_weights = eval(sample_weights)
        assert isinstance(sample_weights, list) and isinstance(sample_weights[0], float)
        ori_total_num = sum([len(x) for x in list_data_filenames])
        all_data_filenames = []
        for target_proportion, data_filenames in zip(sample_weights, list_data_filenames):
            proportion = len(data_filenames) / ori_total_num
            repeat_times = max(int(target_proportion // proportion), 1)
            all_data_filenames.extend(data_filenames * repeat_times)

    return all_data_filenames

def binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    # in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    in_between_mask = ~(open_mask | closed_mask)

    # is_open_float = tf.cast(open_mask, tf.float32)
    is_open_float = open_mask.astype(np.float32)

    # def scan_fn(carry, i):
    #     return tf.cond(
    #         in_between_mask[i],
    #         lambda: tf.cast(carry, tf.float32),
    #         lambda: is_open_float[i],
    #     )

    # new_actions = tf.scan(
    #     scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    # )
    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = is_open_float[i]
        new_actions[i] = carry
    return new_actions


class BridgeDatasetTorch(IterableDataset):
    def __init__(self, args, local_rank, world_size, is_train=True):
        super().__init__()
        self.args = args
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_train = is_train

        self.relabel_actions = args.relabel_actions
        self.action_metadata = {
            "mean": args.act_mean,
            "std": args.act_std
        }

        self.goal_relabeling_strategy = args.goal_relabeling_strategy
        self.goal_relabeling_kwargs = args.goal_relabeling_kwargs

        self.augment = args.augment
        self.augment_kwargs = args.augment_kwargs
        if is_train and self.augment and self.augment_kwargs:
            self.augment_transform = get_augment_transform(args.augment_kwargs)

        dataset_paths = args.datasets
        train_or_eval = 'train' if is_train else 'val'
        dataset_paths = [os.path.join(x, train_or_eval) for x in dataset_paths]
        
        list_data_filenames = []
        for dataset_path in dataset_paths:
            data_filenames = []
            for fn in sorted(os.listdir(dataset_path), key=lambda x: ('.'.join(x.split('.')[:-1]), int(x.split('.')[-1]))):
                data_filenames.append(os.path.join(dataset_path, fn))
            list_data_filenames.append(data_filenames)

        all_data_filenames = repeat_by_weights(list_data_filenames, args.sample_weights)

        if local_rank <= 0:
            for dataset_path, data_filenames in zip(dataset_paths, list_data_filenames):
                print(f"{len(data_filenames)} trajs in {dataset_path}")
                
            print(f"[# {train_or_eval} trajs before repeating]: {sum([len(x) for x in list_data_filenames])}")
            print(f"[# {train_or_eval} trajs after repeating]: {len(all_data_filenames)}")
            print()

            meta_info_fn = os.path.join(args.save_dir, '_'.join([train_or_eval, 'set', 'meta_info']))
            with open(meta_info_fn, 'w') as f:
                for data_filename in all_data_filenames:
                    print(data_filename, file=f)

        self.all_data_filenames = all_data_filenames
        self.all_data_filenames_shuffled = None

        self.start, self.end = 0, len(all_data_filenames) // self.world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker
            iter_start = self.start
            iter_end = self.end
        else:  # multiple workers
            # 保证dataloader的不同线程读取到的是不同数据
            per_worker = math.ceil((self.end - self.start) / worker_info.num_workers)
            assert isinstance(per_worker, int)
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        sample_iterator = self._sample_generator(iter_start, iter_end)

        return sample_iterator

    def __len__(self):
        return self.end - self.start

    def _sample_generator(self, start, end):
        if self.all_data_filenames_shuffled:
            all_data_filenames = self.all_data_filenames_shuffled
        else:
            all_data_filenames = self.all_data_filenames
            
        # 通过gpu数将数据集分割，保证不同gpu读取到不同数据
        data_filenames = all_data_filenames[self.local_rank :: self.world_size]
        for i, data_filename in enumerate(data_filenames):
            if i < start:
                continue
            if i >= end:
                return StopIteration()
            
            with open(data_filename, 'rb') as f:
                # print(data_filename, flush=True)
                traj = pickle.load(f)

            traj = self.process_traj(traj)
            obs_imgs, next_obs_imgs, goal_imgs, actions = self.convert_traj_to_samples(traj)

            for obs_img, next_obs_img, goal_img, action in zip(
                obs_imgs, next_obs_imgs, goal_imgs, actions
            ):
                obs_img = torch.tensor(obs_img).permute((2, 0, 1))
                next_obs_img = torch.tensor(next_obs_img).permute((2, 0, 1))
                goal_img = torch.tensor(goal_img).permute((2, 0, 1))

                if self.is_train and self.augment:
                    obs_img, next_obs_img, goal_img = self._augment(obs_img, next_obs_img, goal_img)

                yield obs_img, next_obs_img, goal_img, action

    def shuffle(self, seed):
        rng = np.random.default_rng(seed)
        self.all_data_filenames_shuffled = self.all_data_filenames
        rng.shuffle(self.all_data_filenames_shuffled)

    def process_traj(self, traj):
        traj = {
            "observations": {
                "image": traj["observations/images0"],
                "proprio": traj["observations/state"],
            },
            "next_observations": {
                "image": traj["next_observations/images0"],
                "proprio": traj["next_observations/state"],
            },
            "actions": traj["actions"],
            "terminals": traj["terminals"],
            "truncates": traj["truncates"],
        }

        traj = self._process_actions(traj)
        traj = self._add_goals(traj)

        return traj
    
    def convert_traj_to_samples(self, traj):
        obs_imgs = traj["observations"]["image"]
        obs_pros = traj["observations"]["proprio"]
        next_obs_imgs = traj["next_observations"]["image"]
        next_obs_pros = traj["next_observations"]["proprio"]
        actions = traj["actions"]
        terminals = traj["terminals"]
        truncates = traj["truncates"]
        goal_imgs = traj["goals"]["image"]
        goal_pros = traj["goals"]["proprio"]
        rewards = traj["rewards"]
        masks = traj["masks"]

        # return (obs_imgs, obs_pros, next_obs_imgs, next_obs_pros, actions,
        #         terminals, truncates, goal_imgs, goal_pros, rewards, masks)
    
        return obs_imgs, next_obs_imgs, goal_imgs, actions
    
    def _process_actions(self, traj):
        if self.relabel_actions:
            # relabel the first 6 action dims (xyz position, xyz rotation)
            # using the reached proprio
            movement_actions = (
                traj["next_observations"]["proprio"][:, :6]
                - traj["observations"]["proprio"][:, :6]
            )
            # binarize the gripper action
            continuous_gripper_actions = traj["actions"][:, 6]
            binarized_gripper_actions = binarize_gripper_actions(
                continuous_gripper_actions
            )

            traj["actions"] = np.concatenate(
                [movement_actions, binarized_gripper_actions[:, None]],
                axis=1,
            )

        # normalize actions
        if self.action_metadata is not None:
            traj["actions"] = (
                traj["actions"] - self.action_metadata["mean"]
            ) / self.action_metadata["std"]

        return traj

    def _add_goals(self, traj):
        traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
            traj, **self.goal_relabeling_kwargs
        )

        # after goal relabeling, we can set actions and obs to chunked version
        if "action_chunks" in traj:
            traj["actions"] = traj.pop("action_chunks")
        if "obs_chunks" in traj:
            traj["observations"] = traj.pop("obs_chunks")
            traj["next_observations"] = traj.pop("next_obs_chunks")

        return traj

    def _augment(self, observation_image, next_observation_image, goal_image):
        images = torch.stack((observation_image, next_observation_image, goal_image))

        images = self.augment_transform(images)

        # observation_image, next_observation_image, goal_image = torch.split(images, 1)
        return images[0], images[1], images[2]