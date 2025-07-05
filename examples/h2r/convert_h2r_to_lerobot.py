"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import h5py
import os
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "dannyXSC/h2r_video_test"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_DISCRIPTION = {
    "grab_cube2_v1": "grab the green cube into the plate",
    "grab_cup_v1": "grab the cup and change its position",
    "grab_pencil1_v1": "grab the pen into the plate",
    "grab_pencil2_v1": "grab the pen into the plate",
    "grab_to_plate1_and_back_v1": "grab the red cube into the green plate",
    "grab_to_plate1_v1": "grab the red cube into the green plate",
    "grab_to_plate2_and_back_v1": "grab the red cube into the yellow plate",
    "grab_to_plate2_v1": "grab the red cube into the yellow plate",
    "grab_to_plate2_and_pull_v1": "grab the red cube into the green plate and pull the plate",
    "grab_two_cubes2_v1": "grab the green cube into the plate",
    "pull_plate_v1": "pull the plate",
    "push_box_common_v1": "push the box",
    "push_box_random_v1": "push the box",
    "push_box_two_v1": "push the box",
    "push_plate_v1": "push the plate",
}


def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="xarm",
        fps=30,
        features={
            "human_image": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "robot_image": {
                "dtype": "video",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name, discription in RAW_DATASET_DISCRIPTION.items():
        # hdf5 reader
        dataset_path = os.path.join(data_dir, raw_dataset_name)
        # find the file under the dataset path ended with .hdf5
        files = [f for f in os.listdir(dataset_path) if f.endswith(".hdf5")]
        print(dataset_path)
        for episode in files:
            with h5py.File(os.path.join(dataset_path, episode), "r") as f:
                human_video = np.uint8(f["/cam_data/human_camera"])
                robot_video = np.uint8(f["/cam_data/robot_camera"])
                end_position = f["/end_position"][()]
                gripper_state = f["/gripper_state"][()]
                end_state = np.concatenate(
                    [end_position, gripper_state[:, None]], axis=1
                )
                action = f["/action"][()]

                # BGR -> RGB
                human_video = human_video[:, :, :, ::-1]
                robot_video = robot_video[:, :, :, ::-1]

                for i in range(len(human_video)):
                    dataset.add_frame(
                        {
                            "human_image": human_video[i],
                            "robot_image": robot_video[i],
                            "state": end_state[i],
                            "actions": action[i],
                        }
                    )
                dataset.save_episode(task=discription)

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["h2r", "xarm"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
