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

# REPO_NAME = "dannyXSC/mimicgen"  # Name of the output dataset, also used for the Hugging Face Hub
# DATASET_PATH_LIST = {"/tmp/core_datasets/mug_cleanup/demo_src_mug_cleanup_task_D0/demo.hdf5":"pick the mug into the desk"}

def main(data_dir: str, task_name: str, task_description: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / task_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=task_name,
        robot_type="Panda",
        fps=30,
        features={
            "agentview_image": {
                "dtype": "video",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "robot0_eye_in_hand_image": {
                "dtype": "video",
                "shape": (84, 84, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),
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
    with h5py.File(data_dir) as file:
        demos = file["data"]
        for i in range(len(demos)):
            
            demo = demos[f"demo_{i}"]
            actions = demo["actions"][:]
            
            agentview_image = demo["obs/agentview_image"][:]
            robot0_eye_in_hand_image = demo["obs/robot0_eye_in_hand_image"][:]
            robot0_eef_pos = demo["obs/robot0_eef_pos"][:]
            robot0_eef_quat = demo["obs/robot0_eef_quat"][:]
            robot0_gripper_qpos = demo["obs/robot0_gripper_qpos"][:]
            state = np.concatenate([robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos], axis=-1)
            
            for j in range(len(actions)):
                dataset.add_frame(
                    {
                        "agentview_image": agentview_image[j],
                        "robot0_eye_in_hand_image": robot0_eye_in_hand_image[j],
                        "state": state[j],
                        "actions": actions[j],  
                    }
                )
            dataset.save_episode(task=task_description)
                
    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["h2r", "xarm"],
    #         private=True,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)
