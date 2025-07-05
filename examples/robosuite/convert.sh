uv run examples/robosuite/convert_robosuite_to_lerobot.py \
    --data_dir /tmp/core_datasets/mug_cleanup/demo_src_mug_cleanup_task_D0/demo.hdf5 \
    --task_name mug_cleanup \
    --task_description "pick the mug into the desk"

uv run examples/robosuite/convert_robosuite_to_lerobot.py \
    --data_dir /tmp/core_datasets/hammer_cleanup/demo_src_hammer_cleanup_task_D0/demo.hdf5 \
    --task_name hammer_cleanup \
    --task_description "pick the hammer into the desk"

uv run examples/robosuite/convert_robosuite_to_lerobot.py \
    --data_dir /tmp/core_datasets/kitchen/demo_src_kitchen_task_D0/demo.hdf5 \
    --task_name kitchen \
    --task_description "clean the kitchen"

uv run examples/robosuite/convert_robosuite_to_lerobot.py \
    --data_dir /tmp/core_datasets/square/demo_src_square_task_D0/demo.hdf5 \
    --task_name square \
    --task_description "insert the square ring onto the pole."

uv run examples/robosuite/convert_robosuite_to_lerobot.py \
    --data_dir /tmp/core_datasets/threading/demo_src_threading_task_D0/demo.hdf5 \
    --task_name threading \
    --task_description "insert the rod into the hole."