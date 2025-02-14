import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# def make_h2r_example() -> dict:
#     """Creates a random input example for the Libero policy."""
#     return {
#         "observation/state": np.random.rand(8),
#         "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image": np.random.randint(
#             256, size=(224, 224, 3), dtype=np.uint8
#         ),
#         "prompt": "do something",
#     }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _encode_rotation(rotation) -> np.ndarray:
    rot_ember = transforms.EulerRotationEmbed()
    # position 3 + rotation 6 + gripper 1 = 10
    new_shape = rotation.shape[:-1] + (10,)
    result = np.zeros(new_shape)  # 可以用 np.zeros 或 np.empty，根据需求决定

    result[..., :3] = rotation[..., :3]
    result[..., 3:9] = rot_ember.angles_to_embed(rotation[..., 3:6])
    result[..., 9] = rotation[..., 6]
    return result


def _decode_rotation(embed) -> np.ndarray:
    rot_ember = transforms.EulerRotationEmbed()
    # position 3 + rotation 6 + gripper 1 = 10
    new_shape = embed.shape[:-1] + (7,)
    result = np.zeros(new_shape)  # 可以用 np.zeros 或 np.empty，根据需求决定

    result[..., :3] = embed[..., :3]
    result[..., 3:6] = rot_ember.embed_to_angles(embed[..., 3:9])
    result[..., 6] = embed[..., 9]
    return result


@dataclasses.dataclass(frozen=True)
class H2rInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    if_rotation_embed: bool = True

    def __call__(self, data: dict) -> dict:
        mask_padding = (
            self.model_type == _model.ModelType.PI0
        )  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        # rotation embedding
        if self.if_rotation_embed:
            state = _encode_rotation(data["observation/state"])
        else:
            state = data["observation/state"]
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        # base_image = _parse_image(data["observation/human_image"])
        base_image = _parse_image(data["observation/robot_image"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                # TODO: mask meaning
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            # We are padding from 7 to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            if self.if_rotation_embed:
                actions = _encode_rotation(data["actions"])
            else:
                actions = data["actions"]
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": _decode_rotation(np.asarray(data["actions"][:, :10]))}
