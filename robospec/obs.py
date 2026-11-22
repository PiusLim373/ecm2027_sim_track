"""Decode gRPC observations into numpy arrays (schema owned by upstream chamber_schema)."""
import numpy as np

from robofinals.distributed.chamber_schema import OBSERVATION_SCHEMA

FIELDS = {name: (np.dtype(dtype).newbyteorder("<"), shape) for name, dtype, shape in OBSERVATION_SCHEMA}
CAMERAS = ("first_person_camera_rgb", "left_hand_camera_rgb", "right_hand_camera_rgb")
ACTION_DIM = 19
CONTROL_DT = 0.02  # seconds of sim time per Step


def decode(observation):
    """One pb.Observation -> {field: array} with the README shapes."""
    return {name: np.frombuffer(getattr(observation, name), dtype=dtype).reshape(shape).copy()
            for name, (dtype, shape) in FIELDS.items()}
