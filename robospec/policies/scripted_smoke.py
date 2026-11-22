"""Open-loop motion test: base, both arms and gripper, one phase at a time."""
import numpy as np

from robospec.policies.base import Policy

# Base-frame link7 home poses [x, y, z, qx, qy, qz, qw] taken from demo starts.
HOME_L = np.array([0.349, 0.299, 0.989, -0.005, -0.991, -0.130, 0.038], np.float32)
HOME_R = np.array([0.349, -0.299, 0.989, 0.005, -0.991, 0.130, 0.038], np.float32)

# (name, steps, base [vx, vy, wz], left arm offset xyz, right arm offset xyz, gripper mode)
# Arm offsets are peak values, ramped out and back with a half sine over the phase.
# Gripper mode "pulse_l" / "pulse_r": that gripper closes in the first half, opens in the second.
PHASES = [
    ("hold home pose",            50, (0.0, 0.0, 0.0), None,             None,             None),
    ("backward vx=-0.2 m/s",     100, (-0.2, 0.0, 0.0), None,            None,             None),
    ("strafe left vy=0.2 m/s",   100, (0.0, 0.2, 0.0), None,             None,             None),
    ("rotate wz=0.3 rad/s",      100, (0.0, 0.0, 0.3), None,             None,             None),
    ("stop",                      50, (0.0, 0.0, 0.0), None,             None,             None),
    ("right arm up 10 cm + back", 150, (0.0, 0.0, 0.0), None,            (0.0, 0.0, 0.10), None),
    ("left arm out 10 cm + back", 150, (0.0, 0.0, 0.0), (0.0, 0.10, 0.0), None,            None),
    ("left gripper close/open",   100, (0.0, 0.0, 0.0), None,             None,             "pulse_l"),
    ("right gripper close/open",  100, (0.0, 0.0, 0.0), None,             None,             "pulse_r"),
    ("hold",                      50, (0.0, 0.0, 0.0), None,             None,             None),
]
TOTAL_STEPS = sum(p[1] for p in PHASES)


def phase_at(t: int):
    """Return (phase tuple, progress in [0, 1)) for step t; holds the last phase after the end."""
    for phase in PHASES:
        if t < phase[1]:
            return phase, t / phase[1]
        t -= phase[1]
    return PHASES[-1], 0.0


class ScriptedSmoke(Policy):
    max_steps = TOTAL_STEPS

    def __init__(self):
        self.t = {}

    def reset(self, env_index):
        self.t[env_index] = 0

    def act(self, observations):
        actions = np.zeros((len(observations), 19), np.float32)
        for i in range(len(observations)):
            t = self.t.get(i, 0)
            (_, _, base, off_l, off_r, grip), s = phase_at(t)
            ramp = np.sin(np.pi * s)
            a = actions[i]
            a[0:7], a[7:14] = HOME_L, HOME_R
            if off_l is not None:
                a[0:3] += ramp * np.asarray(off_l, np.float32)
            if off_r is not None:
                a[7:10] += ramp * np.asarray(off_r, np.float32)
            a[14] = a[15] = 1.0  # >= 0 open, < 0 close
            if grip in ("pulse_l", "pulse_r") and s < 0.5:
                a[14 if grip == "pulse_l" else 15] = -1.0
            a[16:19] = base
            self.t[i] = t + 1
        return actions
