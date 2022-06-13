from env.base_env import norm, denorm
from utils.define import ACTION_INFO, COMBAT_OBS_INFO

import math
import numpy as np

from typing import Optional
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.trajectories.time_step import TimeStep
from tf_agents.trajectories.policy_step import PolicyStep
from tf_agents.typing.types import Seed, NestedArray
from tf_agents.trajectories.time_step import time_step_spec
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec


def _add_bias(num):
    if abs(num) < 1e-4:
        num = 1e-4 if num > 0 else -1e-4
    return num


def get_target_euler_diff(rx, ry, rz, roll, pitch, yaw):
    rx, ry = _add_bias(rx), _add_bias(ry)

    target_pitch = math.atan(rz / math.sqrt(rx ** 2 + ry ** 2))
    pitch_diff = pitch - target_pitch

    target_yaw = math.atan(ry / rx) if rx > 0 else math.atan(ry / rx) + math.pi
    target_yaw = target_yaw if target_yaw > 0 else 2 * math.pi + target_yaw
    yaw_diff = yaw - target_yaw
    if yaw_diff < -math.pi:
        yaw_diff = 2 * math.pi + yaw_diff
    elif yaw_diff > math.pi:
        yaw_diff = 2 * math.pi - yaw_diff

    return 0., pitch_diff, yaw_diff


def calc_pid(p, i, d):
    return 2000. * p + 10. * i + 50. * d


def min_max(x, mn, mx):
    if x > mx:
        return mx
    elif x < mn:
        return mn
    return x


class PIDPolicy(PyPolicy):
    def __init__(self):
        self.pitch_error_sum = 0.
        self.yaw_error_sum = 0.
        self.pitch_error_last = 0.
        self.yaw_error_last = 0.
        ts_spec, = time_step_spec(
            BoundedArraySpec((COMBAT_OBS_INFO[0],), np.float32,
                             minimum=COMBAT_OBS_INFO[1], maximum=COMBAT_OBS_INFO[2], name='observation'),
            ArraySpec(shape=(), dtype=np.float32, name='reward')
        ),
        action_spec = BoundedArraySpec((ACTION_INFO[0],), np.float32,
                                       minimum=ACTION_INFO[0]*[-1], maximum=ACTION_INFO[0]*[1], name='action')
        super(PIDPolicy, self).__init__(
            time_step_spec=ts_spec,
            action_spec=action_spec
        )

    def _action(self,
                time_step: TimeStep,
                policy_state: NestedArray,
                seed: Optional[Seed] = None) -> PolicyStep:

        obs = denorm(time_step.observation, COMBAT_OBS_INFO[1], COMBAT_OBS_INFO[2])
        rx, ry, rz, roll, pitch, yaw = obs[14] - obs[0], obs[15] - obs[1], obs[16] - obs[2], obs[3], obs[4], obs[5]
        vx, vy, vz = obs[20] - obs[6], obs[21] - obs[7], obs[22] - obs[8]
        evx, evy, evz = obs[20], obs[21], obs[22]
        epx, epy, epz = obs[14], obs[15], obs[16]
        print("evx, evy, evz", evx, evy, evz)
        print("epx, epy, epz", epx, epy, epz)
        pitch_omega, yaw_omega = obs[10], obs[11]

        dist = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

        roll_diff, pitch_diff, yaw_diff = get_target_euler_diff(rx, ry, rz, roll, pitch, yaw)

        pitch_action = 400. if pitch_diff < 0. else -400.
        yaw_action = 400. if yaw_diff < 0. else -400.
        throttle_action = 1000. if dist > 300. else 0.

        pitch_error = -pitch_diff
        yaw_error = -yaw_diff

        print("pitch_omega", pitch_omega, "pitch_error", pitch_error)
        print("yaw_omega", yaw_omega, "yaw_error", yaw_error)

        pitch_action = calc_pid(pitch_error, pitch_error + self.pitch_error_sum, pitch_error - self.pitch_error_last)
        yaw_action = calc_pid(yaw_error, yaw_error + self.yaw_error_sum, yaw_error - self.yaw_error_last)
        pitch_action = min_max(pitch_action, -1e3, 1e3)
        yaw_action = min_max(yaw_action, -1e3, 1e3)
        throttle_action = 1000. if dist > 400. else dist / 2.

        self.pitch_error_sum += pitch_error
        self.yaw_error_sum += yaw_error
        self.pitch_error_last = pitch_error
        self.yaw_error_last = yaw_error

        norm_action = norm(np.array([0., pitch_action, yaw_action, throttle_action]), ACTION_INFO[1], ACTION_INFO[2])

        print("pitch_diff", pitch_diff, "pitch_action", pitch_action)
        print("yaw_diff", yaw_diff, "yaw_action", yaw_action)

        return PolicyStep(norm_action, None, None)
