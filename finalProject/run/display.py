import display_20220521_v1 as example

from env.display_env import DisplayEnv
from utils.define import ACTION_INFO, COMBAT_OBS_INFO

import time
import random
import numpy as np
import tensorflow as tf

from tf_agents.policies import policy_loader
from tf_agents.trajectories.time_step import time_step_spec
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.specs.array_spec import BoundedArraySpec, ArraySpec

_IP = ['127.0.0.1'] * 2
_PORT = [20000, 20001]


def _make_random_policy():
    random_policy = RandomPyPolicy(
        time_step_spec(
            BoundedArraySpec((COMBAT_OBS_INFO[0],), np.float32, minimum=COMBAT_OBS_INFO[1], maximum=COMBAT_OBS_INFO[2],
                             name='observation'),
            ArraySpec(shape=(), dtype=np.float32, name='reward')
        ),
        BoundedArraySpec((ACTION_INFO[0],), np.float32, minimum=ACTION_INFO[1], maximum=ACTION_INFO[2], name='action')
    )
    get_state_fcn = lambda _: _
    return random_policy, get_state_fcn, COMBAT_OBS_INFO


def _gen_random_init_pos():
    return [random.uniform(-5e3, 5e3) for _ in range(2)] + [random.uniform(1e3, 3e3)] + \
           [random.uniform(COMBAT_OBS_INFO[1][i], COMBAT_OBS_INFO[2][i]) for i in range(3, 6)]


def main():
    # policy = policy_loader.load('./save/example/sac/eval_policy')
    policy = policy_loader.load('./save/20220521/sac/eval_policy')

    opp_policy, opp_state_fcn, opp_state_info = _make_random_policy()

    display = DisplayEnv(
        ip=_IP, port=_PORT,

        state_size=[len(example.state_min), opp_state_info[0]],
        state_min=[example.state_min, opp_state_info[1]],
        state_max=[example.state_max, opp_state_info[2]],
        get_state_fcn=[example.get_state, opp_state_fcn],
        policies=[policy, opp_policy],

        gen_init_pos_fcn=_gen_random_init_pos
    )

    ending = False
    for _ in range(5): 
        display.reset()
        while not ending:
            time.sleep(0.1) 
            ending = display.step()
        ending = False
        time.sleep(5.) 


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config).as_default():
        main()
