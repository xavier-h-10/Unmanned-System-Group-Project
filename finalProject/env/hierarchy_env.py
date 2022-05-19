from env.combat_env import CombatEnv

import numpy as np

from typing import Tuple, Callable, List
from tf_agents.typing.types import NestedArray
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories.time_step import transition
from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper


class HierarchyWrapper(PyEnvironmentBaseWrapper):
    def __init__(self,
                 env: CombatEnv,
                 policies: List[Tuple[Callable, Callable, Callable]]):
        super(HierarchyWrapper, self).__init__(env)

        self._loaded = False
        self._policies = policies
        self._action_spec = BoundedArraySpec((), np.int32, minimum=0, maximum=len(policies) - 1)

    def _step(self, action):
        if not self._loaded:
            self._loaded = True
            self._policies = [(t[0](), t[1], t[2]) for t in self._policies]
        origin_action = self._convert_back(action).astype(np.float32)
        return self._env.step(origin_action)

    def action_spec(self) -> NestedArray:
        return self._action_spec

    def _convert_back(self, after):
        def action_map(ad):
            raw_obs = self._env.all_observations[0]
            policy, process_obs, process_action = self._policies[ad]
            norm_obs = self._env.convert_state(process_obs(raw_obs), False).astype(np.float32)
            raw_action = policy.action(transition(
                observation=norm_obs,
                reward=0.
            )).action

            return process_action(raw_action)

        if after.ndim == 0:
            before = action_map(after)
        else:
            before = list(map(action_map, after))

        return np.array(before, dtype=np.float32)

