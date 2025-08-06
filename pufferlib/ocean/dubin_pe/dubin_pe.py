import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.dubin_pe import binding

class DubinPE(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        num_cars=3,  # Default to 1 evader + 2 pursuers
        render_mode=None,
        report_interval=1024,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(17,),  # 17 observations per agent: pos(2), heading(1), speed(1), spawn_pos(2), target_rel(4), rewards(3), type(1), nearest(3)
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Discrete(3)  # 0=left, 1=straight, 2=right

        self.num_agents = num_envs*num_cars
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0

        super().__init__(buf)
        # Convert actions to int32 for discrete actions
        self.actions = self.actions.astype(np.int32)

        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i*num_cars:(i+1)*num_cars],
                self.actions[i*num_cars:(i+1)*num_cars],
                self.rewards[i*num_cars:(i+1)*num_cars],
                self.terminals[i*num_cars:(i+1)*num_cars],
                self.truncations[i*num_cars:(i+1)*num_cars],
                i,
                num_agents=num_cars,
            ))

        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        # Convert continuous actions to discrete if needed
        if isinstance(actions, np.ndarray) and actions.dtype == np.float32:
            # Map from continuous [-1, 1] to discrete {0, 1, 2}
            discrete_actions = np.clip((actions + 1) * 1.5, 0, 2).astype(np.int32)
            self.actions[:] = discrete_actions
        else:
            self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = DubinPE(num_envs=100, num_cars=3)  # Test with smaller setup
    env.reset()
    tick = 0

    # Generate discrete actions for testing
    actions = [np.random.randint(0, 3, size=env.num_agents) for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")

if __name__ == "__main__":
    test_performance()
