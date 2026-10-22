"""SAC (stable-baselines3) with an asymmetric actor-critic, as in L2F: the actor reads observation["actor"], what the
airplane measures, and the critics read observation["critic"], which adds privileged simulator information.

Needs pip install -e ".[rl]". Kept in the package (not in the training script) so that a saved model loads anywhere.
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy

from .environment import WEIGHTS, EnvConfig, RewardCurriculum, reward_from_observations


class SelectKey(BaseFeaturesExtractor):
    """Passes one part of the dict observation through, unchanged."""

    def __init__(self, observation_space: spaces.Dict, key: str = "actor"):
        super().__init__(observation_space, features_dim=observation_space[key].shape[0])
        self.key = key

    def forward(self, observations):
        return observations[self.key]


class AsymmetricSACPolicy(SACPolicy):
    """By default SB3 gives the same observation to every network; here each one gets its own part."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", SelectKey)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None):
        return super().make_actor(SelectKey(self.observation_space, "actor"))

    def make_critic(self, features_extractor=None):
        return super().make_critic(SelectKey(self.observation_space, "critic"))


class SB3Policy:
    """A trained model with the interface of environment.evaluate (mean action, no sampling)."""

    def __init__(self, model):
        self.model = model

    def reset(self, env):
        pass

    def __call__(self, env, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action


class CurriculumCallback(BaseCallback):
    """Applies RewardCurriculum during training: sends the new weights to every environment and recomputes the
    rewards already in the replay buffer, so that the critics learn one reward at a time (L2F, p. 4 and 6).

    A stage is taken at the end of a period only if the mean course error over that period is below
    curriculum.advance_course_error. The mean course error is logged as train/course_error_deg.
    """

    def __init__(self, curriculum: RewardCurriculum, config: EnvConfig):
        super().__init__()
        self.curriculum = curriculum
        self.config = config
        self.stage = 0
        self.current = None
        self.error_sum = 0.0
        self.error_count = 0
        self.next_check = curriculum.period

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "course_error" in info:
                self.error_sum += abs(info["course_error"])
                self.error_count += 1
        mean_error = self.error_sum / max(self.error_count, 1)
        self.logger.record("train/course_error_deg", float(np.rad2deg(mean_error)))
        if self.num_timesteps >= self.next_check:
            if mean_error < self.curriculum.advance_course_error and self.stage < self.curriculum.n_stages:
                self.stage += 1
            self.next_check += self.curriculum.period
            self.error_sum, self.error_count = 0.0, 0
        reward_config = self.curriculum.at_stage(self.stage)
        if reward_config != self.current:
            self.current = reward_config
            self.training_env.env_method("set_reward", reward_config)
            self._recompute(reward_config)
        self.logger.record("curriculum/stage", self.stage)
        for name in WEIGHTS:
            self.logger.record(f"curriculum/{name}", getattr(reward_config, name))
        return True

    def _recompute(self, reward_config):
        b = self.model.replay_buffer
        n = b.buffer_size if b.full else b.pos
        if n == 0:
            return
        observation = b.observations["critic"][:n]
        next_observation = b.next_observations["critic"][:n]
        flat = lambda x: x.reshape(-1, x.shape[-1])
        terminated = (b.dones[:n] * (1 - b.timeouts[:n])).reshape(-1) > 0.5  # a time limit is not a termination
        rewards = reward_from_observations(
            flat(observation), flat(next_observation), flat(b.actions[:n]), terminated, self.config, reward_config)
        b.rewards[:n] = rewards.reshape(b.rewards[:n].shape)
