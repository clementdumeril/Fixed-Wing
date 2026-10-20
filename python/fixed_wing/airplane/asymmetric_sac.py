"""SAC (stable-baselines3) with an asymmetric actor-critic, as in L2F: the actor reads observation["actor"], what the
airplane measures, and the critics read observation["critic"], which adds privileged simulator information.

Needs pip install -e ".[rl]". Kept in the package (not in the training script) so that a saved model loads anywhere.
"""
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy


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
