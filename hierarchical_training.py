import argparse
import random

from gym import spaces
import numpy as np
from pycrorts3 import HierarchicalPycroRts3MultiAgentEnv
from pycrorts3.game.actions import ActionTypes
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.models import ParametricActionsMLP, ParametricActionsCNN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--restore', type=str)
    args = parser.parse_args()

    ray.init(local_mode=args.debug)
    if args.debug:
        tune_config = {
            'log_level': 'DEBUG',
            'num_workers': 1,
        }
    else:
        tune_config = {
            # 'log_level': 'DEBUG',
            'num_workers': 1,
        }

    env_cls = HierarchicalPycroRts3MultiAgentEnv
    if args.use_cnn:
        # env_cls = SquareConnect4Env
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsCNN)
        model_config = {
            'custom_model': 'parametric_actions_model',
            'conv_filters': [[16, [2, 2], 1], [32, [2, 2], 1], [64, [3, 3], 2]],
            'conv_activation': 'leaky_relu',
            'fcnet_hiddens': [128, 128],
            'fcnet_activation': 'leaky_relu',
        }
    else:
        # env_cls = Connect4Env
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsMLP)
        model_config = {
            'custom_model': 'parametric_actions_model',
            'fcnet_hiddens': [128, 128],
            'fcnet_activation': 'leaky_relu',
        }

    register_env('pycrorts', lambda cfg: env_cls(cfg))
    env = env_cls()
    obs_space, action_space = env.observation_space, env.action_space

    def hierarchical_policy_mapping_fn(agent_id):
        if agent_id.startswith('low_level_'):
            return 'low_level_policy'
        else:
            return 'high_level_policy'


    tune.run(
        'PPO',
        name='main',
        stop={
            'timesteps_total': int(1e6),
        },
        config=dict({
            'env': 'pycrorts',
            'env_config': {},
            'lr': 0.001,
            'clip_param': 0.2,
            'lambda': 0.95,
            'kl_coeff': 1.0,
            'entropy_coeff': 0.01,
            'multiagent': {
                'policy_mapping_fn': hierarchical_policy_mapping_fn,
                'policies': {
                    'high_level_policy': (None,
                                          obs_space,
                                          spaces.Discrete(len(ActionTypes)),
                                          {'gamma': 0.9}),
                    'low_level_policy': (None,
                                         spaces.Tuple([obs_space, spaces.Discrete(len(ActionTypes))]),
                                         action_space,
                                         {'gamma': 0.0}),
                },
            },
        }, **tune_config),
        # checkpoint_at_end=True,
        restore=args.restore,
    )
