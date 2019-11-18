import argparse

from pycrorts3 import PycroRts3MultiAgentEnv, SquarePycroRts3MultiAgentEnv
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from src.models import ParametricActionsMLP, ParametricActionsCNN
from src.policies import RandomPolicy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default='PPO')
    parser.add_argument('--map', type=str, default='4x4_melee_light2')
    parser.add_argument('--use-cnn', action='store_true')
    parser.add_argument('--num-learners', type=int, default=2)
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
            'num_workers': 1,
            'num_gpus': 1,
        }

    env_config = {'map_filename': args.map}
    if args.use_cnn:
        env_cls = SquarePycroRts3MultiAgentEnv
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsCNN)
        model_config = {
            'custom_model': 'parametric_actions_model',
            'conv_filters': [[16, [2, 2], 1], [32, [2, 2], 1], [64, [3, 3], 2]],
            'conv_activation': 'leaky_relu',
            'fcnet_hiddens': [128, 128],
            'fcnet_activation': 'leaky_relu',
        }
    else:
        env_cls = PycroRts3MultiAgentEnv
        ModelCatalog.register_custom_model('parametric_actions_model', ParametricActionsMLP)
        model_config = {
            'custom_model': 'parametric_actions_model',
            'fcnet_hiddens': [128, 128],
            'fcnet_activation': 'leaky_relu',
        }
    register_env('pycrorts', lambda cfg: env_cls(cfg))
    env = env_cls(env_config)
    obs_space, action_space = env.observation_space, env.action_space

    trainable_policies = {
        f'learned{i:02d}': (None, obs_space, action_space, {'model': model_config}) for i in range(args.num_learners)
    }
    if args.policy == 'PPO':
        tune_config.update({
            'lr': 0.001,
            'gamma': 0.995,
            'clip_param': 0.2,
            'lambda': 0.95,
            'kl_coeff': 1.0,

            'num_workers': 5,
            # 'num_workers': 20,
            # 'num_envs_per_worker': 8,
            'train_batch_size': 32000,
            # 'train_batch_size': 64000,
            # 'train_batch_size': 128000,
            # 'train_batch_size': 480000,  # 20*8*3000 (workers * num_envs_per_wkr * max_steps_per_game)
        })
    elif args.policy in ['DQN', 'APEX']:
        tune_config.update({
            'hiddens': [],
            'dueling': False,
        })
        if args.policy == 'APEX':
            tune_config.update({
                'num_workers': 5,
            })

    def random_policy_mapping_fn(agent_id):
        player_id, unit_id = agent_id.split('.')
        if player_id == '0':
            return 'learned00'
        else:
            return 'random'

    def name_trial(trial):
        """Give trials a more readable name in terminal & Tensorboard."""
        debug = '-debug' if args.debug else ''
        return f'{trial.trainable_name}-{args.map}{debug}'

    tune.run(
        args.policy,
        name='pycrorts',
        trial_name_creator=name_trial,
        stop={
            # 'timesteps_total': int(1e6),
            'timesteps_total': int(10e6),
            # 'policy_reward_mean/learned00': 0.6,
        },
        config=dict({
            'env': 'pycrorts',
            'env_config': env_config,
            'multiagent': {
                'policies': {
                    **trainable_policies,
                    'random': (RandomPolicy, obs_space, action_space, {}),
                },
                'policies_to_train': [*trainable_policies],
                'policy_mapping_fn': random_policy_mapping_fn,
            },
        }, **tune_config),
        checkpoint_at_end=True,
        restore=args.restore,
    )
