from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Presets:
    state_shape: tuple[int, ...]
    num_actions: int | Sequence[int] | None
    lr: float
    epochs: int
    train_steps_per_epoch: int
    test_episodes: int
    batch_size: int
    buffer_size: int
    dataset: str = "eight_state"
    activation: str | None = "relu"
    reward_clip: bool = False
    gen_per_step: int = 500
    ent_coef: float = 0.05
    gamma: float = 0.99
    gae_lambda: float = 0.95
    delta_multiplier: float = 0.0
    episode_steps: int = 20
    reward_scale: float = 100.0
    random_start: bool = True
    landscape_amplification: float = 1.0
    stochastic: bool = True
    n_frames: int = 1
    delta_horizon: int = 1

    @staticmethod
    def p1_ss():
        return Presets(
            state_shape=(16,),
            num_actions=15,
            lr=0.0001,
            epochs=50,
            train_steps_per_epoch=4000,
            test_episodes=10,
            batch_size=128,
            buffer_size=20000,
            dataset="eight_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=20,
            reward_scale=100.0,
            random_start=True,
        )

    @staticmethod
    def p1_test():
        return Presets(
            state_shape=(16,),
            num_actions=15,
            lr=0.0001,
            epochs=2,
            train_steps_per_epoch=10,
            test_episodes=2,
            batch_size=8,
            buffer_size=50,
            dataset="eight_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=20,
            reward_scale=100.0,
            random_start=True,
        )

    @staticmethod
    def p1_ls():
        return Presets(
            state_shape=(16,),
            num_actions=15,
            lr=0.0001,
            epochs=50,
            train_steps_per_epoch=4000,
            test_episodes=10,
            batch_size=128,
            buffer_size=20000,
            dataset="eight_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=20,
            reward_scale=100.0,
            random_start=True,
        )

    @staticmethod
    def p2_ls():
        return Presets(
            state_shape=(16,),
            num_actions=15,
            lr=1e-4,
            epochs=200,
            train_steps_per_epoch=8000,
            test_episodes=10,
            batch_size=128,
            buffer_size=50000,
            dataset="eight_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=20,
            reward_scale=100.0,
            random_start=True,
        )

    @staticmethod
    def p2_test():
        return Presets(
            state_shape=(16,),
            num_actions=150,
            lr=0.001,
            epochs=2,
            train_steps_per_epoch=10,
            test_episodes=2,
            batch_size=8,
            buffer_size=50,
            dataset="eight_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=20,
            reward_scale=1.0,
            random_start=True,
        )

    @staticmethod
    def p_four_state_ls():
        return Presets(
            state_shape=(4,),
            num_actions=4,
            lr=0.0001,
            epochs=50,
            train_steps_per_epoch=4000,
            test_episodes=10,
            batch_size=128,
            buffer_size=20000,
            dataset="four_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=100,
            reward_scale=100.0,
            random_start=True,
        )

    @staticmethod
    def p_four_state_stacked():
        return Presets(
            state_shape=(12,),
            num_actions=4,
            lr=0.0001,
            epochs=50,
            train_steps_per_epoch=4000,
            test_episodes=10,
            batch_size=128,
            buffer_size=20000,
            dataset="four_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=100,
            reward_scale=100.0,
            random_start=True,
            n_frames=3,
        )

    @staticmethod
    def p_three_state_ls():
        return Presets(
            state_shape=(3,),
            num_actions=4,
            lr=0.0001,
            epochs=50,
            train_steps_per_epoch=4000,
            test_episodes=10,
            batch_size=128,
            buffer_size=20000,
            dataset="three_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=100,
            reward_scale=100.0,
            random_start=True,
        )

    @staticmethod
    def p_three_state_stacked():
        return Presets(
            state_shape=(9,),
            num_actions=4,
            lr=0.0001,
            epochs=50,
            train_steps_per_epoch=4000,
            test_episodes=10,
            batch_size=128,
            buffer_size=20000,
            dataset="three_state",
            activation="relu",
            reward_clip=False,
            ent_coef=0.05,
            episode_steps=100,
            reward_scale=100.0,
            random_start=True,
            n_frames=3,
        )
