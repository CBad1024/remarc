import gymnasium
import numpy as np
import pandas as pd
from examples.train import log_trajectory_step


def run_one_drug_sim(
    env: gymnasium.Env,
    drug_A: int = 0,
    num_episodes: int = 10,
    episode_length: int = 20,
    signature: str | None = None,
):
    """
    Simulates the environment for a number of episodes using a given policy.

    Args:
        env: the evol_env_wf environment
        drug_A (int): The drug to use.
        num_episodes (int): The number of simulation episodes.
        episode_length (int): The length of each episode.

    Returns:
        pd.DataFrame: A dataframe containing the simulation history.
    """
    # print("Running simulation ...")
    states = []
    actions = []
    time_steps = []
    episodes = []
    fitnesses = []

    for i in range(num_episodes):
        env.reset()
        obs = env.get_obs()
        # print(obs)

        for j in range(episode_length):
            states.append(obs)

            action = int(drug_A)

            obs, rew, terminated, truncated, info = env.step(action)
            actions.append(int(action))
            fitnesses.append(env.get_fitness())
            time_steps.append(j)
            episodes.append(i)

            # Real-time trajectory logging
            log_trajectory_step(
                signature, i, j, int(np.argmax(obs)), env.get_fitness(), int(action)
            )

    results_df = pd.DataFrame(
        {
            "Episode": episodes,
            "Time Step": time_steps,
            "State": states,
            "Action": actions,
            "Fitness": fitnesses,
        }
    )
    return results_df
