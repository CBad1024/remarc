import dataclasses
import datetime
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, VectorEnvWrapper
from tianshou.policy import PPOPolicy, BasePolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.logger.base import BaseLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from ..core.hyperparameters import Presets as P
from ..envs import WrightFisherEnv


# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

def get_activation(activation_name: str | None) -> type[nn.Module]:
    """Return a torch activation class by name, defaulting to ReLU."""
    if not activation_name:
        return nn.ReLU

    mapping = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    return mapping.get(activation_name.lower(), nn.ReLU)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class VectorRewardClip(VectorEnvWrapper):
    """Clips rewards produced by a vectorised environment to [reward_min, reward_max]."""

    def __init__(self, venv, reward_min: float = -5.0, reward_max: float = 5.0):
        super().__init__(venv)
        self.reward_min = reward_min
        self.reward_max = reward_max

    def step(self, action, id=None):
        if id is None:
            obs, rew, term, trunc, info = self.venv.step(action)
        else:
            obs, rew, term, trunc, info = self.venv.step(action, id)
        rew = np.clip(rew, self.reward_min, self.reward_max)
        return obs, rew, term, trunc, info


# ---------------------------------------------------------------------------
# Paths & logging infrastructure
# ---------------------------------------------------------------------------

# Resolve PROJECT_ROOT relative to this file's location (remarc/agents/tianshou_agent.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

log_path = os.path.join(PROJECT_ROOT, "log", "RL")
metrics_path = os.path.join(PROJECT_ROOT, "log", "metrics")
os.makedirs(log_path, exist_ok=True)
os.makedirs(metrics_path, exist_ok=True)

writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)


class MetricsLogger:
    """Writes per-epoch reward metrics to CSV files."""

    def __init__(self, signature: str):
        self.signature = signature
        self.filename = os.path.join(metrics_path, f"{signature}.csv") if signature else None

        if self.filename:
            with open(self.filename, "w") as f:
                f.write("epoch,mean_reward,std_reward,loss\n")

    def log(self, epoch: int, mean_reward: float, std_reward: float, loss=None):
        if self.filename:
            with open(self.filename, "a") as f:
                loss_str = f",{loss}" if loss is not None else ","
                f.write(f"{epoch},{mean_reward},{std_reward}{loss_str}\n")


class LossCapturingLogger(BaseLogger):
    """Wraps a TensorboardLogger to intercept the most recent training loss."""

    def __init__(self, base_logger: TensorboardLogger, last_loss_ref: list):
        self.base_logger = base_logger
        self.last_loss = last_loss_ref

    def __getattr__(self, name):
        return getattr(self.base_logger, name)

    def write(self, step_type: str, step: int, data: dict) -> None:
        self.base_logger.write(step_type, step, data)

    def prepare_dict_for_logging(self, log_data: dict) -> dict:
        return self.base_logger.prepare_dict_for_logging(log_data)

    def finalize(self) -> None:
        self.base_logger.finalize()

    def save_data(self, epoch: int, env_step: int, gradient_step: int, save_checkpoint_fn=None) -> None:
        self.base_logger.save_data(epoch, env_step, gradient_step, save_checkpoint_fn)

    def restore_data(self) -> tuple:
        return self.base_logger.restore_data()

    def restore_logged_data(self, log_path: str) -> dict:
        return self.base_logger.restore_logged_data(log_path)

    def log_update_data(self, log_data, step):
        loss = None

        if isinstance(log_data, dict):
            loss = log_data.get("loss") or log_data.get("loss/total") or log_data.get("loss/clip")
        elif hasattr(log_data, "loss"):
            loss = log_data.loss
        elif hasattr(log_data, "__getitem__"):
            try:
                loss = log_data["loss"]
            except (KeyError, TypeError):
                try:
                    loss = log_data["loss/total"]
                except (KeyError, TypeError):
                    pass

        # PPO stats dicts may wrap loss as {"mean": float, ...}
        if isinstance(loss, dict) and "mean" in loss:
            loss = loss["mean"]

        if loss is not None and not isinstance(loss, dict):
            try:
                self.last_loss[0] = float(loss)
            except (TypeError, ValueError):
                pass

        return self.base_logger.log_update_data(log_data, step)


# ---------------------------------------------------------------------------
# Policy snapshot logging
# ---------------------------------------------------------------------------

def log_policy_snapshot(signature: str, policy, n_states: int):
    """Serialize current policy Q-values / action logits to a JSON file for the dashboard."""
    if not signature:
        return

    log_dir = os.path.join(PROJECT_ROOT, "log", "policies")
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"{signature}_live.json")

    state_tensor = torch.FloatTensor(np.identity(n_states))

    with torch.no_grad():
        if hasattr(policy, "actor"):  # PPO
            actor_out = policy.actor(state_tensor)
            if isinstance(actor_out, tuple):
                actor_out = actor_out[0]
            q_values = actor_out.cpu().numpy().tolist()
        else:
            return

    snapshot = {
        "n_states": n_states,
        "q_values": q_values,  # Key reused from DQN era for UI compatibility
    }
    with open(filename, "w") as f:
        json.dump(snapshot, f)


# ---------------------------------------------------------------------------
# Environment loading helpers
# ---------------------------------------------------------------------------

def load_testing_envs():
    envs = pickle.load(open(os.path.join(log_path, "testing_envs.pkl"), "rb"))
    return envs


def load_best_policy(p: P, filename: str = "best_policy.pth", env_type: str = "wf", ppo: bool = True):
    # Build a fresh environment from Presets rather than relying on
    # testing_envs.pkl, which is overwritten by every training run and
    # may have a different obs_shape than the checkpoint being loaded.
    import math
    seq_length = int(math.log2(p.state_shape[0]))
    env = WrightFisherEnv(seq_length=seq_length, num_drugs=p.num_actions)
    test_envs = DummyVectorEnv([lambda: env])
    policy = get_ppo_policy(p, test_envs).eval()
    policy = load_best_fn(policy, filename)
    return policy


def load_random_policy(p: P):
    """Return a freshly initialised (untrained) PPO policy on WF landscapes.

    Note: This returns an *untrained* policy whose actions are effectively random.
    Used as a random-policy baseline for evaluation.
    """
    import math
    seq_length = int(math.log2(p.state_shape[0]))
    env = WrightFisherEnv(seq_length=seq_length, num_drugs=p.num_actions)
    test_envs = DummyVectorEnv([lambda: env])
    return get_ppo_policy(p, test_envs)


# ---------------------------------------------------------------------------
# Training — Wright-Fisher Landscapes
# ---------------------------------------------------------------------------

def save_run_params(p: P, signature: str | None):
    """Persist the full parameter set for this run to a JSON file.

    Saved to ``log/params/<signature>_params.json`` (or a timestamped
    fallback when no signature is provided).  Called at the very start
    of training so the config is recorded even if the run crashes.
    """
    params_dir = os.path.join(PROJECT_ROOT, "log", "params")
    os.makedirs(params_dir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = signature if signature else f"run_{ts}"
    filename = os.path.join(params_dir, f"{label}_params.json")

    payload = dataclasses.asdict(p)
    # Convert tuples (e.g. state_shape) to lists for JSON compatibility
    for k, v in payload.items():
        if isinstance(v, tuple):
            payload[k] = list(v)
    payload["signature"] = signature
    payload["timestamp"] = ts

    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Run parameters saved to: {filename}")


def train_wf_landscapes(p: P, signature: str | None = None):
    # --- Save params before anything else ---
    save_run_params(p, signature)

    v_N = int(np.log2(p.state_shape[0]))

    from ..core.landscapes import Landscape

    landscape_list = []

    if hasattr(p, "dataset") and p.dataset == "chen":
        from remarc.envs import define_chen_landscapes
        print("Using Chen landscapes for Wright-Fisher training (raw fitness, no normalization).")
        v_N = 3
        chen_data = define_chen_landscapes()
        num_drugs = len(chen_data)

        g_min, g_max = np.min(chen_data), np.max(chen_data)
        print(f"Chen Data: Min={g_min:.4f}, Max={g_max:.4f}, Range={g_max-g_min:.4f}")

        for i in range(num_drugs):
            landscape_list.append(Landscape(v_N, sigma=0.0, ls=chen_data[i], g_min=g_min, g_max=g_max))

    elif hasattr(p, "dataset") and p.dataset == "four_state":
        from remarc.envs import define_four_state_landscapes
        amp = getattr(p, "landscape_amplification", 1.0)
        print(f"Using Four-State landscapes for Wright-Fisher training (amplification={amp:.1f}).")
        v_N = 2
        four_state_data = define_four_state_landscapes(amplification=amp)
        num_drugs = len(four_state_data)

        g_min, g_max = np.min(four_state_data), np.max(four_state_data)
        print(f"Four-State Data: Min={g_min:.4f}, Max={g_max:.4f}, Range={g_max-g_min:.4f}")

        for i in range(num_drugs):
            landscape_list.append(Landscape(v_N, sigma=0.0, ls=four_state_data[i], g_min=g_min, g_max=g_max))

    else:
        print("Using synthetic random landscapes for Wright-Fisher training.")
        num_drugs = 10
        landscape_list = [Landscape(v_N, sigma=0.5) for _ in range(num_drugs)]

    # Save shared landscapes so train.py evaluation uses the same data
    print("Saving landscapes to active_landscapes.pkl")
    with open(os.path.join(log_path, "active_landscapes.pkl"), "wb") as f:
        pickle.dump(landscape_list, f)

    print(f"Number of drugs: {len(landscape_list)}")
    train_envs, test_envs = WrightFisherEnv.getEnv(
        4, 2,
        landscape_list=landscape_list,
        gen_per_step=getattr(p, "gen_per_step", 500),
        seq_length=v_N,
        random_start=getattr(p, "random_start", True),
        episode_steps=getattr(p, "episode_steps", 20),
        reward_scale=getattr(p, "reward_scale", 100.0),
        stochastic=getattr(p, "stochastic", True),
    )

    print("Saving testing environments to testing_envs.pkl")
    with open(os.path.join(log_path, "testing_envs.pkl"), "wb") as f:
        pickle.dump([getattr(worker, 'env') for worker in train_envs.workers], f)

    if getattr(p, "reward_clip", False):
        print("Enable reward clipping (WF)")
        train_envs = VectorRewardClip(train_envs, reward_min=-5.0, reward_max=5.0)

    policy = get_ppo_policy(p, train_envs)

    def save_best_v2(policy: BasePolicy):
        base_name = "best_policy.pth"
        filename = f"{Path(base_name).stem}_{signature}.pth" if signature else base_name
        os.makedirs(log_path, exist_ok=True)
        torch.save(policy.state_dict(), os.path.join(log_path, filename))
        print(f"Best policy saved to: {os.path.join(log_path, filename)}")

    train_collector = Collector(policy, train_envs, VectorReplayBuffer(p.buffer_size, 4))
    test_collector = Collector(policy, test_envs)

    # Warm-up collection before training begins
    train_collector.reset()
    train_collector.collect(n_step=p.batch_size * 10)

    print(f"Max epochs set to: {p.epochs}")

    sig = signature if signature else "wf_ls"
    metrics_logger = MetricsLogger(sig)
    last_loss = [None]  # Mutable container so nested callbacks can update it

    def test_fn(epoch, env_step):
        if test_collector:
            stats = test_collector.collect(n_episode=p.test_episodes)
            if stats.returns_stat:
                metrics_logger.log(epoch, stats.returns_stat.mean, stats.returns_stat.std, last_loss[0])
            log_policy_snapshot(sig, policy, 2**v_N)

    def train_fn(epoch, env_step):
        # Decay entropy more aggressively
        if epoch < p.epochs * 0.2:
            current_ent_coef = getattr(p, "ent_coef", 0.05)
        else:
            # Drop to 0.01 by 60% of training
            decay_steps = p.epochs * 0.4
            current_ent_coef = max(policy.ent_coef - (getattr(p, "ent_coef", 0.05) - 0.01) / decay_steps, 0.01)
        policy.ent_coef = current_ent_coef

    wrapped_logger = LossCapturingLogger(logger, last_loss)

    drug_trainer = OnpolicyTrainer(
        policy=policy,
        max_epoch=p.epochs,
        batch_size=p.batch_size,
        train_collector=train_collector,
        test_collector=test_collector,
        step_per_epoch=p.train_steps_per_epoch,
        repeat_per_collect=8,  # Recommended value for PPO
        episode_per_test=p.test_episodes,
        step_per_collect=p.train_steps_per_epoch,
        train_fn=train_fn,
        test_fn=test_fn,
        # pyrefly: ignore [bad-argument-type]
        stop_fn=lambda mean_rewards: None,
        save_best_fn=save_best_v2,
        logger=wrapped_logger,
    )
    result = drug_trainer.run()
    print(f"Drug Cycling Training finished with result: {result}")

    test_result = test_collector.collect(n_episode=p.test_episodes)
    print(f"Final testing result: {test_result}")


# ---------------------------------------------------------------------------
# Policy constructors
# ---------------------------------------------------------------------------

def get_ppo_policy(p: P, train_envs: DummyVectorEnv | VectorEnvWrapper) -> PPOPolicy:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activation = get_activation(p.activation)

    obs_shape = train_envs.get_env_attr("observation_space")[0].shape
    action_shape = train_envs.get_env_attr("action_space")[0].n

    net = Net(
        state_shape=obs_shape,
        hidden_sizes=[64, 64],
        activation=activation,
        device=device,
    )
    actor = Actor(
        preprocess_net=net,
        action_shape=action_shape,
        hidden_sizes=[32],
        device=device,
    ).to(device)
    critic = Critic(
        preprocess_net=net,
        hidden_sizes=[32],
        device=device,
    ).to(device)

    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=p.lr)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        action_space=train_envs.get_env_attr("action_space")[0],
        discount_factor=0.99,
        max_grad_norm=0.5,
        vf_coef=0.5,
        ent_coef=getattr(p, "ent_coef", 0.05),
        gae_lambda=0.95,
        reward_normalization=True,
        action_scaling=False,
        deterministic_eval=True,
        dual_clip=None,
        value_clip=True,
        eps_clip=0.2,
        advantage_normalization=True,
        recompute_advantage=False,
    )
    print("Policy Action Space:", policy.action_space)
    return policy


# ---------------------------------------------------------------------------
# Policy persistence
# ---------------------------------------------------------------------------

def load_best_fn(policy: BasePolicy, filename: str = "best_policy.pth", path: str = log_path) -> BasePolicy:
    full_path = os.path.join(path, filename)
    print(f"Loading best policy from: {full_path}")
    policy.load_state_dict(torch.load(full_path))
    policy.eval()
    return policy
