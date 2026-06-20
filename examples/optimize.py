import optuna
from pathlib import Path
import sys
import os

# CRITICAL: Prevent PyTorch from causing massive CPU thrashing during multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import time
import datetime

# Ensure remarc is in path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from remarc.agents.tianshou_agent import train_wf_landscapes
from remarc.core.hyperparameters import Presets as P

def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.01, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    
    # Static parameters
    epochs = 100
    gae_lambda = 0.95
    train_steps_per_epoch = 2000
    reward_scale = 100.0
    gen_per_step = 10
    
    p = P(
        state_shape=(4,),
        num_actions=4,
        buffer_size=20000,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        batch_size=batch_size,
        epochs=epochs,
        train_steps_per_epoch=train_steps_per_epoch,
        reward_scale=reward_scale,
        gen_per_step=gen_per_step,
        dataset="four_state",
        test_episodes=50,
        episode_steps=20
    )
    
    seeds = [42, 43, 44]
    rewards = []
    
    for seed in seeds:
        signature = f"optuna_trial_{trial.number}_seed_{seed}"
        try:
            reward = train_wf_landscapes(p, signature=signature, trial=trial, seed=seed)
            rewards.append(reward)
        except optuna.exceptions.TrialPruned as e:
            # If any seed fails the pruning check, abort the entire trial
            logging.info(f"Trial {trial.number} pruned at seed {seed}")
            raise e
        except Exception as e:
            logging.error(f"Trial {trial.number} failed at seed {seed} with exception: {e}")
            raise e
            
    mean_reward = sum(rewards) / len(rewards)
    logging.info(f"Trial {trial.number} Completed. Mean Reward: {mean_reward}")
    return mean_reward



class ETACallback:
    def __init__(self, total_trials: int, cooldown_seconds: int):
        self.total_trials = total_trials
        self.cooldown_seconds = cooldown_seconds
        self.durations = []
        
    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state.is_finished():
            if trial.datetime_start and trial.datetime_complete:
                duration = (trial.datetime_complete - trial.datetime_start).total_seconds()
                self.durations.append(duration)
            print(f"\nTrial {trial.number} finished. Cooling down CPU for {self.cooldown_seconds} seconds...")
            time.sleep(self.cooldown_seconds)
            completed_trials = len(self.durations)
            remaining_trials = self.total_trials - completed_trials
            if remaining_trials > 0 and completed_trials > 0:
                avg_duration = sum(self.durations) / completed_trials
                estimated_remaining_seconds = remaining_trials * (avg_duration + self.cooldown_seconds)
                eta = datetime.timedelta(seconds=int(estimated_remaining_seconds))
                print(f"--- Estimated Time Remaining for {remaining_trials} trials: {eta} ---\n")

if __name__ == "__main__":
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "wf_landscapes_optimization"
    
    # Check for Cloud DB URL
    db_url = os.environ.get("OPTUNA_DB_URL")
    if db_url:
        storage_url = db_url
        safe_print_url = db_url.split('@')[1] if '@' in db_url else 'cloud database'
        print(f"Connecting to Cloud Postgres DB at {safe_print_url}...")
        
        from sqlalchemy.pool import NullPool
        storage_backend = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"poolclass": NullPool, "pool_pre_ping": True}
        )
    else:
        # Local SQLite fallback
        db_path = project_root / "log" / "optuna_study.db"
        db_path.parent.mkdir(exist_ok=True)
        storage_url = f"sqlite:///{db_path}"
        storage_backend = storage_url
        print(f"Connecting to local SQLite DB at {db_path}...")
    
    # Set up median pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1
    )
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_backend,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner
    )
    
    print(f"Starting Optuna optimization for Wright-Fisher landscapes...")
    
    total_trials = 1000
    cooldown = 0
    study.optimize(objective, n_trials=total_trials, n_jobs=32, callbacks=[ETACallback(total_trials, cooldown)])
    
    print("\nOptimization finished.")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (test reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
