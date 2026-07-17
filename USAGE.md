# REMaRC Project Usage Guide

This document provides a comprehensive guide on how to navigate and use the REMaRC (Reinforcement-learning based Evolutionary Markovian Resistance Control) project.

## Directory Structure and Important Files

The project is structured into core logic, examples (runnable scripts), and tests. Here's a breakdown of the most important files and directories:

### Core Framework (`remarc/`)
This folder contains the core logic of the RL framework and evolutionary environments:
* **`remarc/envs/wright_fisher_env.py`**: The core gym environment implementing the Wright-Fisher evolutionary dynamics model. This is where the simulation happens.
* **`remarc/core/landscapes.py`**: Definitions for empirical and theoretical fitness landscapes (how different phenotypes perform under different drug conditions).
* **`remarc/core/hyperparameters.py`**: Configuration constants and hyperparameters for experiments.
* **`remarc/agents/tianshou_agent.py`**: The main reinforcement learning agent implementations utilizing the `tianshou` library (e.g., PPO).
* **`remarc/agents/onnx_agent.py` & `greedy_agent.py`**: Alternate agents, including an ONNX exporter for inference outside of Python and simple heuristic/greedy agents for baselining.

### Scripts and Examples (`examples/`)
Runnable scripts for training, evaluating, and visualizing policies:
* **`examples/train.py`**: The main entry point for training the Reinforcement Learning agents on various landscapes.
* **`examples/run_plots.py`**: Runs evaluations and generates comparison plots for trained policies vs baselines.
* **`examples/dashboard.py`**: The Streamlit application for interactive visualization of the models and landscapes.
* **`examples/cwt_analysis.py`**: Continuous Wavelet Transform (CWT) analysis tools for interpreting the dynamics.
* **`examples/optimize.py`**: Optuna-based hyperparameter optimization script.
* **`examples/export_model.py` & `examples/onnx_client.py`**: Tools to export the trained PyTorch policy to an `.onnx` file and a minimal client for running inference.

### Configuration & Tooling
* **`Makefile`**: Contains shortcuts for common commands (running tests, dashboard, etc.).
* **`pyproject.toml` & `uv.lock`**: Python packaging and dependency management files (using `uv`).
* **`run_experiments.sh`**: A bash script to run parallel evaluations across different datasets and configurations.

---

## How to Run the Project

This project uses `uv` for dependency management. Ensure you have installed the project via `uv sync` before running these commands.

### 1. Training a Policy
You can train a reinforcement learning agent using `train.py`.
```bash
uv run examples/train.py --wf-train
```
*(Additional flags can be passed to configure the state space, landscape, etc.)*

### 2. Interactive Streamlit Dashboard
To visually explore the fitness landscapes and see how different policies (RL vs Single Drug vs Random) behave interactively:
```bash
uv run streamlit run examples/dashboard.py
```
Alternatively, using the Makefile:
```bash
make dashboard
```

### 3. Evaluating and Generating Plots
To run rigorous evaluations of a policy against baselines on different state dimensions (e.g., 3-state, 4-state, 8-state datasets) and generate result plots:
```bash
uv run python examples/run_plots.py --dataset four_state --dh 5 --delta 0.5 --train
```

### 4. Running the Batch Experiment Script
If you want to run a full suite of experiments across 3-state, 4-state, and 8-state environments concurrently (will run up to 2 jobs in parallel):
```bash
bash run_experiments.sh
```
Check the generated `log_run_*.txt` files for output logs.

### 5. Hyperparameter Optimization
To run hyperparameter tuning with Optuna:
```bash
uv run python examples/optimize.py
# Or via make:
make optimize
```

### 6. Exporting to ONNX
If you need to deploy the trained policy, you can export it to ONNX format.
```bash
make export-onnx
```
You can then test the exported model with:
```bash
make run-client
```

---

## Logging
* Training logs and checkpoints are saved in the `log/` directory.
* Tensorboard logs are saved to `log/tensorboard`. You can view them using:
  ```bash
  make tensorboard
  ```
