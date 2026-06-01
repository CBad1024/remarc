# REMaRC (Reinforcement-learning based Evolutionary Markovian Resistance Control)

REMaRC is a focused reinforcement learning framework for optimizing drug cycling strategies against evolving populations.
It uses **Wright-Fisher dynamics** with **empirical and theoretical fitness landscapes** (e.g., Chen et al., 2023) to train RL agents that learn adaptive treatment policies.

## Installation

```bash
git clone https://github.com/DavisWeaver/remarc
cd remarc
uv sync
```

## Running the framework

```bash
uv run examples/train.py --wf-train
```

## Interactive Streamlit Interface

```bash
uv run streamlit run examples/dashboard.py
```
