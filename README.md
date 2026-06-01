# EvoDM (Wright-Fisher Baseline Edition)

This is a stripped-down, focused version of the Evolutionary Decision Making (evodm) reinforcement learning framework.
It is explicitly narrowed to focus solely on **Wright-Fisher dynamics** using **empirical and theoretical fitness landscapes** (e.g., Chen et al., 2023). 

This repository serves as a clean codebase for publication purposes, excluding legacy SSWM implementations and multi-concentration Seascapes.

## Installation

```bash
git clone https://github.com/DavisWeaver/evodm
cd evodm
uv sync
```

## Running the framework

```bash
uv run examples/run.py --wf-train
```

## Interactive Streamlit Interface

```bash
uv run streamlit run examples/streamlit_app.py
```
