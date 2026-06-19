.PHONY: dashboard export-onnx run-client compare-models

# ==========================================
# Configuration Variables
# Override these via CLI (e.g. make export-onnx POLICY=log/RL/other.pth)
# ==========================================
POLICY ?= log/RL/best_policy_fourstate_3e-5LR_1e-4MR_1gps_1000st_128b_50ep_randomstart.pth
HIDDEN_SIZES ?= 256 256 256
ACTOR_SIZES ?= 128
STEPS ?= 500
GEN_PER_STEP ?= 10

dashboard: ## Launch the Streamlit dashboard
	uv run streamlit run examples/dashboard.py

export-onnx: ## Export the RL policy to ONNX format
	uv run python examples/export_model.py --policy $(POLICY) --hidden-sizes $(HIDDEN_SIZES) --actor-sizes $(ACTOR_SIZES)

run-client: ## Run the lightweight ONNX client for inference testing
	uv run python examples/onnx_client.py --state 0.1 0.7 0.1 0.1

compare-models: ## Visually compare Baseline vs ONNX models in simulation
	uv run python examples/compare_models.py --policy $(POLICY) --steps $(STEPS) --gen-per-step $(GEN_PER_STEP)
