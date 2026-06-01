.PHONY: dashboard

dashboard: ## Launch the Streamlit dashboard
	uv run streamlit run examples/dashboard.py
