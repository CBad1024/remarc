#!/bin/bash
set -e

echo "=============================================="
echo "    Evodm RunPod Initialization Script        "
echo "=============================================="

# 1. Install uv if it is not already installed
if ! command -v uv &> /dev/null
then
    echo "[+] uv package manager not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Ensure it's available in the current shell session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    
    # Also permanently add it to .bashrc for future commands in this bash instance
    if ! grep -q "\.local/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    fi
else
    echo "[+] uv package manager is already installed."
fi

# 2. Install dependencies
echo "[+] Syncing dependencies with uv..."
uv sync

# 3. Check for the DB URL
if [ -z "$OPTUNA_DB_URL" ]; then
    echo "[-] WARNING: OPTUNA_DB_URL environment variable is not set!"
    echo "    The job will default to SQLite, which will crash with n_jobs=32."
    echo "    Press Ctrl+C within 10 seconds to cancel, or wait to continue..."
    sleep 10
else
    echo "[+] OPTUNA_DB_URL is securely set! Using Cloud PostgreSQL."
fi

# 4. Start the optimization job
echo "[+] Starting Optuna optimization across 32 vCPUs..."
make optimize
