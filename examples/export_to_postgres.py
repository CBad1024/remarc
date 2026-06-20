import optuna
import os
from pathlib import Path

def export_study():
    project_root = Path(__file__).resolve().parent.parent
    
    # Source: SQLite Database
    sqlite_path = f"sqlite:///{project_root / 'log' / 'optuna_study.db'}"
    
    # Destination: Cloud PostgreSQL Database
    postgres_url = os.environ.get("OPTUNA_DB_URL")
    if not postgres_url:
        print("Error: OPTUNA_DB_URL environment variable is not set.")
        print("Please run: export OPTUNA_DB_URL='postgresql://...' before running this script.")
        return
        
    study_name = "wf_landscapes_optimization"
    
    print(f"Exporting study '{study_name}'...")
    print(f"From: Local SQLite Database")
    print(f"To: Cloud PostgreSQL Database")
    
    try:
        optuna.copy_study(
            from_study_name=study_name,
            from_storage=sqlite_path,
            to_storage=postgres_url,
            to_study_name=study_name
        )
        print("✅ Export completed successfully! All trials are now in the cloud.")
    except optuna.exceptions.DuplicatedStudyError:
        print(f"❌ Study '{study_name}' already exists in the destination database.")
        print("Optuna prevents overwriting. If you want to merge, load the postgres study and use enqueue_trial, or delete the old study from Postgres first.")

if __name__ == "__main__":
    export_study()
