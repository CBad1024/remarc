import os
import shutil
from pathlib import Path

def archive_run(signature):
    if not signature:
        return
    project_root = Path(__file__).resolve().parent.parent
    
    # 1. Delete TensorBoard logs
    tb_dir = project_root / "log" / "tensorboard" / signature
    if tb_dir.exists():
        shutil.rmtree(tb_dir, ignore_errors=True)
        
    # 2. Archive other files
    archive_dir = project_root / "log" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    for folder in ["RL", "plots", "baselines", "metrics", "trajectories", "policies"]:
        src_dir = project_root / "log" / folder
        if not src_dir.exists():
            continue
        for file in src_dir.glob(f"*{signature}*"):
            if file.is_file():
                try:
                    shutil.move(str(file), str(archive_dir / file.name))
                except Exception:
                    pass
