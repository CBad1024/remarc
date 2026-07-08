import os
import sys
import subprocess

def main():
    archs = ["32", "64", "128", "256"]
    
    for arch in archs:
        print(f"Generating dashboard for architecture: {arch}...")
        try:
            subprocess.run(["uv", "run", "examples/plot_dashboard_full.py", "--arch", arch], check=True)
            print(f"Finished architecture: {arch}")
        except subprocess.CalledProcessError as e:
            print(f"Error plotting dashboard for {arch}: {e}")

if __name__ == "__main__":
    main()
