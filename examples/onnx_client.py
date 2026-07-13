import argparse
import numpy as np

# Import the modular agent
from remarc.agents import ONNXAgent


def run_client(model_path, state_vector):
    """
    Demonstrates loading and running inference on an ONNX model using the modular ONNXAgent.
    Notice that this file does NOT import torch, tianshou, or gymnasium!
    """
    print(f"Loading ONNX Model from: {model_path}")

    # 1. Initialize the modular ONNX Agent
    agent = ONNXAgent(model_path)

    # 2. Format the input
    state_array = np.array(state_vector, dtype=np.float32)

    # Validate that frequencies sum to 1.0
    if not np.isclose(np.sum(state_array), 1.0):
        print("Warning: Input genotype frequencies do not sum to 1.0!")

    print(f"Input State (Frequencies): {state_array}")

    # 3. Run Inference using our agent wrapper
    best_drug_index = agent.get_action(state_array)

    # For demonstration, assume 4 standard drugs
    drug_names = ["Drug A", "Drug B", "Drug C", "Drug D"]
    drug_name = (
        drug_names[best_drug_index]
        if best_drug_index < len(drug_names)
        else f"Drug {best_drug_index}"
    )

    print(f"\n--> The ONNX model recommends applying: {drug_name}")
    return best_drug_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lightweight ONNX Client for RL Policy"
    )
    parser.add_argument(
        "--model", type=str, default="rl_policy.onnx", help="Path to the .onnx file"
    )
    parser.add_argument(
        "--state",
        type=float,
        nargs="+",
        default=[0.25, 0.25, 0.25, 0.25],
        help="Space-separated genotype frequencies (e.g. --state 0.1 0.2 0.3 0.4)",
    )
    args = parser.parse_args()

    run_client(args.model, args.state)
