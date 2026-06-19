import argparse
import os
import torch
import torch.nn as nn
import numpy as np

# We'll import the same network classes used in the project
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.optim import Adam
import gymnasium as gym

class PPOActorWrapper(nn.Module):
    """
    Wraps the Tianshou Actor to provide a clean ONNX export interface.
    Tianshou's actor returns (logits, hidden_state). We only want logits.
    """
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, obs):
        # The actor expects (obs, state, info). We pass None for state and {} for info
        logits, _ = self.actor(obs, state=None, info={})
        return logits

def load_policy_for_export(policy_path, state_dim=4, n_actions=4, activation="relu", hidden_sizes=[64, 64], actor_sizes=[32]):
    """
    Reconstructs the PPO policy.
    """
    if activation == "relu":
        act_cls = nn.ReLU
    elif activation == "tanh":
        act_cls = nn.Tanh
    else:
        act_cls = nn.ReLU

    device = torch.device("cpu")

    net = Net(
        state_shape=(state_dim,),
        hidden_sizes=hidden_sizes,
        activation=act_cls,
        device=device,
    )
    actor = Actor(
        preprocess_net=net,
        action_shape=n_actions,
        hidden_sizes=actor_sizes,
        device=device,
    ).to(device)
    critic = Critic(
        preprocess_net=net,
        hidden_sizes=actor_sizes,
        device=device,
    ).to(device)

    optim = Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-4)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        action_space=gym.spaces.Discrete(n_actions),
        discount_factor=0.99,
        deterministic_eval=True,
        action_scaling=False,
    )

    # Load weights
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    policy.eval()
    return policy

def export_and_test(policy_path, onnx_path, state_dim=4, n_actions=4, hidden_sizes=[64, 64], actor_sizes=[32]):
    print(f"Loading PyTorch model from {policy_path}...")
    policy = load_policy_for_export(policy_path, state_dim, n_actions, "relu", hidden_sizes, actor_sizes)
    
    # 1. Wrap the actor
    actor_wrapper = PPOActorWrapper(policy.actor).eval()
    
    # 2. Create dummy input (batch_size=1, state_dim)
    dummy_input = torch.randn(1, state_dim, dtype=torch.float32)
    
    # 3. Export to ONNX
    print(f"Exporting to ONNX at {onnx_path}...")
    torch.onnx.export(
        actor_wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14, # 14 is a very stable modern opset
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_logits'],
        dynamic_axes={'state': {0: 'batch_size'}, 'action_logits': {0: 'batch_size'}}
    )
    print("Export complete!")
    
    # 4. Test Harness: Validate PyTorch vs ONNX
    print("\n--- Running Test Harness ---")
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed. Run `uv pip install onnx onnxruntime` to test the exported model.")
        return

    ort_session = ort.InferenceSession(onnx_path)
    
    num_tests = 1000
    max_diff = 0.0
    
    for i in range(num_tests):
        # Generate random valid state (frequencies sum to 1)
        state_np = np.random.rand(1, state_dim).astype(np.float32)
        state_np = state_np / np.sum(state_np, axis=1, keepdims=True)
        
        # PyTorch Inference
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_np)
            pytorch_out = actor_wrapper(state_tensor).numpy()
            
        # ONNX Inference
        ort_inputs = {ort_session.get_inputs()[0].name: state_np}
        onnx_out = ort_session.run(None, ort_inputs)[0]
        
        # Compare
        diff = np.abs(pytorch_out - onnx_out).max()
        if diff > max_diff:
            max_diff = diff
            
    print(f"Tested {num_tests} random valid states.")
    print(f"Maximum difference between PyTorch and ONNX outputs: {max_diff:.8e}")
    
    if max_diff < 1e-5:
        print("✅ SUCCESS: ONNX model exactly matches PyTorch model!")
    else:
        print("❌ WARNING: ONNX model differs significantly from PyTorch model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to the .pth policy file")
    parser.add_argument("--out", type=str, default="rl_policy.onnx", help="Path for the output .onnx file")
    parser.add_argument("--state-dim", type=int, default=4, help="Dimension of state space")
    parser.add_argument("--n-actions", type=int, default=4, help="Number of actions")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64], help="Base Net hidden layers")
    parser.add_argument("--actor-sizes", type=int, nargs="+", default=[32], help="Actor head hidden layers")
    args = parser.parse_args()
    
    if not os.path.exists(args.policy):
        print(f"Error: Policy file not found at {args.policy}")
        exit(1)
        
    export_and_test(args.policy, args.out, args.state_dim, args.n_actions, args.hidden_sizes, args.actor_sizes)
