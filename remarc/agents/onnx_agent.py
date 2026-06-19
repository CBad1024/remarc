import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class ONNXAgent:
    """
    A lightweight, Tianshou-compatible agent that loads and runs inference using ONNX Runtime.
    This class is completely decoupled from PyTorch.
    """
    def __init__(self, model_path: str):
        if ort is None:
            raise ImportError("onnxruntime is not installed. Run `uv pip install onnxruntime`.")
            
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def get_action(self, state: np.ndarray) -> int:
        """
        Runs inference on a single state vector and returns the chosen action index.
        """
        # ONNX expects float32 and a batch dimension: (1, state_dim)
        state_tensor = np.array(state, dtype=np.float32).reshape(1, -1)
        
        ort_inputs = {self.input_name: state_tensor}
        action_logits = self.session.run(None, ort_inputs)[0]
        
        # Greedy action selection
        return int(np.argmax(action_logits[0]))

    def forward(self, batch, state=None, **kwargs):
        """
        Tianshou compatibility layer. Tianshou policies are called with a batch of observations.
        We return a mock object containing the selected actions so it can be dropped into existing
        eval/plot scripts.
        """
        obs = batch.obs if hasattr(batch, "obs") else batch
        
        acts = []
        for single_obs in obs:
            acts.append(self.get_action(single_obs))
            
        # Return an object that has an .act attribute like a Tianshou Batch
        class MockBatch:
            def __init__(self, act):
                self.act = act
                
        return MockBatch(act=np.array(acts))

    def eval(self):
        """Mock method for Tianshou compatibility."""
        return self
