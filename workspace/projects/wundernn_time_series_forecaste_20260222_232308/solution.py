import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import onnxruntime as ort
from utils import DataPoint, ScorerStepByStep

class PredictionModel:
    """
    Optimized Temporal Fusion DeepLOB Solution.
    
    This solution uses the TemporalFusionDeepLOB model trained on 32 features
    to predict t0 and t1. The model expects input shape (1, 1000, 32) and
    outputs shape (1, 901, 2).
    """

    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.sequence_history = []
        
        # STRICT DOCKER PATHING: Use strict relative path as required
        onnx_path = 'model.onnx'
        
        # Initialize ONNX Runtime Session with optimized settings
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = True
        sess_options.enable_mem_pattern = True
        
        self.ort_session = None
        
        try:
            # Load ONNX Model with optimized providers
            self.ort_session = ort.InferenceSession(
                onnx_path, 
                sess_options, 
                providers=['CPUExecutionProvider']
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            print(f"Loaded ONNX model from {onnx_path}")
            print(f"Model input shape: {self.ort_session.get_inputs()[0].shape}")
            print(f"Model output shape: {self.ort_session.get_outputs()[0].shape}")
                
        except Exception as e:
            print(f"Error loading model resources: {e}")
            self.ort_session = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        """
        Predict method optimized for correct input shape mapping.
        
        The model expects input shape (batch_size=1, sequence_length=1000, features=32)
        and outputs shape (batch_size=1, 901, 2).
        
        We need to return the prediction for the current step (last timestep).
        """
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Update history with the current state
        self.sequence_history.append(data_point.state.copy())

        # If prediction not needed yet, return None
        if not data_point.need_prediction:
            return None
            
        if self.ort_session is None:
            # Return zeros if model failed to load
            return np.zeros(2, dtype=np.float32)

        # Prepare input window (last 1000 steps)
        # The model was trained with a context window of exactly 1000
        history_window = self.sequence_history[-1000:]
        
        # Pad with zeros if history is shorter than 1000
        # This should only happen in early steps of a sequence
        if len(history_window) < 1000:
            padding_needed = 1000 - len(history_window)
            padding = [np.zeros_like(history_window[0])] * padding_needed
            history_window = padding + history_window

        # Convert to numpy array with correct dtype
        data_arr = np.asarray(history_window, dtype=np.float32)
        
        # Add batch dimension: (1, 1000, 32)
        # The model expects exactly this shape
        data_tensor = np.expand_dims(data_arr, axis=0)
        
        # Verify the input shape is correct
        if data_tensor.shape != (1, 1000, 32):
            raise ValueError(f"Input tensor shape {data_tensor.shape} != (1, 1000, 32)")
        
        # Run inference
        ort_inputs = {self.input_name: data_tensor}
        output = self.ort_session.run([self.output_name], ort_inputs)[0]
        
        # Verify output shape
        if output.shape != (1, 901, 2):
            raise ValueError(f"Output tensor shape {output.shape} != (1, 901, 2)")
        
        # The model outputs predictions for steps 100...1000 (901 steps total)
        # We need the prediction for the current step (step 1000)
        # Since we always have exactly 1000 steps when need_prediction=True,
        # we take the last prediction from the output sequence
        prediction = output[0, -1, :]  # Shape (2,)
        
        # Ensure float32 precision for competition evaluator
        return prediction.astype(np.float32)


if __name__ == "__main__":
    # Local testing
    test_file = 'data/raw/valid.parquet'
    
    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("Testing Optimized Temporal Fusion DeepLOB (ONNX)...")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("Valid parquet not found for testing.")