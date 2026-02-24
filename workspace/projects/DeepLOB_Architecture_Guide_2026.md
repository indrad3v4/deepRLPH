# Implementation Guide: Rolling Window DeepLOB with OFI Alpha

## 1. Project Context & The Simulator Trap
The WunderFund Predictorium evaluates models using a step-by-step simulator. It passes a `DataPoint` object one timestep at a time. The model cannot process 1000 steps at once. 
* **Architecture Strategy:** We will use a Rolling Window approach. The model will be trained on sequences of `window_size = 100`.
* **Input Space:** `(batch, 100, 32)`
* **Output Space:** `(batch, 2)`

## 2. Data Loader: Rolling Windows & OFI
1. **OFI Transformation:** Convert raw bid/ask volumes into Order Flow Imbalance (OFI).
2. **Price Normalization:** Normalize LOB prices relative to the mid-price.
3. **Window Extraction:** The DataLoader must slice the 1000-step sequences into overlapping rolling windows of size 100. The target for each window is the $(t_0, t_1)$ values at the final step of that window.

## 3. Architecture: Pure CNN (No RNN Tapeworms)
To prevent massive ONNX graphs and ensure fast CPU training, DO NOT use LSTMs or Attention. 
1. Use a 1D CNN to extract spatial features from the `(batch, 100, 32)` input.
2. Apply Global Average Pooling (GAP) across the temporal dimension.
3. Pass through a Linear layer to output exactly `(batch, 2)`.

## 4. Inference Logic (solution.py)
`solution.py` MUST maintain an internal buffer to handle the step-by-step simulator.
```python
class PredictionModel:
    def __init__(self):
        self.session = ort.InferenceSession('model.onnx')
        self.buffer = [] # Holds up to 100 steps

    def predict(self, data_point):
        # Extract features from data_point
        # append to self.buffer
        # if len(self.buffer) > 100: self.buffer.pop(0)
        # if data_point.need_prediction: return inference output