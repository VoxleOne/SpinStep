# CUDA (GPU) Support in SpinStep

SpinStep supports optional GPU acceleration for batch orientation math using [CuPy](https://cupy.dev/).

## Usage

```python
from spinstep.discrete import DiscreteOrientationSet
set_gpu = DiscreteOrientationSet(orientations, use_cuda=True)
```

If CuPy or a compatible GPU is not found, SpinStep will fall back to CPU (NumPy) and print a warning.

## Requirements

- [CuPy](https://cupy.dev/) (`pip install cupy`)
- NVIDIA GPU with CUDA drivers

## Accelerated Features

- Batch orientation storage and math
- Fast angular distance computations for large orientation sets

## Limitations

- Tree traversal logic remains on CPU (Python object graph)
- Only orientation math is GPU-accelerated

## Example

See [`examples/gpu_orientation_matching.py`](../examples/gpu_orientation_matching.py)

---
[⬅️ 06. Discrete Traversal](06-discrete-traversal.md) | [🏠 Home](index.md) | [08. Troubleshooting ➡️](08-troubleshooting.md)
