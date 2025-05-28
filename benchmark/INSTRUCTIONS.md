# Benchmarking and Exploring Spherical CNN Operations

This document provides instructions on how to benchmark the Spherical CNN operations implemented in `qgnn_example.py` using `pytest-benchmark`. It also emphasizes understanding the internal workings of the CNN through its diagnostic outputs.

## Prerequisites

1.  **Python Environment**: Ensure you have a Python environment set up.
2.  **Required Libraries**: Install the necessary Python libraries.
    ```bash
    pip install torch pytest pytest-benchmark numpy scipy scikit-learn
    ```
    *   `torch`: For PyTorch functionalities.
    *   `pytest`: The testing framework.
    *   `pytest-benchmark`: The pytest plugin for benchmarking.
    *   `numpy`: For numerical operations.
    *   `scipy`: Used for `Rotation` utilities.
    *   `scikit-learn`: Used for `BallTree` optimization (optional, fallback exists).
3.  **CUDA (Optional)**: If you intend to benchmark on a GPU, ensure you have a CUDA-compatible GPU and that PyTorch is installed with CUDA support. The benchmarks will automatically skip CUDA tests if a compatible GPU is not detected.
    *   You can check if CUDA is available to PyTorch by running:
        ```python
        import torch
        print(torch.cuda.is_available())
        ```

## Running the Benchmarks & Observing CNN Behavior

The benchmark tests are defined in `benchmark/test_qgnn_benchmark.py`. These tests utilize the `benchmarkable_cnn_operation` function from `benchmark/qgnn_example.py`.

The `qgnn_example.py` script includes several `print()` statements designed to provide a "nice view" of the CNN in action. These are intentionally left active to help you understand:
*   Warnings if optional dependencies like CuPy (for GPU array operations) or scikit-learn (for BallTree CPU optimization) are not found.
*   Notifications about internal states, such as the handling of zero quaternions or the creation of layers with empty components (e.g., an empty kernel).
*   Diagnostics during the forward pass of the `SphericalCNN` model, like issues with feature dimensions or empty tensors.

This verbose output is beneficial for learning and debugging.

To run the benchmarks and see these diagnostic messages:

1.  Navigate to the root directory of the `SpinStep` repository (or the directory containing the `benchmark` folder).
2.  Execute `pytest`. The benchmark plugin will automatically discover and run the benchmark tests.

    ```bash
    pytest benchmark/test_qgnn_benchmark.py
    ```
    Or simply, if you are in the root directory:
    ```bash
    pytest
    ```

## Benchmark Tests

The following benchmark tests are performed:

*   `test_cnn_operation_cpu_benchmark`: Benchmarks the `benchmarkable_cnn_operation` on the CPU.
*   `test_cnn_operation_cuda_benchmark`: Benchmarks the `benchmarkable_cnn_operation` on a CUDA-enabled GPU. This test is automatically skipped if CUDA is not available.

## Understanding the Benchmark Output

`pytest-benchmark` will output a table summarizing the benchmark results. This table includes statistics like:

*   **Min**: The minimum time taken for an operation.
*   **Max**: The maximum time taken for an operation.
*   **Mean**: The average time taken.
*   **StdDev**: Standard deviation of the time taken.
*   **Median**: The median time taken.
*   **Iterations**: The number of times the benchmarked function was run.
*   **Rounds**: The number of times the benchmark measurement was repeated.

The primary metric to observe is the time taken for the operations (e.g., mean, median).

## Performance Considerations & Pure Speed Measurement

While the diagnostic `print()` statements in `qgnn_example.py` are excellent for educational purposes, they do introduce some overhead, which means the benchmark times will be slightly higher than the raw computation time of the CNN operations.

If you wish to measure the pure computational performance with minimal overhead (e.g., for fine-grained profiling or comparing absolute speed):

1.  **Optional Step**: You can temporarily comment out or remove the `print()` statements within `benchmark/qgnn_example.py`.
    Key areas for these `print()` statements include:
    *   `get_array_module()`
    *   `DiscreteOrientationSet.__init__()`
    *   `SphericalConvLayer.__init__()`
    *   `SphericalCNN.__init__()` and `SphericalCNN.forward()`
2.  Re-run the benchmarks as described above.

This is an optional step for users interested in the lowest possible latency figures, trading off some of the real-time diagnostic output.

## Customizing Benchmarks

You can customize the benchmarks or add new ones by modifying `benchmark/test_qgnn_benchmark.py`:

*   **Different Configurations**: The `benchmarkable_cnn_operation` function accepts `config_override` and `device_str_override` parameters. You can create new test functions that pass different configurations to benchmark various scenarios (e.g., larger batch sizes, different model parameters, etc.).
    *   Example (commented out in `test_qgnn_benchmark.py`):
        ```python
        # def test_cnn_operation_larger_batch_benchmark(benchmark):
        #     output = benchmark(lambda: benchmarkable_cnn_operation(device_str_override="cpu", batch_size_override=16))
        #     assert output is not None
        #     assert output.shape == (16, 2) # Assuming default num_classes=2
        ```
*   **Device**: You can explicitly test CPU or CUDA by setting the `device_str_override` to `"cpu"` or `"cuda"`.

## Direct Execution of `qgnn_example.py`

For a direct, non-benchmarked run that also shows the diagnostic outputs, you can execute `qgnn_example.py`:

```bash
python benchmark/qgnn_example.py
```
This will run the `benchmarkable_cnn_operation` with default CPU settings and then with CUDA settings (if available), printing shapes and other information as defined in its `if __name__ == "__main__":` block.

## Notes

*   The `benchmarkable_cnn_operation` itself is designed to minimize console output. The main source of verbose output during benchmarks originates from the underlying model classes in `qgnn_example.py`.
*   The `DiscreteOrientationSet` class includes optimizations (like `BallTree` for CPU). If `scikit-learn` is not installed, these optimizations will be skipped (this usually prints a warning, contributing to the educational output).
*   For the most consistent benchmark results (even with print statements), ensure your environment is stable and no other resource-intensive processes are running in the background.