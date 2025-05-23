import pytest
import torch # Import torch to check for CUDA availability

# Assuming qgnn_example.py is in the same directory or accessible via PYTHONPATH
from qgnn_example import benchmarkable_cnn_operation 

def test_cnn_operation_cpu_benchmark(benchmark):
    """
    Benchmarks the CNN operation on CPU.
    This includes model initialization, dummy data creation, and a forward pass.
    """
    # benchmarkable_cnn_operation will use its internal default config,
    # which defaults to CPU if CUDA is not specified or available.
    # We explicitly ask for CPU here.
    output = benchmark(lambda: benchmarkable_cnn_operation(device_str_override="cpu"))
    
    assert output is not None
    # Based on default config: batch_size=2, num_classes=2
    assert output.shape == (2, 2) 

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cnn_operation_cuda_benchmark(benchmark):
    """
    Benchmarks the CNN operation on CUDA, if available.
    """
    # Override config to ensure use_cuda is True and device is cuda
    cuda_config_override = {'use_cuda': True}
    output = benchmark(lambda: benchmarkable_cnn_operation(config_override=cuda_config_override, device_str_override="cuda"))
    
    assert output is not None
    assert output.shape == (2, 2)
    assert output.is_cuda # Ensure it ran on CUDA

# You can add more benchmark tests with different configurations:
# def test_cnn_operation_larger_batch_benchmark(benchmark):
#     output = benchmark(lambda: benchmarkable_cnn_operation(device_str_override="cpu", batch_size_override=16))
#     assert output is not None
#     assert output.shape == (16, 2)
