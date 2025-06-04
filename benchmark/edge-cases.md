# Edge Cases

Iimportant points to consider for the robustness and scalability of the project. Addressing these cases is crucial for understanding the boundaries of the current design and for planning future enhancements.

## 1. Large-Scale Graphs (Very High-Resolution Spherical Data)

When we talk about "large-scale graphs" in this context, I imagine scenarios where the number of nodes in our `DiscreteOrientationSet` objects becomes extremely large (e.g., millions or tens of millions, approaching resolutions used in detailed geophysical models, astronomical surveys, or complex molecular structures).

### Implications for the Current Design:

*   **`DiscreteOrientationSet` Storage & Initialization:**
    *   Storing millions of quaternions (each 4 floats) is generally feasible in memory, especially on GPUs.
    *   However, the *generation* of these DOS objects (e.g., `from_latitude_band_grid`) might become slow if not carefully optimized, though it's a one-time setup cost per DOS.

*   **`query_within_angle` Performance (The Major Bottleneck):**
    *   **CPU (with BallTree):** BallTree construction time scales roughly as `O(N log N)` and query time as `O(log N)` for `N` nodes. However, BallTrees are less effective in higher dimensions (our quaternions are 4D, or rotvecs are 3D). For very large `N`, the constant factors and memory overhead of the tree itself can become problematic. Also, its performance relies on the data having some exploitable structure, which might not always be the case for arbitrary point sets.
    *   **GPU (Brute-Force):** Our current GPU approach calculates distances to all `N` points. While highly parallelizable, the complexity is `O(N)` per query. For millions of kernel points (output nodes * kernel size) each querying millions of input nodes, this can become computationally prohibitive, even on GPU. For example, if an output layer has `M_out` nodes, the kernel has `K` points, and the input layer has `N_in` nodes, one convolution involves roughly `M_out * K * N_in` distance calculations in the brute-force scenario (though only `M_out * K` calls to `query_within_angle` which itself does `N_in` work).
    *   **Nearest Neighbor Search:** We are currently finding the *single closest* match within the angular threshold. If multiple points fall within the threshold, we only use one. For very dense data, this might discard useful information or lead to arbitrary choices if multiple points are almost equidistant.

*   **Memory Footprint in PyTorch Layers:**
    *   The `gathered_activations` tensor in `SphericalConvLayer` (and similar in `SphericalPoolLayer`) has dimensions like `(batch_size, in_channels, num_kernel_points)`. This itself isn't directly dependent on `N_in` or `N_out` (the total number of nodes in input/output DOS), but the process of *filling* it involves indexing into potentially huge input tensors `x` (shape `batch_size, N_in, in_channels`).
    *   The output tensor `output` (shape `batch_size, out_channels, N_out`) will be large if `N_out` is large.

*   **Relevance of "Latitude Bands" and "Local Flat Kernels":**
    *   The "latitude band" strategy is well-suited for data that has an inherent spherical grid-like structure, similar to pixels in an image.
    *   For *arbitrary* large-scale graphs (e.g., social networks, citation graphs, or highly irregular point clouds embedded on a sphere), this specific geometric prior might be less appropriate. The concept of a "latitude band" might not even apply.
    *   "Local flat kernels" assume some degree of local planarity or regularity. On highly irregular graphs, defining a consistent "local patch" this way might be problematic.

### Potential Directions & Considerations for Large-Scale Graphs:

*   **Advanced Spatial Indexing (GPU):** For GPU, explore more sophisticated spatial indexing structures beyond brute-force, such as k-d trees adapted for GPU, or grid-based methods (e.g., hashing grid cells) to quickly cull distant points before brute-force checks.
*   **Sparse Operations:** If the graph is sparse (many nodes, but each only interacts with a few others), sparse tensor representations and operations in PyTorch could be beneficial. This would require a way to define the graph connectivity explicitly rather than relying solely on proximity queries.
*   **Graph Neural Network (GNN) Approaches:** For data that is truly a graph (defined by nodes and explicit edges/connections rather than just point locations), established GNN architectures (e.g., GCN, GraphSAGE, GAT) might be more suitable. These often use message passing over the defined graph structure.
    *   One could imagine a hybrid model: use `SpinStep` to define node orientations, then build an explicit graph (e.g., k-nearest neighbors based on angular distance), and then apply GNN layers.
*   **Hierarchical Methods / Multi-Resolution DOS:**
    *   Process the sphere at multiple resolutions. Start with a coarse DOS and refine. This is common in GIS and some GNN pooling strategies.
    *   Our "geometry-defined stride" already hints at this, but for very large scales, more explicit hierarchical DOS management (e.g., like HEALPix pixelization) might be needed.
*   **Approximate Nearest Neighbor (ANN) Search:** If exact nearest neighbors within the threshold aren't strictly necessary, ANN algorithms (e.g., HNSW, LSH) could offer significant speedups, though they introduce approximation errors.

## 2. Degenerate Quaternions

These are quaternions that can cause numerical instability or have ambiguous interpretations.

*   **Zero Quaternions (`[0,0,0,0]`) or Near-Zero (before normalization):**
    *   **Current Handling:** In `DiscreteOrientationSet.__init__`, we detect these (norm < 1e-9), print a warning, and set them to the identity quaternion `[0,0,0,1]` before normalization.
    *   **Is this sufficient?**
        *   **Pros:** Prevents crashes (division by zero). Allows the pipeline to proceed. Identity is a neutral orientation.
        *   **Cons:** Information is lost. The original data indicated an invalid/unknown orientation, and we're replacing it with a specific, valid one. This might mask upstream data issues.
        *   **Alternative:** Raise an error and force the user to clean their input data. Or, introduce a special "invalid/degenerate" flag for such nodes, and ensure downstream operations can handle this flag (e.g., by skipping these nodes or assigning them zero activation). This adds complexity.
    *   **Impact on `multiply_quaternions_xp`:** If a `[0,0,0,0]` quaternion *were* to bypass the check and enter this function, the result would likely be `[0,0,0,0]`, which might silently propagate issues. Our current fix in `DiscreteOrientationSet` prevents this for DOS orientations.
    *   **Impact on Features:** If a degenerate quaternion is an *input to the network* (e.g., part of the input data that the DOS nodes represent), setting its associated DOS node to identity means its features will be associated with the identity orientation. This might be acceptable or problematic depending on the application.

*   **Double Cover (`q` and `-q` represent the same orientation):**
    *   **Current Handling:** Our angular distance calculation in `query_within_angle` uses `dot_products = self.xp.abs(self.xp.sum(orientations * query_quat_xp_bc, axis=1))`. The `abs()` correctly handles the double cover, ensuring that `q` and `-q` are treated as having zero angular distance from each other (as `abs(q . q) = 1` and `abs(q . -q) = 1`).
    *   **This seems robust.** The `DiscreteOrientationSet` normalizes its quaternions, but it doesn't enforce a canonical choice (e.g., positive scalar part). This is fine because the distance metric handles it.

*   **NaN or Inf in Quaternion Components:**
    *   **Origin:** Could arise from corrupted input data, or from numerical errors in upstream calculations if not handled carefully (e.g., division by a vector that unexpectedly became zero).
    *   **Current Handling:** There's no explicit check for NaN/Inf in `DiscreteOrientationSet` or the PyTorch layers *beyond* what NumPy/CuPy/PyTorch might do by default (which is often to propagate NaNs).
    *   **Impact:** NaNs will poison calculations. `norm` will be NaN, dot products will be NaN, activations will become NaN.
    *   **Recommendation:**
        *   Input validation: Sanitizing input data to check for and handle/reject NaN/Inf quaternions at the earliest stage is crucial.
        *   The `DiscreteOrientationSet` could optionally include checks for NaNs/Infs during initialization and raise errors.
        *   Robust math: Ensure no internal calculations (e.g., divisions) could inadvertently produce NaNs if inputs are otherwise valid (e.g., our handling of zero norm before division).

**Summary of Recommendations for Degenerate Quaternions:**

1.  **Zero/Near-Zero:** The current "set to identity with warning" is a pragmatic choice for allowing execution. For critical applications, consider adding an option to raise an error or a more sophisticated "masking" mechanism for these nodes.
2.  **Double Cover:** The current approach is sound.
3.  **NaN/Inf:** Strongly recommend adding explicit checks for NaN/Inf components in input quaternions passed to `DiscreteOrientationSet` and potentially at other critical data input points, raising errors if found.

Addressing these liminal cases thoroughly often involves trade-offs between robustness, performance, and implementation complexity. For the current stage of the project, acknowledging them and having a plan for how they *could* be handled is a good step. For production-level systems, more rigorous handling would be essential.

---
| [🏠 Home](index.md) |
