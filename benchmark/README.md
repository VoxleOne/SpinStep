# 🌀 A Spherical Convolutional Neural Network

Our goal was ambitious yet focused: to construct a Computer Vision **Convolutional Neural Network (CNN)** capable of operating directly on **spherical data**. This required a harmonious blend of:

* **`SpinStep`**, our geometric engine for managing spherical orientations and locations.
* **`PyTorch`**, a flexible deep learning framework for model composition and training.

---

## 🧭 The Geometric Engine: `SpinStep` and `DiscreteOrientationSet`

At the core of our spherical handling lies the `DiscreteOrientationSet` (DOS) class from the `SpinStep` library. This class is responsible for defining and manipulating sets of spherical orientations using quaternions.

### 🧱 Layer Geometry with DOS Factories

We defined layer topologies using factory methods that generate specific spatial arrangements:

* **`from_latitude_band_grid(...)`**
  Creates a set of orientations along a latitude band — ideal for constructing input or intermediate layers.

  ```python
  self.layer0_dos = DiscreteOrientationSet.from_latitude_band_grid(
      center_latitude_deg=0,
      band_width_deg=60,
      num_longitude_points=32,
      num_latitude_points=5,
      use_cuda=use_cuda_flag
  )
  ```

* **`from_local_grid_kernel(...)`**
  Used to create local kernel geometries centered on a reference orientation, helpful for convolution and pooling windows.

  ```python
  self.kernel1_dos = DiscreteOrientationSet.from_local_grid_kernel(
      rows=3,
      cols=3,
      angular_spacing_deg=10,
      use_cuda=use_cuda_flag
  )
  ```

### 🔍 Orientation Matching with `query_within_angle(...)`

This method identifies orientations in the set that lie within a specified angular distance of a query quaternion — critical for convolutional and pooling operations.

```python
matched_indices, matched_distances = self.input_layer_dos.query_within_angle(
    kernel_pt_abs_quat_xp,
    self.angular_threshold_rad,
    return_distances=True
)
```

We optimized this for:

* **CPU**: via `sklearn.neighbors.BallTree` (approximate using rotation vectors).
* **GPU or small sets**: using brute-force, but parallelizable matching.

### ⚙️ Flexibility: `use_cuda` and `xp`

All methods support seamless switching between `NumPy` and `CuPy` backends, managed via the `xp` array module reference.

---

## 🧠 Bridging Geometry and Learning with PyTorch

With the spherical foundation in place, we built custom PyTorch layers to operate on orientation-aware data.

### 🔷 `SphericalConvLayer`

This layer performs convolution using geometric proximity on the sphere:

* Inputs:

  * `input_layer_dos`: input node orientations
  * `output_layer_dos`: output orientations (defines stride/downsampling)
  * `kernel_dos_relative`: the convolution window, defined in relative quaternions
* **Key steps**:

  1. For each output node, compute absolute kernel positions using quaternion multiplication.
  2. Use `query_within_angle` to find matching input nodes.
  3. Gather input activations.
  4. Apply learned kernel weights and bias.

### 🔷 `SphericalPoolLayer`

Similar structure to the convolution layer, but performs **max** or **average pooling** over matching inputs. It leverages the same quaternion-based spatial matching.

---

## 🏗️ The `SphericalCNN` Orchestrator

The `SphericalCNN` class strings everything together:

* Composes `SphericalConvLayer`, activations (e.g., `ReLU`), and `SphericalPoolLayer` modules in sequence.
* Controls the architecture using a **`config` dictionary**:

  * DOS parameters
  * Channel counts
  * Angular thresholds

**Insight**: The degree of downsampling is dictated by the density of the `output_layer_dos`. A sparser output means larger receptive fields and fewer nodes.

---

## 🚀 Example & Insights

We provided a standalone example in `spherical_cnn_example.py` showing:

* A full forward pass with dummy data
* Integration of `SpinStep` geometry and PyTorch logic
* Practical use of quaternion math in deep learning

### Key Takeaways:

* The geometry (DOS) **shapes the model** — its structure, resolution, and field of view.
* Using quaternion-based orientation allows **rotation-aware, globally consistent filters**.
* Custom geometry design is a **creative control lever** in spherical CNN design.

---

## Node vs. Orientation Sets

We made a distinction between:

* `Node`: Represents a single spatial unit.
* `DiscreteOrientationSet`: Represents collections (used for layer-wide operations).

The CNN layers exclusively use `DiscreteOrientationSet` as their interface to geometry.

## Flat vs. Spherical

Okay, let's explain this to someone familiar with standard "flat" CNNs -- like those used for computer vision -- but new to our spherical approach.

Imagine you're building a standard CNN to process a 2D image. Here's how our Spherical CNN components would correspond to the layers and concepts in that flat CNN:

**1. Input Layer (The Image Itself)**

*   **Flat CNN:** You have a 2D grid of pixels (e.g., 256x256 pixels), and each pixel has channel values (e.g., 3 channels for RGB). The structure is an implicit, regular grid.
*   **Spherical CNN:**
    *   Instead of a flat grid of pixels, we have a set of **points defined on the surface of a sphere**. These points are explicitly managed by a `DiscreteOrientationSet` (DOS) object (e.g., `self.input_layer_dos`). This DOS could represent points arranged in our "latitude bands" or any other spherical distribution.
    *   The "pixels values" or features at these spherical points are stored in a PyTorch tensor `x` (e.g., shape `[batch_size, num_input_sphere_points, in_channels]`).
    *   **Analogy:** The `input_layer_dos` defines *where* our "pixels" are on the sphere, and the input tensor `x` provides the "colors" or features for those spherical pixels.

**2. Convolutional Layer (`nn.Conv2d`)**

*   **Flat CNN:** A small, flat filter/kernel (e.g., 3x3) slides across the input image. At each position, it performs an element-wise multiplication with the pixel values it covers and sums them up (a dot product) to produce a single value in the output feature map. It uses learnable weights.
*   **Spherical CNN (`SphericalConvLayer`):**
    *   **Kernel:** We also have a "kernel," but it's a `kernel_dos_relative` – a small set of *relative orientations* (e.g., a 3x3 local "patch" defined by angular offsets). This kernel is also "locally flat" in its conception.
    *   **"Sliding" Operation:**
        1.  The `SphericalConvLayer` has an `output_layer_dos` that defines the "positions" of the output features on the sphere.
        2.  For each point in this `output_layer_dos` (which acts as the center of our kernel's application):
            a.  We take our `kernel_dos_relative` and combine its relative orientations with the current output point's absolute orientation. This gives us the *absolute orientations on the sphere* where our kernel "points" are looking.
            b.  For each of these absolute kernel point orientations, we use `input_layer_dos.query_within_angle(...)` to find the closest corresponding point(s) in the *input layer's* set of spherical points.
            c.  We gather the feature values from the input tensor at these matched input points.
        3.  We then perform a weighted sum of these gathered features, using learnable weights (analogous to the flat CNN's kernel weights).
    *   **Analogy:** The `SphericalConvLayer` also applies a learnable, local kernel. However, "local" is defined by angular proximity on the sphere, and the "sliding" happens by re-orienting the relative kernel at each point of the `output_layer_dos`. The connection between the kernel and the input layer is made dynamically via `query_within_angle`.

**3. Activation Function (`nn.ReLU`, `nn.Sigmoid`, etc.)**

*   **Flat CNN:** Applied element-wise to the output feature map of a convolutional layer.
*   **Spherical CNN:** Exactly the same. We use standard PyTorch activation functions (e.g., `self.relu = nn.ReLU()`) applied element-wise to the tensor produced by `SphericalConvLayer`.
    *   **Analogy:** No real difference in function, just applied to features that are spatially arranged on a sphere.

**4. Pooling Layer (`nn.MaxPool2d`, `nn.AvgPool2d`)**

*   **Flat CNN:** Reduces the size of the feature map by taking the max or average value within a small window (e.g., 2x2) that slides across the feature map.
*   **Spherical CNN (`SphericalPoolLayer`):**
    *   Similar to our `SphericalConvLayer`, it uses an `input_layer_dos`, an `output_layer_dos`, and a `pool_kernel_dos_relative` (which defines the "pooling window" on the sphere).
    *   For each point in the `output_layer_dos`:
        1.  It determines the absolute orientations of its pooling window points.
        2.  It queries the `input_layer_dos` to find matching input points within this window.
        3.  It gathers the features from these matched input points.
        4.  It then performs a max or average operation on these gathered features.
    *   **Analogy:** It summarizes information in a local spherical neighborhood, defined by the `pool_kernel_dos_relative`.

**5. Stride**

*   **Flat CNN:** An integer parameter (e.g., `stride=2`) in convolutional or pooling layers that determines how many pixels the kernel/window shifts at a time, effectively downsampling the output.
*   **Spherical CNN ("Geometry-Defined Stride"):**
    *   This is a key difference and a unique outcome of our design. We don't have an explicit `stride` parameter.
    *   Downsampling is achieved by defining an `output_layer_dos` (for a `SphericalConvLayer` or `SphericalPoolLayer`) that is *sparser* or has fewer points than its `input_layer_dos`.
    *   For example, if `input_layer_dos` has 1000 points representing a dense region, and `output_layer_dos` has only 250 points representing the same (or a transformed) region, the layer naturally downsamples.
    *   **Analogy:** The "stride" is implicitly encoded in how the density and arrangement of points change from one layer's DOS to the next.

**6. Fully Connected Layer (`nn.Linear`)**

*   **Flat CNN:** After several conv/pool layers, the 2D feature maps are often "flattened" into a 1D vector and fed into one or more fully connected layers for classification or regression.
*   **Spherical CNN:**
    *   After our spherical convolution and pooling layers, the output will be a tensor like `[batch_size, channels, num_output_sphere_points]`.
    *   We would similarly "flatten" this (e.g., reshape or pool across the spherical dimension) into a tensor like `[batch_size, total_features]`.
    *   This flattened tensor can then be fed into standard PyTorch `nn.Linear` layers.
    *   **Analogy:** The principle is the same once the spatial (spherical) structure is collapsed.

**In a Nutshell for the Outsider:**

Think of our Spherical CNN as taking the core ideas of a flat CNN (local receptive fields, shared weights, hierarchical feature extraction) and adapting them to work on data that lives on the surface of a sphere.

*   The fixed, regular grid of pixels is replaced by an **explicitly defined set of points on the sphere (`DiscreteOrientationSet`)**.
*   The "neighborhood" for convolutions and pooling is determined by **angular distance on the sphere** rather than fixed array offsets.
*   Downsampling ("stride") is controlled by **changing the density of points** between layers.

The `SphericalConvLayer` and `SphericalPoolLayer` are our custom building blocks that know how to "see" and process features based on this spherical geometry, using the `DiscreteOrientationSet` as their guide.

## 🎯 Wrap-up

This project is a true blend of geometry, machine learning, and low-level implementation. The `SphericalCNN` demonstrates:

* Quaternion-driven spatial logic
* Custom CNN layer design rooted in geometric meaning
* A strong interplay between mathematical theory and PyTorch engineering

> The `DiscreteOrientationSet` is not just a utility — it is the **geometric soul** of the network.

---
| [🏠 Home](index.md) |
