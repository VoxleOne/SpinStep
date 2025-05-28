# 🌀 A Spherical Convolutional Neural Network

Our goal was ambitious yet focused: to construct a **Convolutional Neural Network (CNN)** capable of operating directly on **spherical data**. This required a harmonious blend of:

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

## 🧩 Node vs. Orientation Sets

We made a distinction between:

* `Node`: Represents a single spatial unit.
* `DiscreteOrientationSet`: Represents collections (used for layer-wide operations).

The CNN layers exclusively use `DiscreteOrientationSet` as their interface to geometry.

---

## 🎯 Wrap-up

This project was a true blend of geometry, machine learning, and low-level implementation. The `SphericalCNN` demonstrates:

* Quaternion-driven spatial logic
* Custom CNN layer design rooted in geometric meaning
* A strong interplay between mathematical theory and PyTorch engineering

> The `DiscreteOrientationSet` is not just a utility — it is the **geometric soul** of the network.

---
