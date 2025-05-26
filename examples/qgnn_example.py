import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R_scipy # Used in DiscreteOrientationSet

# --- Utility: Array Backend (NumPy/CuPy) ---
def get_array_module(use_cuda=False):
    if use_cuda:
        try:
            import cupy
            return cupy
        except ImportError:
            print("Warning: CuPy not found. Falling back to NumPy.")
            return np
    return np

# --- SpinStep's DiscreteOrientationSet code (incorporating new methods) ---
class DiscreteOrientationSet:
    def __init__(self, orientations, use_cuda=False):
        xp = get_array_module(use_cuda)
        if not isinstance(orientations, xp.ndarray):
            arr = xp.array(orientations, dtype=xp.float64)
        else:
            arr = orientations.astype(xp.float64) # Ensure float64 for precision
            
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("Each orientation must be a quaternion [x, y, z, w]")
        
        norms = xp.linalg.norm(arr, axis=1)
        if xp.any(norms < 1e-9): # Check for zero quaternions before division
            zero_quat_indices = xp.where(norms < 1e-9)[0]
            if isinstance(zero_quat_indices, tuple): # cupy returns tuple
                 zero_quat_indices = zero_quat_indices[0]
            if hasattr(zero_quat_indices, 'get'): # cupy array
                zero_quat_indices_list = zero_quat_indices.get().tolist()
            else: # numpy array
                zero_quat_indices_list = zero_quat_indices.tolist()
            # Set zero quaternions to identity [0,0,0,1] instead of raising error, or handle as needed
            # For now, let's print a warning and set to identity
            print(f"Warning: Zero or near-zero quaternion(s) in orientation set at indices: {zero_quat_indices_list}. Setting to identity.")
            arr[zero_quat_indices_list] = xp.array([0.0, 0.0, 0.0, 1.0])
            norms[zero_quat_indices_list] = 1.0 # Avoid division by zero for these
            # Re-check if any are still problematic (should not be if set to identity)
            if xp.any(norms < 1e-9):
                 raise ValueError(f"Persistent zero quaternions after attempting to fix at indices: {zero_quat_indices_list}")


        arr = arr / norms[:, None]
        
        self.orientations = arr
        self.xp = xp
        self.use_cuda = use_cuda
        self._balltree = None
        if not use_cuda:
            try:
                from sklearn.neighbors import BallTree
                cpu_orientations = arr if not hasattr(arr, 'get') else arr.get()
                if len(arr) > 0 : # Only build BallTree if there are points
                    # Using quaternion directly (4D points) - BallTree handles various metrics
                    # For angular distance, a custom metric or transformation is better.
                    # Using raw quaternions with Euclidean distance is not ideal for angular queries.
                    # Let's use rotvecs as before for an approximation.
                    self.rotvecs = R_scipy.from_quat(cpu_orientations).as_rotvec()
                    if len(arr) > 10: # Heuristic for BallTree overhead
                        self._balltree = BallTree(self.rotvecs, metric='euclidean')
            except ImportError:
                print("Warning: scikit-learn not found. BallTree optimization unavailable.")

    @classmethod
    def from_latitude_band_grid(cls, center_latitude_deg, band_width_deg,
                                num_longitude_points, num_latitude_points,
                                use_cuda=False):
        xp = get_array_module(use_cuda) # xp is numpy for scipy ops
        orientations_list = []
        center_latitude_rad = np.deg2rad(center_latitude_deg)
        band_width_rad = np.deg2rad(band_width_deg)

        if num_latitude_points == 0 or num_longitude_points == 0:
            return cls(xp.empty((0,4), dtype=xp.float64), use_cuda=use_cuda)

        if num_latitude_points == 1:
            latitude_values_rad = [center_latitude_rad]
        else:
            lat_step = band_width_rad / (num_latitude_points - 1)
            min_lat_rad = center_latitude_rad - band_width_rad / 2
            latitude_values_rad = [min_lat_rad + i * lat_step for i in range(num_latitude_points)]
        latitude_values_rad = [np.clip(lat, -np.pi/2 + 1e-6, np.pi/2 - 1e-6) for lat in latitude_values_rad] # Avoid poles for phi calc
        polar_angles_rad = [np.pi/2 - lat for lat in latitude_values_rad]

        lon_step_rad = 2 * np.pi / num_longitude_points
        longitude_values_rad = [i * lon_step_rad for i in range(num_longitude_points)]

        for phi_rad in polar_angles_rad:
            for theta_rad in longitude_values_rad:
                rot = R_scipy.from_euler('ZYX', [theta_rad, phi_rad, 0], degrees=False)
                orientations_list.append(rot.as_quat())
        
        if not orientations_list:
             return cls(xp.empty((0,4), dtype=xp.float64), use_cuda=use_cuda)
        return cls(xp.array(orientations_list, dtype=xp.float64), use_cuda=use_cuda)

    @classmethod
    def from_local_grid_kernel(cls, rows, cols, angular_spacing_deg, use_cuda=False):
        xp = get_array_module(use_cuda) # xp is numpy for scipy ops
        relative_orientations_list = []
        angular_spacing_rad = np.deg2rad(angular_spacing_deg)
        center_row = (rows - 1) / 2.0
        center_col = (cols - 1) / 2.0

        if rows == 0 or cols == 0:
            return cls(xp.empty((0,4), dtype=xp.float64), use_cuda=use_cuda)

        for r_idx in range(rows):
            for c_idx in range(cols):
                y_angular_offset = (r_idx - center_row) * angular_spacing_rad
                x_angular_offset = (c_idx - center_col) * angular_spacing_rad
                rot = R_scipy.from_euler('yx', [x_angular_offset, y_angular_offset], degrees=False)
                relative_orientations_list.append(rot.as_quat())
        
        if not relative_orientations_list:
            return cls(xp.empty((0,4), dtype=xp.float64), use_cuda=use_cuda)
        return cls(xp.array(relative_orientations_list, dtype=xp.float64), use_cuda=use_cuda)

    def query_within_angle(self, quat_xp, angle_rad, return_distances=False):
        # quat_xp is an xp array (1,4) or (4,)
        if quat_xp.ndim == 1:
            query_quat_xp_bc = quat_xp[self.xp.newaxis, :] # Broadcastable (1,4)
        else:
            query_quat_xp_bc = quat_xp

        if not self.use_cuda and self._balltree and len(self.orientations) > 0:
            query_rotvec_np = R_scipy.from_quat(query_quat_xp_bc.get() if hasattr(query_quat_xp_bc,'get') else query_quat_xp_bc).as_rotvec()
            # Note: BallTree query_radius returns a list of arrays
            indices_list = self._balltree.query_radius(query_rotvec_np, r=angle_rad) 
            indices = indices_list[0] # Get array for the first query point
            
            if return_distances: # Recompute precise angular distances for these candidates
                if len(indices) == 0:
                    return self.xp.array([], dtype=int), self.xp.array([], dtype=float)
                selected_quats_xp = self.orientations[indices]
                dot_products = self.xp.abs(self.xp.sum(selected_quats_xp * query_quat_xp_bc, axis=1))
                dot_products = self.xp.clip(dot_products, -1.0, 1.0)
                distances = 2 * self.xp.arccos(dot_products)
                # Filter again by the precise angle_rad, as BallTree was approximate
                final_indices_mask = distances < angle_rad
                return indices[final_indices_mask], distances[final_indices_mask]
            return self.xp.array(indices, dtype=int) # Ensure xp array
        else: # Brute force (GPU or small CPU sets)
            if len(self.orientations) == 0: # No points to query
                 return self.xp.array([], dtype=int), self.xp.array([], dtype=float) if return_distances else self.xp.array([], dtype=int)

            dot_products = self.xp.abs(self.xp.sum(self.orientations * query_quat_xp_bc, axis=1))
            dot_products = self.xp.clip(dot_products, -1.0, 1.0)
            angles = 2 * self.xp.arccos(dot_products)
            
            indices = self.xp.where(angles < angle_rad)[0]
            if isinstance(indices, tuple): indices = indices[0]
            if return_distances:
                return indices, angles[indices]
            return indices

    def as_numpy(self):
        if self.use_cuda: return self.xp.asnumpy(self.orientations)
        return self.orientations
    def __len__(self):
        return self.orientations.shape[0]

# --- Helper: Quaternion Multiplication ---
def multiply_quaternions_xp(q1, q2, xp):
    if q1.ndim == 1: q1 = q1[xp.newaxis, :] 
    w1, x1, y1, z1 = q1[:, 3], q1[:, 0], q1[:, 1], q1[:, 2]
    w2, x2, y2, z2 = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return xp.stack([x, y, z, w], axis=-1)

# --- Spherical CNN Layers ---
class SphericalConvLayer(nn.Module):
    def __init__(self, input_layer_dos: DiscreteOrientationSet, 
                 output_layer_dos: DiscreteOrientationSet,
                 kernel_dos_relative: DiscreteOrientationSet,
                 in_channels: int, out_channels: int, 
                 angular_threshold_rad: float = 0.1):
        super().__init__()
        self.input_layer_dos = input_layer_dos
        self.output_layer_dos = output_layer_dos
        self.kernel_dos_relative = kernel_dos_relative
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.angular_threshold_rad = angular_threshold_rad

        num_kernel_points = len(self.kernel_dos_relative)
        if num_kernel_points == 0: # Handle empty kernel
            print("Warning: SphericalConvLayer created with an empty kernel.")
            # Set weight and bias to allow forward pass to run but produce zeros perhaps
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 0)) 
        else:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, num_kernel_points))
        
        if len(output_layer_dos) == 0:
            print("Warning: SphericalConvLayer created with an empty output_layer_dos.")
            self.bias = nn.Parameter(torch.empty(out_channels, 0))
        else:
            self.bias = nn.Parameter(torch.randn(out_channels, len(self.output_layer_dos)))


        self.output_orientations_xp = self.output_layer_dos.orientations
        self.kernel_relative_quats_xp = self.kernel_dos_relative.orientations
        self.xp = self.input_layer_dos.xp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
        batch_size = x.shape[0]
        num_output_nodes = len(self.output_layer_dos)
        num_kernel_points = len(self.kernel_dos_relative)

        if num_output_nodes == 0: # If output DOS is empty, return empty tensor matching expected channel dim
            return torch.empty(batch_size, 0, self.out_channels, device=x.device, dtype=x.dtype)

        output = torch.zeros(batch_size, self.out_channels, num_output_nodes, device=x.device, dtype=x.dtype)
        
        if num_kernel_points == 0: # If kernel is empty, output remains zeros (after bias)
            if len(self.output_layer_dos) > 0 : # Add bias if output nodes exist
                 output += self.bias.unsqueeze(0)
            return output.permute(0, 2, 1)


        for i_out_node in range(num_output_nodes):
            center_node_abs_quat_xp = self.output_orientations_xp[i_out_node] 
            target_kernel_points_abs_quats_xp = multiply_quaternions_xp(
                center_node_abs_quat_xp, self.kernel_relative_quats_xp, self.xp)
            
            gathered_activations = torch.zeros(batch_size, self.in_channels, num_kernel_points, device=x.device, dtype=x.dtype)

            for k_idx in range(num_kernel_points):
                kernel_pt_abs_quat_xp = target_kernel_points_abs_quats_xp[k_idx]
                matched_indices, matched_distances = self.input_layer_dos.query_within_angle(
                    kernel_pt_abs_quat_xp, self.angular_threshold_rad, return_distances=True)
                
                best_input_node_idx = -1
                if len(matched_indices) > 0:
                    min_dist_idx_in_matched = self.xp.argmin(matched_distances)
                    best_input_node_idx = matched_indices[min_dist_idx_in_matched]
                    if hasattr(best_input_node_idx, 'item'): best_input_node_idx = best_input_node_idx.item()
                    elif not isinstance(best_input_node_idx, (int, np.integer)): best_input_node_idx = int(best_input_node_idx)
                if best_input_node_idx != -1:
                    gathered_activations[:, :, k_idx] = x[:, :, best_input_node_idx]
            
            conv_out = torch.sum(gathered_activations.unsqueeze(1) * self.weight.unsqueeze(0), dim=[-1, -2])
            output[:, :, i_out_node] = conv_out + self.bias[:, i_out_node].unsqueeze(0)
        return output.permute(0, 2, 1)

class SphericalPoolLayer(nn.Module):
    def __init__(self, input_layer_dos: DiscreteOrientationSet,
                 output_layer_dos: DiscreteOrientationSet,
                 pool_kernel_dos_relative: DiscreteOrientationSet,
                 pool_type: str = 'max', angular_threshold_rad: float = 0.1):
        super().__init__()
        if pool_type not in ['max', 'avg']: raise ValueError("pool_type must be 'max' or 'avg'")
        self.input_layer_dos = input_layer_dos
        self.output_layer_dos = output_layer_dos
        self.pool_kernel_dos_relative = pool_kernel_dos_relative
        self.pool_type = pool_type
        self.angular_threshold_rad = angular_threshold_rad
        self.output_orientations_xp = self.output_layer_dos.orientations
        self.kernel_relative_quats_xp = self.pool_kernel_dos_relative.orientations
        self.xp = self.input_layer_dos.xp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        batch_size, num_channels, _ = x.shape
        num_output_nodes = len(self.output_layer_dos)
        num_pool_kernel_points = len(self.pool_kernel_dos_relative)

        if num_output_nodes == 0:
             return torch.empty(batch_size, 0, num_channels, device=x.device, dtype=x.dtype)

        output = torch.zeros(batch_size, num_channels, num_output_nodes, device=x.device, dtype=x.dtype)

        if num_pool_kernel_points == 0: # If pool kernel empty, output remains zero (no operation)
            return output.permute(0,2,1)


        for i_out_node in range(num_output_nodes):
            center_node_abs_quat_xp = self.output_orientations_xp[i_out_node]
            target_pool_points_abs_quats_xp = multiply_quaternions_xp(
                center_node_abs_quat_xp, self.kernel_relative_quats_xp, self.xp)
            
            if self.pool_type == 'max':
                gathered_activations = torch.full((batch_size, num_channels, num_pool_kernel_points), 
                                                  float('-inf'), device=x.device, dtype=x.dtype)
            else: # 'avg'
                gathered_activations = torch.zeros((batch_size, num_channels, num_pool_kernel_points), 
                                                   device=x.device, dtype=x.dtype)
            match_counts = torch.zeros(num_pool_kernel_points, device=x.device, dtype=torch.int) # Simplified for avg

            for k_idx in range(num_pool_kernel_points):
                pool_pt_abs_quat_xp = target_pool_points_abs_quats_xp[k_idx]
                matched_indices, matched_distances = self.input_layer_dos.query_within_angle(
                    pool_pt_abs_quat_xp, self.angular_threshold_rad, return_distances=True)
                best_input_node_idx = -1
                if len(matched_indices) > 0:
                    min_dist_idx_in_matched = self.xp.argmin(matched_distances)
                    best_input_node_idx = matched_indices[min_dist_idx_in_matched]
                    if hasattr(best_input_node_idx, 'item'): best_input_node_idx = best_input_node_idx.item()
                    elif not isinstance(best_input_node_idx, (int, np.integer)): best_input_node_idx = int(best_input_node_idx)
                if best_input_node_idx != -1:
                    gathered_activations[:, :, k_idx] = x[:, :, best_input_node_idx]
                    match_counts[k_idx] = 1
            
            if self.pool_type == 'max':
                pooled_val, _ = torch.max(gathered_activations, dim=-1) 
            else: # 'avg'
                sum_activations = torch.sum(gathered_activations, dim=-1)
                num_valid_matches = torch.sum(match_counts, dim=-1).clamp(min=1).float()
                pooled_val = sum_activations / num_valid_matches
            output[:, :, i_out_node] = pooled_val
        return output.permute(0, 2, 1)

# --- Main SphericalCNN Model ---
class SphericalCNN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        use_cuda_flag = self.config.get('use_cuda', False)
        ang_thresh = np.deg2rad(self.config.get('angular_threshold_deg', 5.0))

        # Layer 0 (Input) DOS
        self.layer0_dos = DiscreteOrientationSet.from_latitude_band_grid(
            center_latitude_deg=config.get('layer0_center_lat', 0), 
            band_width_deg=config.get('layer0_width_deg', 60),
            num_longitude_points=config.get('layer0_lon', 16), 
            num_latitude_points=config.get('layer0_lat', 4),
            use_cuda=use_cuda_flag
        )
        # Kernel for Conv1
        self.kernel1_dos = DiscreteOrientationSet.from_local_grid_kernel(
            rows=config.get('kernel1_rows', 3), cols=config.get('kernel1_cols', 3), 
            angular_spacing_deg=config.get('kernel1_spacing_deg', 10), use_cuda=use_cuda_flag
        )
        # Layer 1 (after Conv1) DOS
        self.layer1_dos_output_conv1 = DiscreteOrientationSet.from_latitude_band_grid(
            center_latitude_deg=config.get('layer1_center_lat', 0), 
            band_width_deg=config.get('layer1_width_deg', 50),
            num_longitude_points=config.get('layer1_lon', 12),
            num_latitude_points=config.get('layer1_lat', 3),
            use_cuda=use_cuda_flag
        )
        self.layers.append(SphericalConvLayer(
            self.layer0_dos, self.layer1_dos_output_conv1, self.kernel1_dos,
            config.get('in_channels', 1), config.get('conv1_out_channels', 4), ang_thresh ))
        self.layers.append(nn.ReLU())

        # Pooling Layer 1
        self.layer2_dos_output_pool1 = DiscreteOrientationSet.from_latitude_band_grid(
            center_latitude_deg=config.get('layer2_center_lat', 0), 
            band_width_deg=config.get('layer2_width_deg', 40),
            num_longitude_points=config.get('layer2_lon', 6),
            num_latitude_points=config.get('layer2_lat', 2),
            use_cuda=use_cuda_flag
        )
        self.pool_kernel1_dos = DiscreteOrientationSet.from_local_grid_kernel(
            rows=config.get('pool_kernel1_rows', 2), cols=config.get('pool_kernel1_cols', 2), 
            angular_spacing_deg=config.get('pool_kernel1_spacing_deg', 12), use_cuda=use_cuda_flag
        )
        self.layers.append(SphericalPoolLayer(
            self.layer1_dos_output_conv1, self.layer2_dos_output_pool1, self.pool_kernel1_dos,
            config.get('pool1_type', 'max'), ang_thresh ))
        
        # Example: Flatten and a Fully Connected layer for classification
        self.num_final_nodes = len(self.layer2_dos_output_pool1)
        self.final_features_dim = self.num_final_nodes * config.get('conv1_out_channels', 4) # Channels from last conv
        
        if self.final_features_dim == 0 and self.num_final_nodes == 0 : # Handle case where DOS leads to zero features
            print("Warning: Final feature dimension is 0. FC layer will be trivial.")
            # Create a dummy FC layer that can handle zero input features if num_classes is also 0 (not typical)
            # Or, more practically, this situation means the DOS config is problematic.
            self.fc = nn.Linear(1, config.get('num_classes', 2)) # Avoid 0-in_feature error if num_classes > 0
        elif self.final_features_dim == 0 and self.num_final_nodes > 0: # Channels became zero somehow (not expected with this setup)
            print("Warning: Final feature dimension is 0 due to zero channels, but nodes exist.")
            self.fc = nn.Linear(1, config.get('num_classes', 2))
        else:
            self.fc = nn.Linear(self.final_features_dim, config.get('num_classes', 2))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != len(self.layer0_dos):
            raise ValueError(f"Input tensor second dim ({x.shape[1]}) != nodes in layer0_dos ({len(self.layer0_dos)})")
        if x.shape[2] != self.config.get('in_channels', 1):
             raise ValueError(f"Input tensor third dim ({x.shape[2]}) != in_channels ({self.config.get('in_channels',1)})")

        for layer in self.layers:
            x = layer(x)
        
        # Check if x became empty due to empty DOS in intermediate layers
        if x.numel() == 0 and self.final_features_dim != 0 :
            print("Warning: Tensor became empty during layer propagation, but expected features for FC. Check DOS configs.")
            # Create a dummy tensor of zeros for FC to prevent error, though this indicates a config issue
            x = torch.zeros(x.shape[0], self.final_features_dim, device=x.device, dtype=x.dtype)
        elif x.numel() == 0 and self.final_features_dim == 0: # Both are zero, FC is likely trivial
             x = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype) # for FC with 1 in_feature
        
        x = x.reshape(x.size(0), -1) # Flatten
        
        # Handle case for FC layer if input features became unexpectedly zero
        if self.fc.in_features == 1 and x.shape[1] == 0 : # FC expects 1, got 0
            x = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        elif x.shape[1] != self.fc.in_features:
            # This can happen if final_features_dim was 0 at init, and x is also 0 from an empty DOS
            if self.fc.in_features == 1 and x.shape[1] == 0: # FC was made trivial, input is empty
                 x = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype) # provide the dummy 1 feature
            else:
                raise ValueError(f"Flattened tensor dim ({x.shape[1]}) != FC in_features ({self.fc.in_features})")

        x = self.fc(x)
        return x
#***********BENCHMARK**************
# ... (all your existing imports like numpy, torch, nn, R_scipy should be at the top)
# ... (all your existing class definitions like get_array_module, DiscreteOrientationSet, 
#      SphericalConvLayer, SphericalPoolLayer, SphericalCNN should be above this point)

def benchmarkable_cnn_operation(config_override=None, device_str_override=None, batch_size_override=None):
    # This function is intended for benchmarking. Minimize prints.
    
    default_cnn_config = {
        'use_cuda': False, # Default to False, can be overridden
        'in_channels': 1, 
        'layer0_lon': 8, 'layer0_lat': 3, 'layer0_width_deg': 50,
        'kernel1_rows': 2, 'kernel1_cols': 2, 'kernel1_spacing_deg': 15, 'conv1_out_channels': 4,
        'layer1_lon': 6, 'layer1_lat': 2, 'layer1_width_deg': 40,
        'pool_kernel1_rows': 2, 'pool_kernel1_cols': 1, 'pool_kernel1_spacing_deg': 18, 'pool1_type': 'max',
        'layer2_lon': 3, 'layer2_lat': 1, 'layer2_width_deg': 30,
        'angular_threshold_deg': 30.0, # The value that worked from our previous debugging
        'num_classes': 2
    }
    cnn_config = default_cnn_config.copy() # Start with defaults
    if config_override:
        cnn_config.update(config_override) # Apply overrides
    
    # Determine device for PyTorch
    if device_str_override:
        device = torch.device(device_str_override)
    else:
        device = torch.device("cuda" if cnn_config['use_cuda'] and torch.cuda.is_available() else "cpu")

    # Ensure use_cuda in config matches the actual device being used, especially if CUDA isn't available
    if device.type == 'cpu':
        cnn_config['use_cuda'] = False
    elif device.type == 'cuda':
        cnn_config['use_cuda'] = True


    model = SphericalCNN(config=cnn_config).to(device) # Assumes SphericalCNN is defined above
    
    batch_size_to_use = batch_size_override if batch_size_override is not None else 2
    num_input_nodes = len(model.layer0_dos)
    in_channels = cnn_config['in_channels']

    if num_input_nodes == 0:
        # For benchmarking, it's better to raise an error than print,
        # as prints might be suppressed or go unnoticed.
        raise ValueError("Input layer (layer0_dos) has 0 nodes. Check CNN configuration for benchmark.")
    
    dummy_input = torch.randn(batch_size_to_use, num_input_nodes, in_channels).to(device)
    
    # The core operation to benchmark: model forward pass
    with torch.no_grad(): # Important for inference benchmarking
        output = model(dummy_input)
    
    return output

# --- Original Example Usage (or modified for benchmarking function) ---
# The if __name__ == "__main__": block should come AFTER the function definition above.
# You can either keep your original __main__ block or update it to use the new function.
# Here's the version that uses the new function:
if __name__ == "__main__":
    print("Spherical CNN Example (Direct Run using benchmarkable_cnn_operation)")
    
    try:
        print("Attempting direct run with default CPU config...")
        output_cpu = benchmarkable_cnn_operation(device_str_override="cpu")
        print("Direct run with CPU successful!")
        print(f"Output tensor shape: {output_cpu.shape}")
        # Ensure output is not empty before trying to access elements
        if output_cpu.numel() > 0:
             print(f"Output values (first batch, first item): {output_cpu[0,0]}")
        else:
             print("Output tensor is empty.")


        if torch.cuda.is_available():
            print("\nAttempting direct run with CUDA config...")
            # For CUDA, ensure 'use_cuda': True is effectively set in the config passed to SphericalCNN
            output_cuda = benchmarkable_cnn_operation(device_str_override="cuda") # This will set cnn_config['use_cuda']=True
            print("Direct run with CUDA successful!")
            print(f"Output tensor shape: {output_cuda.shape}")
            if output_cuda.numel() > 0:
                print(f"Output values (first batch, first item): {output_cuda[0,0]}")
            else:
                print("CUDA Output tensor is empty.")
        else:
            print("\nCUDA not available, skipping CUDA direct run example.")

    except Exception as e:
        print(f"Error during direct run: {e}")
        import traceback
        traceback.print_exc()


#***********END BENCHMARK*******************

# --- Example Usage ---
if __name__ == "__main__":
    print("Spherical CNN Example")
    
    # Configuration for the Spherical CNN
    cnn_config = {
        'use_cuda': False, # Set to True if CuPy and GPU are available
        'in_channels': 1, # e.g., grayscale spherical image data
        
        'layer0_lon': 8, 'layer0_lat': 3, # Input DOS: 8*3=24 nodes
        'layer0_width_deg': 50,

        'kernel1_rows': 2, 'kernel1_cols': 2, 'kernel1_spacing_deg': 15, # Conv kernel
        'conv1_out_channels': 4, # Number of output channels for conv1
        
        'layer1_lon': 6, 'layer1_lat': 2, # Conv1 output DOS: 6*2=12 nodes
        'layer1_width_deg': 40,

        'pool_kernel1_rows': 2, 'pool_kernel1_cols': 1, 'pool_kernel1_spacing_deg': 18, # Pool kernel
        'pool1_type': 'max',
        
        'layer2_lon': 3, 'layer2_lat': 1, # Pool1 output DOS: 3*1=3 nodes
        'layer2_width_deg': 30,

        'angular_threshold_deg': 10.0, # Angular threshold for matching points
        'num_classes': 2 # Example number of output classes
    }

    # Determine device for PyTorch
    device = torch.device("cuda" if cnn_config['use_cuda'] and torch.cuda.is_available() else "cpu")
    if cnn_config['use_cuda'] and device.type == 'cpu':
        print("Warning: Configured use_cuda=True, but torch.cuda is not available. Using CPU.")
        cnn_config['use_cuda'] = False # Ensure DOS uses NumPy if PyTorch is on CPU

    print(f"Using PyTorch device: {device}")
    print(f"DOS objects will use_cuda: {cnn_config['use_cuda']}")

    # Instantiate the Spherical CNN
    model = SphericalCNN(config=cnn_config).to(device)
    # print("\nModel Architecture:")
    # print(model)

    # Create dummy input data
    batch_size = 2
    num_input_nodes = len(model.layer0_dos)
    in_channels = cnn_config['in_channels']
    
    if num_input_nodes == 0:
        print("Error: Input layer (layer0_dos) has 0 nodes. Cannot create dummy input.")
    else:
        dummy_input = torch.randn(batch_size, num_input_nodes, in_channels).to(device)
        print(f"\nCreated dummy input tensor with shape: {dummy_input.shape}")

        # Perform a forward pass
        try:
            print("Performing forward pass...")
            with torch.no_grad(): # No need to track gradients for this example
                output = model(dummy_input)
            print(f"Forward pass successful!")
            print(f"Output tensor shape: {output.shape}")
            print(f"Output values (first batch, first item): {output[0,0] if output.numel() > 0 else 'N/A'}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
