# spinstep/graph.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-06-09
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial import KDTree

from .node import Node 
from .orientations import get_discrete_node_orientation, get_number_of_nodes_at_tier
from .utils.quaternion_utils import (
    get_relative_spin_quaternion, 
    quaternion_angular_distance_scipy,
    quaternion_multiply # For initiate_spin_step
)

class SpinStepGraph:
    def __init__(self, layer_definitions, 
                 num_tangential_neighbors=6, 
                 radial_connection_mode='closest_orientation', 
                 radial_angle_threshold_rad=np.pi/4):
        self.layer_definitions = layer_definitions
        self.num_tangential_neighbors = num_tangential_neighbors
        self.radial_connection_mode = radial_connection_mode
        self.radial_angle_threshold_rad = radial_angle_threshold_rad
        self.nodes_by_id = {}
        self.layers = []
        self._build_graph()

    def _build_graph(self):
        for i, layer_def in enumerate(self.layer_definitions):
            layer_idx = i
            radius = layer_def.get('radius', 1.0)
            resolution_tier = layer_def['resolution_tier']
            current_layer_node_list = []
            num_nodes_on_this_layer = get_number_of_nodes_at_tier(resolution_tier)

            for node_idx_on_layer in range(num_nodes_on_this_layer):
                orientation = get_discrete_node_orientation(resolution_tier, node_idx_on_layer)
                node = Node(layer_idx, node_idx_on_layer, resolution_tier, orientation, radius)
                self.nodes_by_id[node.id] = node
                current_layer_node_list.append(node)
            self.layers.append(current_layer_node_list)

        for layer_idx, current_layer_nodes in enumerate(self.layers):
            if not current_layer_nodes: continue

            if self.num_tangential_neighbors > 0 and len(current_layer_nodes) > 1:
                node_positions = np.array([node.position for node in current_layer_nodes])
                kdtree = KDTree(node_positions)
                
                for i_node_loop, node in enumerate(current_layer_nodes):
                    query_k = min(self.num_tangential_neighbors + 1, len(current_layer_nodes))
                    distances, indices = kdtree.query(node.position, k=query_k)
                    added_tangential_count = 0
                    for k_neighbor_idx in range(len(indices)):
                        neighbor_in_layer_list_idx = indices[k_neighbor_idx]
                        if current_layer_nodes[neighbor_in_layer_list_idx] == node: continue
                        neighbor_node_obj = current_layer_nodes[neighbor_in_layer_list_idx]
                        spin_q = get_relative_spin_quaternion(node.orientation, neighbor_node_obj.orientation)
                        node.add_tangential_neighbor(neighbor_node_obj, spin_q)
                        added_tangential_count +=1
                        if added_tangential_count >= self.num_tangential_neighbors: break
        
        for layer_idx, current_layer_nodes_for_radial in enumerate(self.layers):
            if not current_layer_nodes_for_radial: continue
            for node in current_layer_nodes_for_radial:
                if layer_idx + 1 < len(self.layers):
                    outer_layer_nodes = self.layers[layer_idx + 1]
                    if outer_layer_nodes:
                        if self.radial_connection_mode == 'match_index':
                            # ... (match_index logic as before)
                            outer_layer_def = self.layer_definitions[layer_idx + 1]
                            num_nodes_on_outer_layer = get_number_of_nodes_at_tier(outer_layer_def['resolution_tier'])
                            if node.node_index_on_layer < num_nodes_on_outer_layer:
                                candidate_id = (layer_idx + 1, node.node_index_on_layer)
                                if candidate_id in self.nodes_by_id:
                                    node.set_radial_outward_neighbor(self.nodes_by_id[candidate_id])
                        elif self.radial_connection_mode == 'closest_orientation':
                            best_neighbor = None
                            min_angle = self.radial_angle_threshold_rad 
                            for candidate_neighbor in outer_layer_nodes:
                                angle = quaternion_angular_distance_scipy(node.orientation, candidate_neighbor.orientation)
                                if angle < min_angle:
                                    min_angle = angle
                                    best_neighbor = candidate_neighbor
                            if best_neighbor: node.set_radial_outward_neighbor(best_neighbor)
                
                if layer_idx - 1 >= 0:
                    inner_layer_nodes = self.layers[layer_idx - 1]
                    if inner_layer_nodes:
                        if self.radial_connection_mode == 'match_index':
                            # ... (match_index logic as before)
                            inner_layer_def = self.layer_definitions[layer_idx - 1]
                            num_nodes_on_inner_layer = get_number_of_nodes_at_tier(inner_layer_def['resolution_tier'])
                            if node.node_index_on_layer < num_nodes_on_inner_layer:
                                candidate_id = (layer_idx - 1, node.node_index_on_layer)
                                if candidate_id in self.nodes_by_id:
                                    node.set_radial_inward_neighbor(self.nodes_by_id[candidate_id])
                        elif self.radial_connection_mode == 'closest_orientation':
                            best_neighbor = None
                            min_angle = self.radial_angle_threshold_rad
                            for candidate_neighbor in inner_layer_nodes:
                                angle = quaternion_angular_distance_scipy(node.orientation, candidate_neighbor.orientation)
                                if angle < min_angle:
                                    min_angle = angle
                                    best_neighbor = candidate_neighbor
                            if best_neighbor: node.set_radial_inward_neighbor(best_neighbor)

    def get_node(self, layer_index, node_index_on_layer):
        return self.nodes_by_id.get((layer_index, node_index_on_layer))

    def get_tangential_neighbor_details(self, current_node_id, tangential_neighbor_index):
        current_node = self.get_node(*current_node_id)
        if not current_node: return None
        if not (0 <= tangential_neighbor_index < len(current_node.tangential_neighbors)): return None
        neighbor_node_obj, relative_spin_q = current_node.tangential_neighbors[tangential_neighbor_index]
        return neighbor_node_obj.id, relative_spin_q

    def get_radial_neighbor_details(self, current_node_id, direction):
        current_node = self.get_node(*current_node_id)
        if not current_node: return None
        neighbor_node_obj = None
        if direction == 'outward': neighbor_node_obj = current_node.radial_outward_neighbor
        elif direction == 'inward': neighbor_node_obj = current_node.radial_inward_neighbor
        else: return None
        if not neighbor_node_obj: return None
        relative_spin_q = get_relative_spin_quaternion(current_node.orientation, neighbor_node_obj.orientation)
        return neighbor_node_obj.id, relative_spin_q

    def initiate_spin_step(self, current_node_id, current_absolute_orientation_q, spin_instruction_q):
        current_node = self.get_node(*current_node_id)
        if not current_node: return None

        new_absolute_orientation_q = quaternion_multiply(spin_instruction_q, current_absolute_orientation_q)
        # norm_new_abs = np.linalg.norm(new_absolute_orientation_q) # Already normalized in quaternion_multiply
        # if norm_new_abs > 1e-8: new_absolute_orientation_q /= norm_new_abs
        # else: new_absolute_orientation_q = np.array([0.,0.,0.,1.])

        candidate_nodes_set = {current_node} # Use a set to ensure uniqueness from start
        for neighbor_obj, _ in current_node.tangential_neighbors:
            candidate_nodes_set.add(neighbor_obj)
        if current_node.radial_outward_neighbor:
            candidate_nodes_set.add(current_node.radial_outward_neighbor)
        if current_node.radial_inward_neighbor:
            candidate_nodes_set.add(current_node.radial_inward_neighbor)
        
        candidate_nodes = list(candidate_nodes_set) # Convert to list for iteration

        if not candidate_nodes: # Should not happen
            return current_node.id, new_absolute_orientation_q, spin_instruction_q 

        best_next_node = current_node # Default to current node
        min_angle_to_new_orientation = quaternion_angular_distance_scipy(current_node.orientation, new_absolute_orientation_q)

        for candidate_node_obj in candidate_nodes:
            if candidate_node_obj == current_node: continue # Already calculated for current_node
            angle = quaternion_angular_distance_scipy(candidate_node_obj.orientation, new_absolute_orientation_q)
            if angle < min_angle_to_new_orientation:
                min_angle_to_new_orientation = angle
                best_next_node = candidate_node_obj
        
        return best_next_node.id, new_absolute_orientation_q, spin_instruction_q
