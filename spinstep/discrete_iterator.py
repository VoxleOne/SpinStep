# discrete_iterator.py — MIT License
# Author: Eraldo Marques <eraldo.bernardo@gmail.com> — Created: 2025-05-14
# See LICENSE.txt for full terms. This header must be retained in redistributions.

import numpy as np
from scipy.spatial.transform import Rotation as R

class DiscreteQuaternionIterator:
    """
    Traverses a tree/graph using a discrete set of orientation steps.
    Each "step" is a quaternion from the provided orientation set.
    """

    def __init__(self, start_node, orientation_set, angle_threshold=np.pi/8, max_depth=100):
        print(f"[Iterator __init__] max_depth set to: {max_depth}, angle_threshold: {angle_threshold:.4f}") # DEBUG
        self.orientation_set = orientation_set
        self.angle_threshold = angle_threshold
        self.max_depth = max_depth
        # Validate node
        if not hasattr(start_node, "orientation") or not hasattr(start_node, "children"):
            raise AttributeError("Node must have .orientation and .children")
        # Initial state for the start_node is its own orientation
        self.stack = [(start_node, R.from_quat(start_node.orientation), 0)] # Node, its current world orientation, depth
        print(f"[Iterator __init__] Initial stack: {[(n.id, d) for n,s,d in self.stack]}") # DEBUG

    def __iter__(self):
        return self

    def __next__(self):
        print(f"\n[Iterator __next__] Top of __next__ call. Current stack: {[(n.id, d) for n,s,d in self.stack]}") # DEBUG
        while self.stack:
            node, current_node_world_orientation, depth = self.stack.pop()
            print(f"[Iterator __next__] Popped from stack: Node '{node.id}' (depth {depth}). Iterator max_depth: {self.max_depth}") # DEBUG
            print(f"[Iterator __next__]   Node '{node.id}' world orientation: {current_node_world_orientation.as_quat()}") # DEBUG

            if depth >= self.max_depth:
                print(f"[Iterator __next__] !!! Node '{node.id}' (depth {depth}) meets or exceeds max_depth ({self.max_depth}). Skipping this node (due to 'continue'). It will NOT be returned.") # DEBUG
                continue # This skips returning the node if it's AT max_depth or beyond

            # If we are here, depth < self.max_depth. This node will be returned in this iteration.
            # Now, explore its children to potentially add them to the stack.
            # Children will be added with 'depth + 1'.
            print(f"[Iterator __next__] Node '{node.id}' (depth {depth}) is within max_depth. Processing its children to add to stack for future iterations.") # DEBUG
            
            if not hasattr(node, "children"): # Minimal check, assuming orientation exists if it got this far
                print(f"[Iterator __next__] Node {node.id} has no 'children' attribute. Skipping child processing for it.")
                # Still need to return the node itself if it passed the depth check
                print(f"[Iterator __next__] >>> Returning node (processed, no children attribute): '{node.id}'") # DEBUG
                return node


            # The 'current_node_world_orientation' is the world orientation of 'node'.
            # 'step_quat_from_set' is a relative rotation from the orientation_set.
            # 'potential_next_world_orientation' is what the world orientation would be if we take this 'step_quat_from_set' from 'node'.
            for step_quat_from_set in self.orientation_set.orientations:
                potential_next_world_orientation = current_node_world_orientation * R.from_quat(step_quat_from_set)
                print(f"[Iterator __next__]   Node '{node.id}': Trying step from set: {step_quat_from_set}. Potential next world orientation: {potential_next_world_orientation.as_quat()}") # DEBUG

                for child_node_obj in getattr(node, "children", []): # Iterate over actual children of 'node'
                    print(f"[Iterator __next__]     Considering child of '{node.id}': Node '{child_node_obj.id}' (its world_orientation: {child_node_obj.orientation})") # DEBUG
                    
                    child_actual_world_orientation = R.from_quat(child_node_obj.orientation)

                    # Angle calculation: (q1.inv * q2).magnitude() gives theta/2. So, angle = 2 * magnitude.
                    angle_rad = 2 * (potential_next_world_orientation.inv() * child_actual_world_orientation).magnitude()
                    print(f"[Iterator __next__]       Angle between potential_next_orientation and child '{child_node_obj.id}' actual orientation: {angle_rad:.4f} rad (Threshold: {self.angle_threshold:.4f} rad)") # DEBUG

                    if angle_rad < self.angle_threshold:
                        print(f"[Iterator __next__]       >>> Angle condition MET for child '{child_node_obj.id}'.") # DEBUG
                        child_new_depth = depth + 1
                        print(f"[Iterator __next__]       Adding to stack: Node '{child_node_obj.id}' (depth {child_new_depth}), world_orientation: {child_actual_world_orientation.as_quat()}") # DEBUG
                        self.stack.append((child_node_obj, child_actual_world_orientation, child_new_depth))
                    # else:
                        # print(f"[Iterator __next__]       Angle condition NOT MET for child '{child_node_obj.id}'.") # DEBUG
            
            print(f"[Iterator __next__] >>> Returning node (processed, children considered): '{node.id}'") # DEBUG
            return node # Return the current node 'node'

        print("[Iterator __next__] Stack became empty inside 'while self.stack' loop or was empty initially. Raising StopIteration.") # DEBUG
        raise StopIteration
