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
        print(f"[Iterator __init__] max_depth set to: {max_depth}, angle_threshold: {angle_threshold:.4f}")
        self.orientation_set = orientation_set
        self.angle_threshold = angle_threshold
        self.max_depth = max_depth
        if not hasattr(start_node, "orientation") or not hasattr(start_node, "children"):
            raise AttributeError("Node must have .orientation and .children")
        self.stack = [(start_node, R.from_quat(start_node.orientation), 0)]
        print(f"[Iterator __init__] Initial stack: {[(n.id, d) for n,s,d in self.stack]}")

    def __iter__(self):
        print("[Iterator __iter__] __iter__ called")
        return self

    def __next__(self):
        print(f"\n[Iterator __next__] Top of __next__ call. Current stack: {[(n.id, d) for n,s,d in self.stack]}")
        if not self.stack:
            print("[Iterator __next__] Stack is initially empty. Raising StopIteration.")
            raise StopIteration

        while self.stack:
            node, current_node_world_orientation, depth = self.stack.pop()
            print(f"[Iterator __next__] Popped from stack: Node '{node.id}' (depth {depth}). Iterator max_depth: {self.max_depth}")
            print(f"[Iterator __next__]   Node '{node.id}' world orientation: {current_node_world_orientation.as_quat()}")

            # CORRECTED DEPTH CHECK:
            # Process node if its depth is less than or equal to max_depth.
            # Skip if depth is strictly greater than max_depth.
            if depth > self.max_depth:
                print(f"[Iterator __next__] !!! Node '{node.id}' (depth {depth}) is strictly GREATER than max_depth ({self.max_depth}). Skipping this node. It will NOT be returned.")
                continue

            # If we are here, depth <= self.max_depth. This node will be returned.
            # Now, explore its children to potentially add them to the stack.
            # Children will be added with 'depth + 1'.
            # Only add children if their new depth (depth + 1) is also <= max_depth.
            print(f"[Iterator __next__] Node '{node.id}' (depth {depth}) is within or at max_depth. Processing its children to add to stack for future iterations.")
            
            child_potential_depth = depth + 1
            if child_potential_depth <= self.max_depth:
                print(f"[Iterator __next__]   Children of '{node.id}' will be at depth {child_potential_depth}. This is <= max_depth ({self.max_depth}). Considering them.")
                if not hasattr(node, "children"):
                    print(f"[Iterator __next__] Node {node.id} has no 'children' attribute.")
                else:
                    for step_quat_from_set in self.orientation_set.orientations:
                        potential_next_world_orientation = current_node_world_orientation * R.from_quat(step_quat_from_set)
                        print(f"[Iterator __next__]   Node '{node.id}': Trying step from set: {step_quat_from_set}. Potential next world orientation: {potential_next_world_orientation.as_quat()}")

                        for child_node_obj in getattr(node, "children", []):
                            print(f"[Iterator __next__]     Considering child of '{node.id}': Node '{child_node_obj.id}' (its world_orientation: {child_node_obj.orientation})")
                            
                            child_actual_world_orientation = R.from_quat(child_node_obj.orientation)
                            
                            # Corrected angle calculation: (q1.inv * q2).magnitude() gives theta/2. So, angle = 2 * magnitude.
                            angle_rad = 2 * (potential_next_world_orientation.inv() * child_actual_world_orientation).magnitude()
                            print(f"[Iterator __next__]       Angle between potential_next_orientation and child '{child_node_obj.id}' actual orientation: {angle_rad:.4f} rad (Threshold: {self.angle_threshold:.4f} rad)")

                            if angle_rad < self.angle_threshold:
                                print(f"[Iterator __next__]       >>> Angle condition MET for child '{child_node_obj.id}'.")
                                print(f"[Iterator __next__]       Adding to stack: Node '{child_node_obj.id}' (depth {child_potential_depth}), world_orientation: {child_actual_world_orientation.as_quat()}")
                                self.stack.append((child_node_obj, child_actual_world_orientation, child_potential_depth))
            else:
                print(f"[Iterator __next__]   Children of '{node.id}' would be at depth {child_potential_depth}. This is > max_depth ({self.max_depth}). No children of '{node.id}' will be added to stack.")
            
            print(f"[Iterator __next__] >>> Returning node (processed, children considered if applicable): '{node.id}' (depth {depth})")
            return node 

        print("[Iterator __next__] Stack became empty. Raising StopIteration.")
        raise StopIteration
