# test_api.py — SpinStep Test Suite — MIT License
# Author: SpinStep Contributors
# See LICENSE.txt for full terms.

"""Tests for public API stability, exports, and packaging markers."""

import importlib
import re

import pytest


class TestPackageMetadata:
    """Verify package-level metadata and markers."""

    def test_version_format(self) -> None:
        """Version string follows PEP 440."""
        import spinstep

        assert hasattr(spinstep, "__version__")
        # PEP 440: N.N.NaN, N.N.N, etc.
        assert re.match(
            r"^\d+\.\d+\.\d+(a\d+|b\d+|rc\d+)?$", spinstep.__version__
        ), f"Invalid version format: {spinstep.__version__}"

    def test_py_typed_marker(self) -> None:
        """PEP 561 py.typed marker exists."""
        import pathlib

        import spinstep

        pkg_dir = pathlib.Path(spinstep.__file__).parent
        assert (pkg_dir / "py.typed").exists(), "Missing py.typed marker"


class TestTopLevelExports:
    """Verify spinstep.__all__ contains expected symbols."""

    EXPECTED_EXPORTS = [
        # control
        "OrientationState",
        "ControlCommand",
        "integrate_state",
        "compute_orientation_error",
        "compute_relative_state",
        "ReferenceFrame",
        "rebase_state",
        "Agent",
        "AgentManager",
        "EventEmitter",
        "OrientationController",
        "ProportionalOrientationController",
        "PIDOrientationController",
        "OrientationTrajectory",
        "TrajectoryInterpolator",
        "TrajectoryController",
        # math
        "slerp",
        # traversal
        "Node",
        "SpatialNode",
        "QuaternionDepthIterator",
        "DiscreteOrientationSet",
        "DiscreteQuaternionIterator",
        "SceneGraph",
        "BreadthFirstIterator",
        "GraphQuaternionIterator",
    ]

    def test_all_defined(self) -> None:
        import spinstep

        assert hasattr(spinstep, "__all__")
        assert isinstance(spinstep.__all__, list)

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_export_importable(self, name: str) -> None:
        """Every name in __all__ is importable from the top-level package."""
        import spinstep

        assert name in spinstep.__all__, f"{name} missing from __all__"
        assert hasattr(spinstep, name), f"{name} not accessible on spinstep"


class TestMathSubpackageExports:
    """Verify spinstep.math.__all__ contains expected symbols."""

    EXPECTED_EXPORTS = [
        "quaternion_multiply",
        "quaternion_conjugate",
        "quaternion_normalize",
        "quaternion_inverse",
        "slerp",
        "squad",
        "quaternion_distance",
        "is_within_angle_threshold",
        "forward_vector_from_quaternion",
        "direction_to_quaternion",
        "angle_between_directions",
        "rotate_quaternion",
        "quaternion_from_euler",
        "rotation_matrix_to_quaternion",
        "quaternion_from_rotvec",
        "quaternion_to_rotvec",
        "batch_quaternion_angle",
        "angular_velocity_from_quaternions",
        "get_relative_spin",
        "get_unique_relative_spins",
        "clamp_rotation_angle",
        "NodeProtocol",
        "SpatialNodeProtocol",
    ]

    def test_all_defined(self) -> None:
        from spinstep import math

        assert hasattr(math, "__all__")

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_export_importable(self, name: str) -> None:
        from spinstep import math

        assert name in math.__all__, f"{name} missing from math.__all__"
        assert hasattr(math, name), f"{name} not accessible on spinstep.math"


class TestControlSubpackageExports:
    """Verify spinstep.control.__all__ contains expected symbols."""

    EXPECTED_EXPORTS = [
        "OrientationState",
        "ControlCommand",
        "integrate_state",
        "compute_orientation_error",
        "compute_relative_state",
        "OrientationController",
        "ProportionalOrientationController",
        "PIDOrientationController",
        "OrientationTrajectory",
        "TrajectoryInterpolator",
        "TrajectoryController",
        "ReferenceFrame",
        "rebase_state",
        "Agent",
        "AgentManager",
        "EventEmitter",
    ]

    def test_all_defined(self) -> None:
        from spinstep import control

        assert hasattr(control, "__all__")

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_export_importable(self, name: str) -> None:
        from spinstep import control

        assert name in control.__all__, f"{name} missing from control.__all__"
        assert hasattr(control, name), f"{name} not accessible on spinstep.control"


class TestTraversalSubpackageExports:
    """Verify spinstep.traversal.__all__ contains expected symbols."""

    EXPECTED_EXPORTS = [
        "Node",
        "SpatialNode",
        "QuaternionDepthIterator",
        "DiscreteOrientationSet",
        "DiscreteQuaternionIterator",
        "SceneGraph",
        "BreadthFirstIterator",
        "GraphQuaternionIterator",
    ]

    def test_all_defined(self) -> None:
        from spinstep import traversal

        assert hasattr(traversal, "__all__")

    @pytest.mark.parametrize("name", EXPECTED_EXPORTS)
    def test_export_importable(self, name: str) -> None:
        from spinstep import traversal

        assert name in traversal.__all__, f"{name} missing from traversal.__all__"
        assert hasattr(
            traversal, name
        ), f"{name} not accessible on spinstep.traversal"


class TestUtilsBackwardCompat:
    """Verify that utils/ re-exports still work for backward compatibility."""

    COMPAT_NAMES = [
        "get_array_module",
        "batch_quaternion_angle",
        "quaternion_from_euler",
        "quaternion_distance",
        "rotate_quaternion",
        "is_within_angle_threshold",
        "quaternion_conjugate",
        "quaternion_multiply",
        "rotation_matrix_to_quaternion",
        "get_relative_spin",
        "get_unique_relative_spins",
        "forward_vector_from_quaternion",
        "direction_to_quaternion",
        "angle_between_directions",
    ]

    @pytest.mark.parametrize("name", COMPAT_NAMES)
    def test_utils_reexport(self, name: str) -> None:
        """Functions remain importable from spinstep.utils."""
        from spinstep import utils

        assert hasattr(utils, name), f"{name} not accessible on spinstep.utils"

    def test_utils_quaternion_functions_are_math_functions(self) -> None:
        """Verify that utils re-exports point to the same objects as math."""
        from spinstep import math as sp_math
        from spinstep import utils as sp_utils

        shared = [
            "quaternion_multiply",
            "quaternion_conjugate",
            "quaternion_distance",
            "quaternion_from_euler",
            "rotation_matrix_to_quaternion",
            "rotate_quaternion",
            "is_within_angle_threshold",
            "forward_vector_from_quaternion",
            "direction_to_quaternion",
            "angle_between_directions",
            "batch_quaternion_angle",
            "get_relative_spin",
            "get_unique_relative_spins",
        ]
        for name in shared:
            assert getattr(sp_utils, name) is getattr(sp_math, name), (
                f"utils.{name} is not the same object as math.{name}"
            )


class TestSubpackagesImportable:
    """Verify all subpackages can be imported."""

    @pytest.mark.parametrize(
        "module",
        [
            "spinstep",
            "spinstep.math",
            "spinstep.math.core",
            "spinstep.math.geometry",
            "spinstep.math.conversions",
            "spinstep.math.interpolation",
            "spinstep.math.analysis",
            "spinstep.math.constraints",
            "spinstep.control",
            "spinstep.control.state",
            "spinstep.control.controllers",
            "spinstep.control.trajectory",
            "spinstep.control.frames",
            "spinstep.control.agent",
            "spinstep.control.agent_manager",
            "spinstep.control.events",
            "spinstep.traversal",
            "spinstep.traversal.node",
            "spinstep.traversal.spatial_node",
            "spinstep.traversal.continuous",
            "spinstep.traversal.discrete",
            "spinstep.traversal.discrete_iterator",
            "spinstep.traversal.scene_graph",
            "spinstep.traversal.graph_iterators",
            "spinstep.serialization",
            "spinstep.utils",
            "spinstep.utils.array_backend",
            "spinstep.utils.quaternion_math",
            "spinstep.utils.quaternion_utils",
        ],
    )
    def test_importable(self, module: str) -> None:
        """Every listed module can be imported without error."""
        importlib.import_module(module)


class TestNodeProtocol:
    """Verify NodeProtocol structural typing works correctly."""

    def test_node_satisfies_protocol(self) -> None:
        """spinstep.traversal.Node satisfies NodeProtocol."""
        from spinstep.math.analysis import NodeProtocol
        from spinstep.traversal.node import Node

        node = Node("test", [0, 0, 0, 1])
        assert isinstance(node, NodeProtocol)

    def test_custom_class_satisfies_protocol(self) -> None:
        """Any class with .orientation satisfies NodeProtocol."""
        import numpy as np

        from spinstep.math.analysis import NodeProtocol

        class MyNode:
            def __init__(self) -> None:
                self.orientation = np.array([0.0, 0.0, 0.0, 1.0])

        assert isinstance(MyNode(), NodeProtocol)


class TestNodeAddChild:
    """Verify Node.add_child() convenience method."""

    def test_add_child_appends(self) -> None:
        from spinstep.traversal.node import Node

        root = Node("root", [0, 0, 0, 1])
        child = Node("child", [1, 0, 0, 0])
        result = root.add_child(child)
        assert child in root.children
        assert result is child

    def test_add_child_returns_child(self) -> None:
        from spinstep.traversal.node import Node

        root = Node("root", [0, 0, 0, 1])
        child = root.add_child(Node("child", [0, 1, 0, 0]))
        assert child.name == "child"
        assert len(root.children) == 1
