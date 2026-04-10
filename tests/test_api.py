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
        "QuaternionDepthIterator",
        "DiscreteOrientationSet",
        "DiscreteQuaternionIterator",
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
        "OrientationController",
        "ProportionalOrientationController",
        "PIDOrientationController",
        "OrientationTrajectory",
        "TrajectoryInterpolator",
        "TrajectoryController",
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
        "QuaternionDepthIterator",
        "DiscreteOrientationSet",
        "DiscreteQuaternionIterator",
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
            "spinstep.traversal",
            "spinstep.traversal.node",
            "spinstep.traversal.continuous",
            "spinstep.traversal.discrete",
            "spinstep.traversal.discrete_iterator",
            "spinstep.utils",
            "spinstep.utils.array_backend",
            "spinstep.utils.quaternion_math",
            "spinstep.utils.quaternion_utils",
        ],
    )
    def test_importable(self, module: str) -> None:
        """Every listed module can be imported without error."""
        importlib.import_module(module)
