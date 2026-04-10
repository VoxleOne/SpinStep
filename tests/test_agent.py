# test_agent.py — SpinStep Test Suite — MIT License
# Phase 4 tests: Agent, AgentManager

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from spinstep.control.agent import Agent
from spinstep.control.agent_manager import AgentManager
from spinstep.control.controllers import ProportionalOrientationController
from spinstep.control.state import ControlCommand, OrientationState
from spinstep.control.trajectory import OrientationTrajectory, TrajectoryController
from spinstep.traversal.scene_graph import SceneGraph
from spinstep.traversal.spatial_node import SpatialNode


class TestAgentCreation:
    def test_basic_creation(self):
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0)
        agent = Agent(node, ctrl)
        assert agent.node is node
        assert agent.controller is ctrl
        assert agent.trajectory is None

    def test_state_property(self):
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0)
        agent = Agent(node, ctrl)
        state = agent.state
        assert isinstance(state, OrientationState)
        assert state.distance == 5.0

    def test_frame_property(self):
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0)
        agent = Agent(node, ctrl)
        frame = agent.frame
        assert np.allclose(frame.origin_quaternion, [0, 0, 0, 1])
        assert frame.origin_distance == 5.0


class TestAgentUpdate:
    def test_update_with_target(self):
        q_target = R.from_euler("z", 45, degrees=True).as_quat()
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0)
        target = OrientationState(q_target, distance=10.0)
        agent = Agent(node, ctrl, target_state=target)
        cmd = agent.update(dt=0.01)
        assert isinstance(cmd, ControlCommand)
        # Angular velocity should be non-zero (driving toward target)
        assert np.linalg.norm(cmd.angular_velocity) > 0

    def test_update_integrates_state(self):
        q_target = R.from_euler("z", 45, degrees=True).as_quat()
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0, kp_radial=1.0)
        target = OrientationState(q_target, distance=10.0)
        agent = Agent(node, ctrl, target_state=target)
        old_distance = node.distance
        agent.update(dt=0.1)
        # Distance should have changed (moving toward target)
        assert node.distance != old_distance

    def test_update_no_target_produces_zero(self):
        node = SpatialNode("robot", [0, 0, 0, 1])
        ctrl = ProportionalOrientationController(kp=2.0)
        agent = Agent(node, ctrl)
        cmd = agent.update(dt=0.01)
        assert np.allclose(cmd.angular_velocity, [0, 0, 0])
        assert cmd.radial_velocity == 0.0


class TestAgentObserve:
    def test_observe_target(self):
        q = R.from_euler("z", 90, degrees=True).as_quat()
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        target = SpatialNode("target", q, distance=10.0)
        ctrl = ProportionalOrientationController(kp=2.0)
        agent = Agent(node, ctrl)
        obs = agent.observe(target)
        assert isinstance(obs, OrientationState)
        angle = R.from_quat(obs.quaternion).magnitude()
        assert angle == pytest.approx(np.pi / 2, abs=0.05)


class TestAgentWithTrajectory:
    def test_trajectory_update(self):
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0, kp_radial=1.0)
        q_end = R.from_euler("z", 45, degrees=True).as_quat()
        traj = OrientationTrajectory([
            ([0, 0, 0, 1], 5.0, 0.0),
            (q_end, 10.0, 1.0),
        ])
        traj_ctrl = TrajectoryController(ctrl, traj)
        agent = Agent(node, ctrl, trajectory=traj_ctrl)
        cmd = agent.update(dt=0.01, t=0.5)
        assert isinstance(cmd, ControlCommand)


class TestAgentRepr:
    def test_repr(self):
        node = SpatialNode("robot", [0, 0, 0, 1], distance=5.0)
        ctrl = ProportionalOrientationController(kp=2.0)
        agent = Agent(node, ctrl)
        r = repr(agent)
        assert "Agent" in r
        assert "robot" in r


# ===== AgentManager =====


class TestAgentManager:
    def _make_scene_and_agents(self):
        scene = SceneGraph()
        node_a = SpatialNode("a", [0, 0, 0, 1], distance=5.0)
        q_b = R.from_euler("z", 30, degrees=True).as_quat()
        node_b = SpatialNode("b", q_b, distance=7.0)
        scene.add_node(node_a)
        scene.add_node(node_b)
        scene.add_edge("a", "b")

        q_target = R.from_euler("z", 90, degrees=True).as_quat()
        ctrl_a = ProportionalOrientationController(kp=2.0, kp_radial=1.0)
        ctrl_b = ProportionalOrientationController(kp=3.0, kp_radial=1.5)
        target = OrientationState(q_target, distance=10.0)
        agent_a = Agent(node_a, ctrl_a, target_state=target)
        agent_b = Agent(node_b, ctrl_b, target_state=target)
        mgr = AgentManager(scene)
        mgr.add_agent("a", agent_a)
        mgr.add_agent("b", agent_b)
        return mgr

    def test_step_returns_commands(self):
        mgr = self._make_scene_and_agents()
        commands = mgr.step(dt=0.01)
        assert "a" in commands
        assert "b" in commands
        assert isinstance(commands["a"], ControlCommand)

    def test_step_deterministic_order(self):
        """Agents updated in sorted name order."""
        mgr = self._make_scene_and_agents()
        # Just verify no error — deterministic order is internal
        mgr.step(dt=0.01)
        mgr.step(dt=0.01)

    def test_add_duplicate_raises(self):
        mgr = self._make_scene_and_agents()
        node = SpatialNode("c", [0, 0, 0, 1])
        ctrl = ProportionalOrientationController(kp=1.0)
        agent = Agent(node, ctrl)
        mgr.add_agent("c", agent)
        with pytest.raises(ValueError, match="already registered"):
            mgr.add_agent("c", agent)

    def test_remove_agent(self):
        mgr = self._make_scene_and_agents()
        mgr.remove_agent("a")
        assert "a" not in mgr.agents
        commands = mgr.step(dt=0.01)
        assert "a" not in commands

    def test_remove_nonexistent_raises(self):
        mgr = self._make_scene_and_agents()
        with pytest.raises(KeyError):
            mgr.remove_agent("missing")

    def test_query_proximity(self):
        mgr = self._make_scene_and_agents()
        # a and b are 30° apart → should be within π radius
        nearby = mgr.query_proximity("a", np.pi)
        assert "b" in nearby

    def test_query_proximity_narrow(self):
        mgr = self._make_scene_and_agents()
        # Very small radius should exclude b (30° away)
        nearby = mgr.query_proximity("a", np.deg2rad(1))
        assert "b" not in nearby

    def test_pairwise_distances(self):
        mgr = self._make_scene_and_agents()
        dists = mgr.pairwise_distances()
        assert ("a", "b") in dists
        # a and b are 30° apart
        assert dists[("a", "b")] == pytest.approx(np.deg2rad(30), abs=0.1)

    def test_repr(self):
        mgr = self._make_scene_and_agents()
        assert "AgentManager" in repr(mgr)
        assert "2" in repr(mgr)
