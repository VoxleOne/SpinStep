# SpinStep VR Demo Design: "Look & Interact"

> **Status:** Implementation Plan (no code committed — awaiting authorization)
> **Author:** Repository Advisor & Designer
> **Date:** 2026-03-25

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SpinStep Capability Mapping](#2-spinstep-capability-mapping)
3. [Architecture Overview](#3-architecture-overview)
4. [Gap Analysis](#4-gap-analysis)
5. [Proposed New Modules](#5-proposed-new-modules)
6. [Public API Design](#6-public-api-design)
7. [Example Usage](#7-example-usage)
8. [Unity / Unreal Integration](#8-unity--unreal-integration)
9. [Phased Delivery Plan](#9-phased-delivery-plan)
10. [File Map: Old → New](#10-file-map-old--new)
11. [Non-Core Directory Classification](#11-non-core-directory-classification)
12. [Testing Strategy](#12-testing-strategy)
13. [Open Questions](#13-open-questions)

---

## 1. Executive Summary

The **"Look & Interact"** VR demo showcases SpinStep's quaternion-driven orientation
framework in a metaverse-style virtual plaza. The user's head orientation (quaternion)
drives all perception and interaction — no menus, no buttons, just natural looking and
turning.

**Core thesis:** SpinStep's existing angle-threshold traversal primitives
(`query_within_angle`, `is_within_angle_threshold`, `QuaternionDepthIterator`) map
directly to *attention cones* — the fundamental building block of this demo.

### What Exists vs. What's New

| Layer | Exists in SpinStep | New for VR Demo |
|-------|-------------------|-----------------|
| Quaternion math | ✅ Full suite | — |
| Angle-threshold queries | ✅ `query_within_angle`, `is_within_angle_threshold` | — |
| Orientation cone concept | ✅ Implicit in traversal iterators | Explicit `AttentionCone` class |
| Forward vector extraction | ❌ | `forward_vector_from_quaternion()` |
| Scene graph integration | ❌ | `SceneEntity` node subclass |
| Audio attention | ❌ | `AudioCone` specialization |
| NPC perception | ❌ | `NPCAttentionAgent` |
| Multi-head attention | ❌ | `MultiHeadAttention` container |
| VR engine bridge | ❌ | JSON/WebSocket protocol |

---

## 2. SpinStep Capability Mapping

### 2.1 Existing Primitives → VR Concepts

| SpinStep Primitive | VR Demo Usage |
|--------------------|---------------|
| `Node(name, orientation)` | Every scene entity (NPC, object, panel) is a `Node` with an orientation |
| `quaternion_distance(q1, q2)` | Compute angular distance between user gaze and entity direction |
| `is_within_angle_threshold(q_current, q_target, threshold)` | Core "is this entity inside my attention cone?" check |
| `DiscreteOrientationSet.query_within_angle(quat, angle)` | Batch query: "which of N entities are inside my cone?" |
| `rotate_quaternion(q, step)` | Smoothly update user/NPC orientation each frame |
| `quaternion_from_euler(angles)` | Convert VR headset Euler angles to quaternion |
| `batch_quaternion_angle(qs1, qs2, xp)` | Compute pairwise distances between user cone and all entities in one call |
| `DiscreteOrientationSet.from_sphere_grid(n)` | Pre-compute NPC perception directions |
| `QuaternionDepthIterator` | Traverse scene graph to find reachable entities in orientation space |
| `get_relative_spin(nf, nt)` | Compute rotation needed for NPC to face user |

### 2.2 Specific Mappings to Demo Mechanics

#### User Gaze → Attention Cone

```
Existing: is_within_angle_threshold(user_quat, entity_quat, cone_half_angle)
Usage:    For each entity in scene, check if it falls within user's visual cone
```

The user's VR headset provides a quaternion. SpinStep's `is_within_angle_threshold()`
already performs the exact operation needed: "is entity B within angle θ of orientation
A?"

#### NPC Attention → Reverse Cone Check

```
Existing: get_relative_spin(npc_node, user_node) → relative rotation quaternion
          quaternion_distance(npc_quat, user_quat) → angular distance
Usage:    NPC checks if user is within its own perception cone
          If yes → NPC rotates toward user using get_relative_spin()
```

#### Object Highlighting → Batch Query

```
Existing: batch_quaternion_angle(user_forward, all_entity_orientations, np)
Usage:    Single vectorized call returns (1×N) distance matrix
          Objects below threshold get highlighted
```

#### Knowledge Panels → Discrete Orientation Set

```
Existing: DiscreteOrientationSet.from_custom(panel_directions)
          .query_within_angle(user_quat, reading_cone_angle)
Usage:    Panels placed at known orientations; user rotates into their cone to read
```

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   VR Engine (Unity/Unreal)            │
│  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ Headset │  │ Renderer │  │ Spatial Audio Eng │   │
│  └────┬────┘  └─────┬────┘  └────────┬──────────┘   │
│       │              │               │               │
│       │    ┌─────────┴───────────────┘               │
│       │    │  Bridge Layer (JSON/WebSocket/C API)     │
└───────┼────┼─────────────────────────────────────────┘
        │    │
        ▼    ▼
┌──────────────────────────────────────────────────────┐
│              SpinStep VR Module (Python)              │
│                                                      │
│  ┌────────────────┐  ┌──────────────────────┐        │
│  │ AttentionCone  │  │ SceneEntity (Node)   │        │
│  │  - origin_quat │  │  - position (vec3)   │        │
│  │  - half_angle  │  │  - orientation (quat) │       │
│  │  - contains()  │  │  - entity_type        │       │
│  │  - query()     │  │  - metadata           │       │
│  └───────┬────────┘  └──────────┬───────────┘        │
│          │                      │                    │
│  ┌───────┴──────────────────────┴───────────┐        │
│  │          AttentionManager                 │        │
│  │  - update(user_quat) → AttentionResult   │        │
│  │  - register_entity(SceneEntity)          │        │
│  │  - get_attended_entities()               │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ┌─────────────────┐  ┌────────────────────┐         │
│  │ NPCAttention    │  │ MultiHeadAttention │         │
│  │  Agent           │  │  - visual_cone     │         │
│  │  - perception   │  │  - audio_cone      │         │
│  │    _cone        │  │  - haptic_cone     │         │
│  │  - face_toward()│  │  - update()        │         │
│  │  - is_aware_of()│  │  - merge_results() │         │
│  └─────────────────┘  └────────────────────┘         │
│                                                      │
│  ┌──────────────────────────────────────────┐        │
│  │     Core SpinStep (existing, unchanged)   │        │
│  │  Node, quaternion_distance,               │        │
│  │  is_within_angle_threshold,               │        │
│  │  batch_quaternion_angle,                  │        │
│  │  DiscreteOrientationSet, rotate_quaternion │       │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

### Design Principles

1. **No modification to existing SpinStep core** 
2. **SpinStep core MUST be imported made into module**
3. **VR module imports existing primitives**
4. **Pure Python with NumPy** — no VR engine dependency inside SpinStep
5. **Engine-agnostic bridge** — JSON protocol for Unity/Unreal communication
6. **Stateless computation** — each `update()` call is a pure function of current state

---

## 4. Gap Analysis

### 4.1 Missing Utilities (Small additions to `spinstep/utils/`)

| Function | Purpose | Complexity |
|----------|---------|------------|
| `forward_vector_from_quaternion(q)` | Extract the forward (look) direction from a quaternion | Trivial — `R.from_quat(q).apply([0, 0, -1])` |
| `direction_to_quaternion(direction)` | Convert a 3D direction vector to an orientation quaternion | Small — `R.align_vectors()` |
| `angle_between_directions(d1, d2)` | Angular distance between two direction vectors | Trivial — `arccos(dot)` |
| `slerp(q1, q2, t)` | Spherical linear interpolation for smooth NPC turning | Small — `scipy.spatial.transform.Slerp` |

### 4.2 New Module: Attention Cone (`spinstep/attention.py`)

SpinStep's traversal iterators implicitly define orientation cones via angle thresholds.
The VR demo requires making this concept **explicit and reusable** as a first-class
object.

**Key insight:** `AttentionCone` is essentially a wrapper around
`is_within_angle_threshold()` and `batch_quaternion_angle()` with a fixed cone geometry.

### 4.3 New Module: Scene Integration (`spinstep/scene.py`)

The existing `Node` class stores `(name, orientation, children)`. The VR demo needs:
- 3D position (for spatial audio distance attenuation)
- Entity type (NPC, object, panel)
- Activation state (highlighted, active, idle)
- Custom metadata

This should be a **subclass of `Node`**, not a replacement.

### 4.4 New Module: NPC Attention Agent (`spinstep/npc.py`)

NPC behavior is fundamentally:
1. Check if any user is inside my perception cone
2. If yes → compute relative rotation to face them (`get_relative_spin`)
3. Smoothly interpolate my orientation toward the user (`slerp`)

All of this composes existing SpinStep primitives.

### 4.5 New Module: Multi-Head Attention (optional, `spinstep/multihead.py`)

For the advanced twist — multiple independent cones (visual, audio, haptic) with
independent thresholds and different update rates.

---

## 5. Proposed New Modules

### 5.1 `spinstep/attention.py` — Attention Cone

```python
"""Orientation-based attention cone for spatial perception."""

class AttentionCone:
    """A directional attention cone defined by an origin quaternion and half-angle.

    The cone represents a region of orientation space centered on
    a quaternion direction. Entities whose direction falls within
    the half-angle are considered "attended."

    Args:
        origin_quat: Center orientation of the cone [x, y, z, w].
        half_angle: Half-angle of the cone in radians.
        falloff: Optional distance-based attenuation ('linear', 'cosine', None).

    Example::

        from spinstep.attention import AttentionCone
        import numpy as np

        cone = AttentionCone([0, 0, 0, 1], half_angle=0.5)
        cone.update_origin([0, 0, 0.1, 0.995])
        entity_quat = [0.1, 0, 0, 0.995]
        print(cone.contains(entity_quat))   # True/False
        print(cone.attenuation(entity_quat))  # 0.0–1.0 strength
    """

    def __init__(self, origin_quat, half_angle, falloff=None): ...
    def update_origin(self, new_quat): ...
    def contains(self, target_quat) -> bool: ...
    def attenuation(self, target_quat) -> float: ...
    def query_batch(self, entity_quats) -> np.ndarray: ...
    def query_batch_with_attenuation(self, entity_quats) -> np.ndarray: ...
```

**Implementation notes:**
- `contains()` delegates to `is_within_angle_threshold(self.origin, target, self.half_angle)`
- `query_batch()` delegates to `batch_quaternion_angle(origin, entities, np)` then filters
- `attenuation()` returns `1.0 - (distance / half_angle)` clamped to `[0, 1]` for linear falloff

### 5.2 `spinstep/scene.py` — Scene Entity

```python
"""Scene entity with position, orientation, and metadata for VR integration."""

class SceneEntity(Node):
    """A scene-graph node with 3D position and VR-specific metadata.

    Extends Node with position, entity type, and activation state
    needed for VR scene management.

    Args:
        name: Entity identifier.
        orientation: Quaternion [x, y, z, w].
        position: 3D world position [x, y, z].
        entity_type: One of 'npc', 'object', 'panel', 'audio_source'.
        metadata: Arbitrary key-value data for the entity.
    """

    def __init__(self, name, orientation, position, entity_type, metadata=None): ...

    @property
    def direction_quaternion(self) -> np.ndarray: ...

    def distance_to(self, other) -> float: ...


class AttentionManager:
    """Manages attention queries across all scene entities.

    Maintains a registry of SceneEntity instances and efficiently
    queries which entities fall within an attention cone each frame.

    Args:
        entities: Initial list of scene entities.

    Example::

        manager = AttentionManager(entities)
        result = manager.update(user_head_quaternion, cone_half_angle=0.5)
        for entity, strength in result.attended:
            print(f"{entity.name}: {strength:.2f}")
    """

    def __init__(self, entities=None): ...
    def register_entity(self, entity): ...
    def unregister_entity(self, name): ...
    def update(self, user_quat, cone_half_angle) -> AttentionResult: ...

class AttentionResult:
    """Result of an attention query — attended entities with attenuation values."""

    attended: list  # List of (SceneEntity, float) tuples
    unattended: list  # Entities outside the cone
```

### 5.3 `spinstep/npc.py` — NPC Attention Agent

```python
"""NPC behavior driven by orientation-based attention."""

class NPCAttentionAgent:
    """An NPC that perceives and reacts to entities within its attention cone.

    The NPC has its own perception cone. When a user enters the cone,
    the NPC smoothly rotates toward them. When the user leaves, the
    NPC returns to its idle orientation.

    Args:
        entity: The SceneEntity representing this NPC.
        perception_half_angle: Half-angle of the NPC's perception cone (radians).
        turn_speed: Interpolation factor for smooth turning (0–1 per update).
        idle_orientation: Default orientation when no user is attended.

    Example::

        npc = NPCAttentionAgent(npc_entity, perception_half_angle=0.8)
        npc.update(user_positions_and_orientations, dt=0.016)
        new_orientation = npc.entity.orientation  # updated
    """

    def __init__(self, entity, perception_half_angle, turn_speed=0.1,
                 idle_orientation=None): ...
    def is_aware_of(self, target_quat) -> bool: ...
    def update(self, targets, dt) -> None: ...
    def face_toward(self, target_quat, t) -> None: ...
```

**Implementation notes:**
- `is_aware_of()` uses `is_within_angle_threshold(self.orientation, target, self.half_angle)`
- `face_toward()` uses scipy `Slerp` for smooth quaternion interpolation
- `update()` finds closest target in cone, then calls `face_toward()`

### 5.4 `spinstep/multihead.py` — Multi-Head Attention (Advanced)

```python
"""Multi-modal attention with independent cones per modality."""

class MultiHeadAttention:
    """Multiple independent attention cones for different modalities.

    Each 'head' is an AttentionCone with its own half-angle and
    falloff, representing a different sensory channel (visual, audio,
    haptic).

    Args:
        heads: Dict mapping modality names to AttentionCone instances.

    Example::

        multi = MultiHeadAttention({
            'visual': AttentionCone(user_quat, half_angle=0.5),
            'audio':  AttentionCone(user_quat, half_angle=1.2),
            'haptic': AttentionCone(user_quat, half_angle=0.3),
        })
        results = multi.update(user_quat, entities)
        # results['visual'] → entities in visual cone
        # results['audio']  → entities in wider audio cone
    """

    def __init__(self, heads): ...
    def update(self, origin_quat, entities) -> dict: ...
    def merge_results(self, strategy='union') -> list: ...
```

---

## 6. Public API Design

### 6.1 Minimal Public Interface

The VR demo module should expose exactly these classes and functions:

```python
# spinstep/__init__.py additions (when authorized):

__all__ = [
    # Existing (unchanged)
    "Node",
    "QuaternionDepthIterator",
    "DiscreteOrientationSet",
    "DiscreteQuaternionIterator",

    # New VR/attention module
    "AttentionCone",
    "SceneEntity",
    "AttentionManager",
    "NPCAttentionAgent",

    # New utility functions
    "forward_vector_from_quaternion",
    "slerp",
]
```

### 6.2 Import Patterns

```python
# Basic attention check
from spinstep import AttentionCone
cone = AttentionCone(user_quat, half_angle=0.5)

# Full scene management
from spinstep import SceneEntity, AttentionManager
from spinstep.npc import NPCAttentionAgent

# Multi-head (advanced, optional import)
from spinstep.multihead import MultiHeadAttention

# Utility functions
from spinstep.utils import forward_vector_from_quaternion, slerp
```

### 6.3 Design Constraints

- All new classes **compose** existing SpinStep primitives (no reimplementation)
- No VR engine imports inside SpinStep (engine-agnostic)
- All orientations use the existing `[x, y, z, w]` quaternion convention
- All angles in radians (matching existing API)
- No side effects at import time (matching existing convention)
- Type hints on all public methods (matching existing convention)
- Google-style docstrings (matching existing convention)

---

## 7. Example Usage

### 7.1 Basic: User Looks at an Object

```python
import numpy as np
from spinstep import AttentionCone
from spinstep.utils import quaternion_from_euler

# User is looking slightly to the right (30° yaw)
user_quat = quaternion_from_euler([30, 0, 0], order='yxz', degrees=True)

# Attention cone: 45° half-angle (90° total field of view)
cone = AttentionCone(user_quat, half_angle=np.radians(45))

# Fountain is at 25° to the right of center
fountain_quat = quaternion_from_euler([25, 5, 0], order='yxz', degrees=True)

print(cone.contains(fountain_quat))       # True — fountain is in view
print(cone.attenuation(fountain_quat))    # 0.89 — close to center of cone
```

### 7.2 Scene: Full Plaza with Multiple Entities

```python
from spinstep import SceneEntity, AttentionManager, AttentionCone
from spinstep.utils import quaternion_from_euler
import numpy as np

# Build scene
fountain = SceneEntity(
    name="fountain",
    orientation=quaternion_from_euler([0, 0, 0]),
    position=[5.0, 0.0, 3.0],
    entity_type="object",
)
npc_vendor = SceneEntity(
    name="vendor",
    orientation=quaternion_from_euler([90, 0, 0]),
    position=[-3.0, 0.0, 2.0],
    entity_type="npc",
)
art_panel = SceneEntity(
    name="vr_art_panel",
    orientation=quaternion_from_euler([180, 0, 0]),
    position=[0.0, 2.0, -4.0],
    entity_type="panel",
    metadata={"content": "VR Art: A New Medium"},
)

# Create attention manager
manager = AttentionManager([fountain, npc_vendor, art_panel])

# Simulate user looking toward the fountain
user_quat = quaternion_from_euler([5, -2, 0], order='yxz', degrees=True)
result = manager.update(user_quat, cone_half_angle=np.radians(45))

for entity, strength in result.attended:
    print(f"  {entity.name}: attention={strength:.2f}")
    # fountain: attention=0.94
```

### 7.3 NPC: Vendor Notices the User

```python
from spinstep.npc import NPCAttentionAgent
from spinstep.utils import quaternion_from_euler
import numpy as np

# NPC vendor with 80° perception cone
npc_agent = NPCAttentionAgent(
    entity=npc_vendor,
    perception_half_angle=np.radians(40),
    turn_speed=0.15,
)

# User walks into the vendor's perception cone
user_quat = quaternion_from_euler([85, 0, 0], order='yxz', degrees=True)

if npc_agent.is_aware_of(user_quat):
    print("Vendor notices the user!")
    # Over several frames, vendor smoothly turns toward user
    for frame in range(10):
        npc_agent.update(targets=[user_quat], dt=1/60)
    print(f"Vendor now facing: {npc_vendor.orientation}")
```

### 7.4 Multi-Head: Visual + Audio + Haptic Cones

```python
from spinstep import AttentionCone
from spinstep.multihead import MultiHeadAttention
from spinstep.utils import quaternion_from_euler
import numpy as np

user_quat = quaternion_from_euler([30, 0, 0], order='yxz', degrees=True)

multi = MultiHeadAttention({
    'visual': AttentionCone(user_quat, half_angle=np.radians(45)),
    'audio':  AttentionCone(user_quat, half_angle=np.radians(90)),  # wider
    'haptic': AttentionCone(user_quat, half_angle=np.radians(20)),  # narrower
})

entities = [fountain, npc_vendor, art_panel]
results = multi.update(user_quat, entities)

print("Visual:", [e.name for e, _ in results['visual']])
print("Audio:",  [e.name for e, _ in results['audio']])   # may include more
print("Haptic:", [e.name for e, _ in results['haptic']])   # only very close

# Merge: entities attended by ANY modality
all_attended = multi.merge_results(strategy='union')
```

### 7.5 Audio Focus: Spatial Audio Gain Based on Cone

```python
from spinstep import AttentionCone
import numpy as np

# Audio cone — wider than visual
audio_cone = AttentionCone(user_quat, half_angle=np.radians(90), falloff='cosine')

# Compute per-source audio gain
audio_sources = [fountain, npc_vendor]  # both have spatial audio
for source in audio_sources:
    gain = audio_cone.attenuation(source.orientation)
    # gain: 0.0 (outside cone) to 1.0 (dead center)
    print(f"{source.name} audio gain: {gain:.2f}")
    # → Pass gain to spatial audio engine
```

---

## 8. Unity / Unreal Integration

### 8.1 Integration Architecture

SpinStep runs as a **Python service** that communicates with the VR engine via a
lightweight bridge. Two integration patterns are supported:

#### Option A: WebSocket Bridge (Recommended for Prototyping)

```
Unity/Unreal  ←→  WebSocket  ←→  Python (SpinStep + FastAPI/asyncio)
```

- VR engine sends headset quaternion + entity states as JSON every frame
- Python computes attention results, NPC updates
- Results sent back as JSON (highlighted entities, NPC rotations, audio gains)
- Latency: ~2–5ms on localhost

#### Option B: C Extension / Native Plugin (Production)

```
Unity/Unreal  ←→  C API (pybind11 or ctypes)  ←→  SpinStep
```

- SpinStep compiled as a shared library via pybind11
- Direct function calls from C# (Unity) or C++ (Unreal)
- Latency: <0.1ms

### 8.2 WebSocket Message Protocol

#### Client → Server (VR Engine → SpinStep)

```json
{
  "type": "frame_update",
  "timestamp": 1616700000.123,
  "user": {
    "head_quaternion": [0.0, 0.1, 0.0, 0.995],
    "position": [1.0, 1.7, 3.0]
  },
  "entities": [
    {
      "id": "fountain",
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position": [5.0, 0.0, 3.0],
      "type": "object"
    }
  ]
}
```

#### Server → Client (SpinStep → VR Engine)

```json
{
  "type": "attention_result",
  "timestamp": 1616700000.125,
  "attended_entities": [
    {"id": "fountain", "attention_strength": 0.89, "highlight": true}
  ],
  "npc_updates": [
    {"id": "vendor", "new_orientation": [0.05, 0.1, 0.0, 0.994], "state": "aware"}
  ],
  "audio_gains": [
    {"id": "fountain_audio", "gain": 0.72}
  ]
}
```

### 8.3 Unity C# Client (Sketch)

```csharp
// Unity MonoBehaviour — sends head orientation to SpinStep each frame
void Update() {
    Quaternion headQuat = Camera.main.transform.rotation;
    // Convert Unity's (x,y,z,w) to SpinStep's [x,y,z,w] (same order)
    string msg = JsonUtility.ToJson(new FrameUpdate {
        head_quaternion = new float[] {
            headQuat.x, headQuat.y, headQuat.z, headQuat.w
        }
    });
    websocket.Send(msg);
}

// Handle response
void OnAttentionResult(AttentionResult result) {
    foreach (var entity in result.attended_entities) {
        GameObject obj = scene.Find(entity.id);
        obj.GetComponent<Renderer>().material.SetFloat("_GlowStrength",
            entity.attention_strength);
    }
}
```

---

## 9. Phased Delivery Plan

### Phase 1: Core Attention Module (Week 1–2)

**Goal:** `AttentionCone` and utility functions — pure SpinStep additions.

| Task | File | LOC Est. | Dependencies |
|------|------|----------|-------------|
| `forward_vector_from_quaternion()` | `spinstep/utils/quaternion_utils.py` | ~10 | scipy |
| `slerp()` | `spinstep/utils/quaternion_utils.py` | ~15 | scipy |
| `direction_to_quaternion()` | `spinstep/utils/quaternion_utils.py` | ~10 | scipy |
| `AttentionCone` class | `spinstep/attention.py` | ~80 | numpy, existing utils |
| Tests for above | `tests/test_attention.py` | ~120 | pytest |
| Update `__init__.py` exports | `spinstep/__init__.py` | ~5 | — |
| Update `utils/__init__.py` exports | `spinstep/utils/__init__.py` | ~5 | — |

**Deliverable:** `AttentionCone` usable standalone — no VR engine needed.

### Phase 2: Scene & NPC Layer (Week 2–3)

**Goal:** `SceneEntity`, `AttentionManager`, `NPCAttentionAgent`.

| Task | File | LOC Est. | Dependencies |
|------|------|----------|-------------|
| `SceneEntity` class | `spinstep/scene.py` | ~60 | Node |
| `AttentionResult` dataclass | `spinstep/scene.py` | ~20 | — |
| `AttentionManager` class | `spinstep/scene.py` | ~80 | AttentionCone |
| `NPCAttentionAgent` class | `spinstep/npc.py` | ~90 | slerp, AttentionCone |
| Tests for scene module | `tests/test_scene.py` | ~100 | pytest |
| Tests for NPC module | `tests/test_npc.py` | ~80 | pytest |

**Deliverable:** Full scene graph with NPC attention — testable in pure Python.

### Phase 3: Multi-Head & Audio (Week 3–4)

**Goal:** `MultiHeadAttention` and audio gain computation.

| Task | File | LOC Est. | Dependencies |
|------|------|----------|-------------|
| `MultiHeadAttention` class | `spinstep/multihead.py` | ~70 | AttentionCone |
| Audio gain helper | `spinstep/attention.py` | ~20 | AttentionCone |
| Tests for multi-head | `tests/test_multihead.py` | ~80 | pytest |

**Deliverable:** Multi-modal attention queries — visual, audio, haptic.

### Phase 4: VR Engine Bridge (Week 4–5)

**Goal:** WebSocket server for Unity/Unreal integration.

| Task | File | LOC Est. | Dependencies |
|------|------|----------|-------------|
| WebSocket server | `examples/vr_bridge_server.py` | ~120 | asyncio, websockets |
| JSON protocol handlers | `examples/vr_bridge_server.py` | ~60 | json |
| Unity client script | `examples/unity/SpinStepClient.cs` | ~80 | Unity WebSocket |
| Demo scene description | `examples/vr_plaza_demo.py` | ~100 | spinstep |

**Deliverable:** Working prototype — Unity sends head quaternion, SpinStep responds
with attention results.

### Phase 5: Polish & Documentation (Week 5–6)

| Task | File |
|------|------|
| VR demo documentation | `docs/10-vr-demo.md` |
| API reference updates | `docs/09-api-reference.md` |
| Demo video script / README | `examples/vr_bridge_server_README.md` |
| Performance benchmarks | `benchmark/vr_attention_benchmark.py` |

---

## 10. File Map: Old → New

No existing files are modified in the core library (except `__init__.py` and
`utils/__init__.py` to add exports). All new functionality lives in new files.

| Action | File | Description |
|--------|------|-------------|
| **NEW** | `spinstep/attention.py` | `AttentionCone` class |
| **NEW** | `spinstep/scene.py` | `SceneEntity`, `AttentionManager`, `AttentionResult` |
| **NEW** | `spinstep/npc.py` | `NPCAttentionAgent` |
| **NEW** | `spinstep/multihead.py` | `MultiHeadAttention` |
| **EDIT** | `spinstep/__init__.py` | Add new exports to `__all__` |
| **EDIT** | `spinstep/utils/__init__.py` | Export `forward_vector_from_quaternion`, `slerp` |
| **EDIT** | `spinstep/utils/quaternion_utils.py` | Add `forward_vector_from_quaternion`, `slerp`, `direction_to_quaternion` |
| **NEW** | `tests/test_attention.py` | Tests for `AttentionCone` |
| **NEW** | `tests/test_scene.py` | Tests for `SceneEntity`, `AttentionManager` |
| **NEW** | `tests/test_npc.py` | Tests for `NPCAttentionAgent` |
| **NEW** | `tests/test_multihead.py` | Tests for `MultiHeadAttention` |
| **NEW** | `examples/vr_bridge_server.py` | WebSocket bridge (not in package) |
| **NEW** | `examples/vr_plaza_demo.py` | Standalone demo script |
| **NEW** | `docs/10-vr-demo.md` | VR demo documentation |

---

## 11. Non-Core Directory Classification

| Directory | Role | In Distributed Package? | Rationale |
|-----------|------|------------------------|-----------|
| `spinstep/` | Core library + VR modules | ✅ Yes | Library code |
| `tests/` | Unit tests | ❌ No (sdist only) | Not needed by users |
| `docs/` | Documentation | ❌ No | Hosted separately |
| `examples/` | Usage examples, VR bridge | ❌ No | Reference only |
| `demos/` | Educational scripts | ❌ No | Learning aids |
| `benchmark/` | Performance benchmarks | ❌ No | Dev-only |

The VR bridge server (`examples/vr_bridge_server.py`) is intentionally placed in
`examples/` — it is a reference implementation, not part of the distributed library.

---

## 12. Testing Strategy

### 12.1 Unit Tests (Pure Python, No VR Engine)

All new modules are testable without a VR engine:

```python
# test_attention.py
def test_cone_contains_center():
    """Entity at cone center should always be contained."""
    cone = AttentionCone([0, 0, 0, 1], half_angle=0.5)
    assert cone.contains([0, 0, 0, 1]) is True

def test_cone_excludes_outside():
    """Entity outside cone should not be contained."""
    cone = AttentionCone([0, 0, 0, 1], half_angle=0.1)
    far_quat = quaternion_from_euler([90, 0, 0])
    assert cone.contains(far_quat) is False

def test_attenuation_center_is_max():
    """Attenuation at center should be 1.0."""
    cone = AttentionCone([0, 0, 0, 1], half_angle=0.5)
    assert cone.attenuation([0, 0, 0, 1]) == pytest.approx(1.0)

def test_attenuation_edge_is_zero():
    """Attenuation at exact cone boundary should be ~0.0."""
    ...

def test_batch_query_returns_correct_indices():
    """Batch query should return indices of attended entities only."""
    ...
```

### 12.2 Integration Tests (Scene + NPC)

```python
# test_scene.py
def test_attention_manager_finds_entities_in_cone():
    """Manager should return entities within user's attention cone."""
    ...

def test_npc_turns_toward_user():
    """NPC should smoothly rotate toward a user in its perception cone."""
    ...

def test_npc_ignores_user_outside_cone():
    """NPC should not react to users outside its perception cone."""
    ...
```

### 12.3 Compatibility

- All existing 62 tests must continue to pass (zero regressions)
- New tests follow existing pytest patterns
- No new external dependencies (all new code uses numpy + scipy, already required)

---

## 13. Open Questions

1. **Quaternion convention alignment with Unity/Unreal:**
   Unity uses `(x, y, z, w)` — same as SpinStep ✅.
   Unreal uses `(x, y, z, w)` — same as SpinStep ✅.
   However, coordinate system handedness differs (Unity: left-handed, Unreal: left-handed, SpinStep/scipy: right-handed). The bridge layer must handle this conversion.

2. **Frame rate for attention updates:**
   Should the Python service run at VR frame rate (90 Hz) or at a lower rate (30 Hz)?
   Recommendation: 30 Hz with interpolation on the engine side.

3. **Cone shape:**
   Current implementation uses angular distance (spherical cap).
   Should we support elliptical cones (different horizontal/vertical FOV)?
   Recommendation: Start with spherical, add elliptical in a future iteration.

4. **GPU acceleration for attention queries:**
   SpinStep already supports CuPy for `DiscreteOrientationSet`. Should `AttentionCone`
   also support GPU? Only useful if scene has >1000 entities.
   Recommendation: CPU-only for Phase 1, GPU option for Phase 5.

5. **NPC AI training integration:**
   The problem statement mentions Unity ML-Agents for NPC training. This would require
   a gym-compatible environment wrapper around the attention system.
   Recommendation: Phase 5+ (post-demo).

---

## Summary

The "Look & Interact" VR demo is **architecturally straightforward** because SpinStep
already has the core mathematical primitives:

- **Attention cone** = `is_within_angle_threshold()` wrapped in a class
- **Batch attention** = `batch_quaternion_angle()` with threshold filtering
- **NPC perception** = reverse `is_within_angle_threshold()` + `get_relative_spin()`
- **Smooth turning** = `slerp()` (new, trivial with scipy)
- **Multi-head** = multiple `AttentionCone` instances with different parameters

The main work is **composition and API design**, not new mathematics. Estimated total
new code: ~400 LOC library + ~400 LOC tests + ~200 LOC examples/bridge.

No existing SpinStep code is modified beyond adding exports. All existing tests continue
to pass.

---

[⬅️ 09. API Reference](09-api-reference.md) | [🏠 Home](index.md)
