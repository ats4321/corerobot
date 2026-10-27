"""PyBullet tabletop scene: a Franka Panda, four 4 cm blocks, two open-top bins, and a mug.

World frame: the table top is the plane z = 0, the robot base sits at the origin,
and +x points away from the robot across the table. Units are meters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pybullet as p
import pybullet_data

SIM_HZ = 240
DT = 1.0 / SIM_HZ

BLOCK_SIZE = 0.04
BIN_INNER = 0.14  # inner side length of a bin's square floor
BIN_WALL_H = 0.07  # rim height above the table
BIN_WALL_T = 0.01
BIN_FLOOR_T = 0.006
MUG_RADIUS = 0.035
MUG_HEIGHT = 0.09

# Sampling region for object centers (reachable, in front of the robot).
WORKSPACE_X = (0.32, 0.72)
WORKSPACE_Y = (-0.36, 0.36)

COLORS = {
    "red": (0.85, 0.12, 0.12, 1),
    "green": (0.15, 0.7, 0.25, 1),
    "blue": (0.15, 0.3, 0.85, 1),
    "yellow": (0.95, 0.85, 0.1, 1),
    "white": (0.92, 0.92, 0.92, 1),
}


@dataclass(frozen=True)
class ObjSpec:
    name: str
    kind: str  # "block" | "bin" | "mug"
    color: str
    radius: float  # footprint radius used for non-overlapping placement


OBJECTS = [
    ObjSpec("red_bin", "bin", "red", BIN_INNER / 2 + BIN_WALL_T + 0.01),
    ObjSpec("blue_bin", "bin", "blue", BIN_INNER / 2 + BIN_WALL_T + 0.01),
    ObjSpec("mug", "mug", "white", MUG_RADIUS + 0.01),
    ObjSpec("red_block", "block", "red", BLOCK_SIZE * 0.75),
    ObjSpec("green_block", "block", "green", BLOCK_SIZE * 0.75),
    ObjSpec("blue_block", "block", "blue", BLOCK_SIZE * 0.75),
    ObjSpec("yellow_block", "block", "yellow", BLOCK_SIZE * 0.75),
]

CAMERAS = {
    # Looking straight down; image top is +x (far side of the table), robot at the bottom.
    "overhead": dict(eye=(0.52, 0.0, 1.25), target=(0.52, 0.0, 0.0), up=(1, 0, 0), fov=45),
    # From the robot's right side, slightly above the table.
    "side": dict(eye=(0.55, -1.05, 0.55), target=(0.52, 0.0, 0.05), up=(0, 0, 1), fov=50),
}


class TabletopEnv:
    """The simulated scene. Call reset(seed) before use."""

    def __init__(self, gui: bool = False, image_size: tuple[int, int] = (640, 480)):
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        self.gui = gui
        self.image_size = image_size
        self.robot: int | None = None
        self.bodies: dict[str, int] = {}
        self.specs = {s.name: s for s in OBJECTS}
        self.sim_time = 0.0
        self.step_hooks: list[Callable[["TabletopEnv"], None]] = []
        self.seed: int | None = None

    # ---- lifecycle ----

    def reset(self, seed: int) -> dict:
        """Rebuild the scene with objects at seeded random, non-overlapping positions.

        The same seed always yields the same layout. Returns scene_state().
        """
        self.seed = seed
        rng = np.random.default_rng(seed)
        c = self.cid
        p.resetSimulation(physicsClientId=c)
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.setTimeStep(DT, physicsClientId=c)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=c)

        self._make_box([0.45, 0.0, -0.02], [0.55, 0.65, 0.02], (0.6, 0.45, 0.3, 1), mass=0)  # table
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=c)

        placed: list[tuple[float, float, float]] = []
        self.bodies = {}
        for spec in OBJECTS:
            x, y = self._sample_xy(rng, spec.radius, placed)
            placed.append((x, y, spec.radius))
            yaw = float(rng.uniform(-math.pi / 4, math.pi / 4)) if spec.kind == "block" else 0.0
            self.bodies[spec.name] = self._build(spec, x, y, yaw)

        self.sim_time = 0.0
        self.step(int(0.25 * SIM_HZ))  # let objects settle
        return self.scene_state()

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)

    def step(self, n: int = 1):
        """Advance physics n steps (1/240 s each), calling step hooks after each."""
        for _ in range(n):
            p.stepSimulation(physicsClientId=self.cid)
            self.sim_time += DT
            for hook in self.step_hooks:
                hook(self)

    # ---- queries ----

    def scene_state(self) -> dict[str, dict]:
        """name -> {position [x,y,z], orientation [x,y,z,w] quaternion, color, kind}."""
        out = {}
        for name, body in self.bodies.items():
            pos, orn = p.getBasePositionAndOrientation(body, physicsClientId=self.cid)
            spec = self.specs[name]
            out[name] = {
                "position": [float(v) for v in pos],
                "orientation": [float(v) for v in orn],
                "color": spec.color,
                "kind": spec.kind,
            }
        return out

    def aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (min, max) of an object."""
        lo, hi = p.getAABB(self.bodies[name], physicsClientId=self.cid)
        return np.array(lo), np.array(hi)

    def render(self, camera: str = "overhead") -> np.ndarray:
        """RGB frame (H, W, 3) uint8 from the 'overhead' or 'side' camera."""
        cam = CAMERAS[camera]
        w, h = self.image_size
        view = p.computeViewMatrix(cam["eye"], cam["target"], cam["up"], physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(cam["fov"], w / h, 0.05, 5.0, physicsClientId=self.cid)
        _, _, rgba, _, _ = p.getCameraImage(
            w, h, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=self.cid
        )
        return np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    # ---- construction helpers ----

    def _sample_xy(self, rng, radius, placed):
        for _ in range(5000):
            x = float(rng.uniform(WORKSPACE_X[0] + radius, WORKSPACE_X[1] - radius))
            y = float(rng.uniform(WORKSPACE_Y[0] + radius, WORKSPACE_Y[1] - radius))
            if all(math.hypot(x - px, y - py) >= radius + pr + 0.02 for px, py, pr in placed):
                return x, y
        raise RuntimeError("could not place objects without overlap")

    def _make_box(self, pos, half, rgba, mass, yaw=0.0):
        c = self.cid
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=c)
        orn = p.getQuaternionFromEuler([0, 0, yaw])
        return p.createMultiBody(mass, col, vis, pos, orn, physicsClientId=c)

    def _build(self, spec: ObjSpec, x: float, y: float, yaw: float) -> int:
        c = self.cid
        rgba = COLORS[spec.color]
        if spec.kind == "block":
            body = self._make_box([x, y, BLOCK_SIZE / 2], [BLOCK_SIZE / 2] * 3, rgba, mass=0.1, yaw=yaw)
            p.changeDynamics(body, -1, lateralFriction=1.0, physicsClientId=c)
            return body
        if spec.kind == "mug":
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=MUG_RADIUS, height=MUG_HEIGHT, physicsClientId=c)
            vis = p.createVisualShape(
                p.GEOM_CYLINDER, radius=MUG_RADIUS, length=MUG_HEIGHT, rgbaColor=rgba, physicsClientId=c
            )
            return p.createMultiBody(0.2, col, vis, [x, y, MUG_HEIGHT / 2], physicsClientId=c)
        # bin: static floor + four walls, base frame at the center of the floor's bottom (z = 0)
        half_in = BIN_INNER / 2
        t = BIN_WALL_T
        wall_h = BIN_WALL_H / 2
        parts = [([half_in + t, half_in + t, BIN_FLOOR_T / 2], [0, 0, BIN_FLOOR_T / 2])]
        for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if sx:
                half = [t / 2, half_in + t, wall_h]
                pos = [sx * (half_in + t / 2), 0, wall_h]
            else:
                half = [half_in + t, t / 2, wall_h]
                pos = [0, sy * (half_in + t / 2), wall_h]
            parts.append((half, pos))
        col = p.createCollisionShapeArray(
            [p.GEOM_BOX] * 5, halfExtents=[h for h, _ in parts], collisionFramePositions=[q for _, q in parts],
            physicsClientId=c,
        )
        vis = p.createVisualShapeArray(
            [p.GEOM_BOX] * 5, halfExtents=[h for h, _ in parts], visualFramePositions=[q for _, q in parts],
            rgbaColors=[rgba] * 5, physicsClientId=c,
        )
        return p.createMultiBody(0, col, vis, [x, y, 0], physicsClientId=c)


def in_bin(state: dict, obj: str, bin_name: str) -> bool:
    """True if obj's center lies inside bin_name's inner footprint and below its rim."""
    o = state[obj]["position"]
    b = state[bin_name]["position"]
    half = BIN_INNER / 2
    return abs(o[0] - b[0]) <= half and abs(o[1] - b[1]) <= half and o[2] < b[2] + BIN_WALL_H
