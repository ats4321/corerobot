"""Robot primitives: the only interface generated code may use to act in the scene.

All coordinates are meters in the world frame: the table top is z = 0, the robot
base is at (0, 0), +x points away from the robot. Every primitive returns a dict
{"ok": bool, "reason": str, "details": dict} and never raises on a physical failure
(unreachable pose, missed grasp, nothing held). Check "ok" before continuing.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pybullet as p

from core_robot.sim.env import BIN_FLOOR_T, SIM_HZ, TabletopEnv

Target = Union[str, tuple[float, float], list[float]]

EE_LINK = 11  # panda_grasptarget: the point between the fingertips
FINGERS = (9, 10)
ARM_JOINTS = list(range(7))
# Tucked toward the base: every arm link stays at x < 0.3, clear of the object workspace,
# so the overhead camera sees the whole scene when the arm is home.
HOME_JOINTS = [0.0, -1.5, 0.0, -2.95, 0.0, 1.6, math.pi / 4]
LOWER = [-2.9, -1.8, -2.9, -3.0, -2.9, -0.08, -2.9]
UPPER = [2.9, 1.8, 2.9, -0.07, 2.9, 3.8, 2.9]
RANGES = [u - l for l, u in zip(LOWER, UPPER)]

HOVER_CLEARANCE = 0.15  # gripper height above a target's top surface for move_above / pre-grasp
CARRY_Z = 0.28  # minimum gripper height while carrying an object
GRASP_TOLERANCE = 0.02  # gripper must get within 2 cm of the object to grab it
SPEED = 0.5  # m/s along straight-line gripper paths
REACH_TOLERANCE = 0.01  # gripper position error allowed at the end of a move


def _ok(reason: str = "", **details) -> dict:
    return {"ok": True, "reason": reason, "details": details}


def _fail(reason: str, **details) -> dict:
    return {"ok": False, "reason": reason, "details": details}


def _r(v) -> list[float]:
    return [round(float(x), 4) for x in v]


class Robot:
    """A Franka Panda arm with a parallel gripper that acts through four primitives.

    Primitives:
      move_above(target)    hover the open gripper above an object or (x, y) point
      pick(obj)             grasp an object from above and lift it
      place(obj, target)    carry the held object over a target and release it there
      open_gripper()        open the fingers, releasing anything held

    A target is an object name (e.g. "red_bin", "blue_block") or an (x, y) table point
    in meters. Every primitive returns {"ok": bool, "reason": str, "details": dict} and
    appends an entry to self.call_log.
    """

    def __init__(self, env: TabletopEnv, calibration_offset: tuple[float, float] | None = None):
        self.env = env
        # Hidden release offset (meters) for the opt-in calibration_offset perturbation.
        self._offset = calibration_offset or (0.0, 0.0)
        self._true: dict | None = None  # true outcome of the last place, for logs only
        self.held: str | None = None
        self._grasp_cid: int | None = None
        self.call_log: list[dict] = []
        self.home()

    # ---- primitives ----

    def move_above(self, target: Target) -> dict:
        """Move the gripper to hover above a target, without touching anything.

        Args:
            target: an object name such as "red_bin", or an (x, y) table point in meters.

        The gripper stops 15 cm above the target's top surface (higher if carrying
        an object). Does not open or close the gripper.

        Returns:
            {"ok": True, "reason": "", "details": {"gripper_position": [x, y, z]}} on success;
            ok=False if the target is unknown or the pose is unreachable.
        """
        return self._logged("move_above", {"target": target}, lambda: self._move_above(target))

    def pick(self, obj: str) -> dict:
        """Grasp an object from above and lift it.

        Args:
            obj: name of the object to pick, e.g. "red_block" or "mug". Bins cannot be picked.

        Opens the gripper, hovers above obj, descends, closes the gripper, and lifts.
        The grasp succeeds only if the gripper gets within 2 cm of the object.
        Fails if something is already held (place it or call open_gripper first).

        Returns:
            {"ok": True, "details": {"held": obj, "object_position": [x, y, z]}} on success;
            ok=False with a reason such as "unknown object", "already holding X",
            "cannot pick a bin", "unreachable", or "grasp missed: gripper N m from object".
        """
        return self._logged("pick", {"obj": obj}, lambda: self._pick(obj))

    def place(self, obj: str, target: Target) -> dict:
        """Carry the held object over a target and release it there.

        Args:
            obj: the object currently held (must match the last successful pick).
            target: where to put it: a bin name ("red_bin") drops it inside the bin; another
                object name ("blue_block", "mug") sets it on top of that object; an (x, y)
                point sets it on the table there.

        Lifts to a safe carry height, moves above the target, lowers the object to just
        above the target surface, opens the gripper, retreats upward, and returns the
        arm to its home pose near the robot base, leaving the table in clear view.

        Returns:
            {"ok": True, "details": {"object_position": [x, y, z], "on_target": True}} when
            the object comes to rest on/in the target; ok=False if obj is not held, the
            target is unknown or unreachable, or the object settled off the target
            (details then include where it ended up).
        """
        return self._logged("place", {"obj": obj, "target": target}, lambda: self._place(obj, target))

    def open_gripper(self) -> dict:
        """Open the gripper fingers, releasing any held object where it is.

        Returns:
            {"ok": True, "details": {"released": name or None}}. Never fails.
        """
        return self._logged("open_gripper", {}, self._open_gripper)

    # ---- setup ----

    def home(self):
        """Snap the arm to its home pose with the gripper open (setup only, not a primitive)."""
        for j, q in zip(ARM_JOINTS, HOME_JOINTS):
            p.resetJointState(self.env.robot, j, q, physicsClientId=self.env.cid)
        for f in FINGERS:
            p.resetJointState(self.env.robot, f, 0.04, physicsClientId=self.env.cid)
        self._hold_arm(HOME_JOINTS)
        self._set_fingers(0.04)
        self._down = p.getQuaternionFromEuler([math.pi, 0, 0])

    # ---- implementation ----

    def _logged(self, name: str, args: dict, fn) -> dict:
        self._true = None
        try:
            result = fn()
        except Exception as e:  # programming errors still surface as a failed call, never a crash
            result = _fail(f"internal error: {type(e).__name__}: {e}")
        entry = {"primitive": name, "args": args, **result, "sim_time": round(self.env.sim_time, 3)}
        if self._true is not None:  # logs keep the true outcome; the caller only sees `result`
            entry["true"] = self._true
        self.call_log.append(entry)
        return result

    def _resolve(self, target: Target):
        """Return (x, y, top_z, name_or_None) for a target, or None if invalid."""
        if isinstance(target, str):
            if target not in self.env.bodies:
                return None
            lo, hi = self.env.aabb(target)
            pos = self.env.scene_state()[target]["position"]
            return pos[0], pos[1], float(hi[2]), target
        try:
            x, y = float(target[0]), float(target[1])
        except (TypeError, ValueError, IndexError):
            return None
        return x, y, 0.0, None

    def _move_above(self, target):
        t = self._resolve(target)
        if t is None:
            return _fail(f"unknown target {target!r}", known_objects=sorted(self.env.bodies))
        x, y, top, _ = t
        z = top + HOVER_CLEARANCE
        if self.held:
            z = max(z, CARRY_Z)
        reached, ee = self._move_to([x, y, z])
        if not reached:
            return _fail("unreachable", requested=_r([x, y, z]), gripper_position=_r(ee))
        return _ok(gripper_position=_r(ee))

    def _pick(self, obj):
        if obj not in self.env.bodies:
            return _fail(f"unknown object {obj!r}", known_objects=sorted(self.env.bodies))
        if self.env.specs[obj].kind == "bin":
            return _fail("cannot pick a bin")
        if self.held:
            return _fail(f"already holding {self.held}")
        self._set_fingers(0.04)
        lo, hi = self.env.aabb(obj)
        pos, orn = p.getBasePositionAndOrientation(self.env.bodies[obj], physicsClientId=self.env.cid)
        yaw = p.getEulerFromQuaternion(orn)[2]
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4  # cube symmetry: nearest face alignment
        grip_orn = p.getQuaternionFromEuler([math.pi, 0, yaw])
        above = [pos[0], pos[1], float(hi[2]) + HOVER_CLEARANCE]
        reached, ee = self._move_to(above, grip_orn)
        if not reached:
            return _fail("unreachable", requested=_r(above), gripper_position=_r(ee))
        # Grasp point: object center, but no deeper than 3 cm below its top and 1.5 cm above the table.
        grasp_z = max(float(pos[2]), float(hi[2]) - 0.03, 0.015)
        self._move_to([pos[0], pos[1], grasp_z], grip_orn, speed=0.25)
        ee = self._ee_pos()
        gap = _dist_to_aabb(ee, *self.env.aabb(obj))
        if gap > GRASP_TOLERANCE:
            self._move_to(above, grip_orn)
            return _fail(f"grasp missed: gripper {gap:.3f} m from object", gripper_position=_r(ee))
        self._attach(obj)
        self._set_fingers(0.0)
        self.env.step(int(0.2 * SIM_HZ))
        reached, ee = self._move_to([pos[0], pos[1], max(above[2], CARRY_Z)], grip_orn)
        obj_pos = self.env.scene_state()[obj]["position"]
        return _ok(held=obj, object_position=_r(obj_pos))

    def _place(self, obj, target):
        if self.held is None:
            return _fail(f"not holding {obj}; call pick({obj!r}) first")
        if obj != self.held:
            return _fail(f"holding {self.held}, not {obj}")
        t = self._resolve(target)
        if t is None:
            return _fail(f"unknown target {target!r}", known_objects=sorted(self.env.bodies))
        x, y, top, name = t
        if name == obj:
            return _fail("cannot place an object on itself")
        kind = self.env.specs[name].kind if name else "point"
        lo, hi = self.env.aabb(obj)
        obj_pos = np.array(self.env.scene_state()[obj]["position"])
        ee = self._ee_pos()
        below_ee = ee[2] - lo[2]  # how far the held object hangs below the gripper point
        surface = (self.env.scene_state()[name]["position"][2] + BIN_FLOOR_T) if kind == "bin" else top
        release_z = surface + below_ee + 0.015
        orn = self._ee_orn()
        dx, dy = self._offset
        rx, ry = x + dx, y + dy  # where the arm really goes (equal to x, y unless perturbed)
        carry = [rx, ry, max(top + HOVER_CLEARANCE, CARRY_Z)]
        reached, ee = self._move_to([ee[0], ee[1], max(ee[2], CARRY_Z)], orn)
        reached, ee = self._move_to(carry, orn)
        if not reached:
            return _fail("unreachable", requested=_r([x, y, carry[2]]), gripper_position=_r(ee - [dx, dy, 0]))
        self._move_to([rx, ry, release_z], orn, speed=0.25)
        self._detach()
        self._set_fingers(0.04)
        self.env.step(int(0.3 * SIM_HZ))
        self._move_to([rx, ry, max(release_z + 0.1, top + HOVER_CLEARANCE)], orn)
        self._go_home()
        self.env.step(int(0.3 * SIM_HZ))
        true_final = self.env.scene_state()[obj]["position"]
        final = true_final
        if (dx, dy) != (0.0, 0.0):
            # Perturbed: the robot believes the object rests exactly at the commanded point.
            # The true outcome goes to the logs only.
            final = [x, y, surface + (hi[2] - lo[2]) / 2]
            self._true = {"object_position": _r(true_final), "release_xy": _r([rx, ry])}
        on = bool(self._on_target(obj, final, x, y, name, kind))
        details = dict(object_position=_r(final), on_target=on, target_xy=_r([x, y]))
        if not on:
            return _fail("object settled off target", **details)
        return _ok(**details)

    def _on_target(self, obj, final, x, y, name, kind) -> bool:
        if kind == "point":
            return math.hypot(final[0] - x, final[1] - y) < 0.03
        lo, hi = self.env.aabb(name)
        inside_xy = lo[0] <= final[0] <= hi[0] and lo[1] <= final[1] <= hi[1]
        if kind == "bin":
            return inside_xy and final[2] < hi[2]
        return inside_xy and final[2] > hi[2] - 0.01

    def _open_gripper(self):
        released = self.held
        self._detach()
        self._set_fingers(0.04)
        self.env.step(int(0.3 * SIM_HZ))
        return _ok(released=released)

    # ---- low-level control ----

    def _attach(self, obj):
        c = self.env.cid
        body = self.env.bodies[obj]
        ee_pos, ee_orn = self._ee_pose()
        ob_pos, ob_orn = p.getBasePositionAndOrientation(body, physicsClientId=c)
        inv_pos, inv_orn = p.invertTransform(ee_pos, ee_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, ob_pos, ob_orn)
        self._grasp_cid = p.createConstraint(
            self.env.robot, EE_LINK, body, -1, p.JOINT_FIXED, [0, 0, 0], rel_pos, [0, 0, 0],
            parentFrameOrientation=rel_orn, physicsClientId=c,
        )
        p.changeConstraint(self._grasp_cid, maxForce=500, physicsClientId=c)
        for link in (*FINGERS, 8):  # no finger contact while held: avoids jitter
            p.setCollisionFilterPair(self.env.robot, body, link, -1, 0, physicsClientId=c)
        self.held = obj

    def _detach(self):
        c = self.env.cid
        if self._grasp_cid is not None:
            p.removeConstraint(self._grasp_cid, physicsClientId=c)
            self._grasp_cid = None
        if self.held:
            body = self.env.bodies[self.held]
            for link in (*FINGERS, 8):
                p.setCollisionFilterPair(self.env.robot, body, link, -1, 1, physicsClientId=c)
        self.held = None

    def _set_fingers(self, width_each: float):
        p.setJointMotorControlArray(
            self.env.robot, list(FINGERS), p.POSITION_CONTROL, targetPositions=[width_each] * 2,
            forces=[40, 40], physicsClientId=self.env.cid,
        )

    def _go_home(self, seconds: float = 1.5):
        """Joint-space move to HOME_JOINTS (smooth, so recorded video stays continuous)."""
        c = self.env.cid
        start = np.array([p.getJointState(self.env.robot, j, physicsClientId=c)[0] for j in ARM_JOINTS])
        goal = np.array(HOME_JOINTS)
        n = int(seconds * SIM_HZ)
        for i in range(1, n + 1):
            self._hold_arm(start + (goal - start) * i / n)
            self.env.step()
        self.env.step(int(0.3 * SIM_HZ))

    def _hold_arm(self, q):
        p.setJointMotorControlArray(
            self.env.robot, ARM_JOINTS, p.POSITION_CONTROL, targetPositions=list(q)[:7],
            forces=[300] * 7, positionGains=[0.08] * 7, physicsClientId=self.env.cid,
        )

    def _ee_pose(self):
        s = p.getLinkState(self.env.robot, EE_LINK, computeForwardKinematics=True, physicsClientId=self.env.cid)
        return s[4], s[5]

    def _ee_pos(self) -> np.ndarray:
        return np.array(self._ee_pose()[0])

    def _ee_orn(self):
        return self._ee_pose()[1]

    def _ik(self, pos, orn):
        rest = [p.getJointState(self.env.robot, j, physicsClientId=self.env.cid)[0] for j in ARM_JOINTS]
        return p.calculateInverseKinematics(
            self.env.robot, EE_LINK, pos, orn, lowerLimits=LOWER, upperLimits=UPPER, jointRanges=RANGES,
            restPoses=rest, maxNumIterations=100, residualThreshold=1e-4, physicsClientId=self.env.cid,
        )

    def _move_to(self, goal, orn=None, speed: float = SPEED) -> tuple[bool, np.ndarray]:
        """Straight-line gripper move to goal; returns (reached, final gripper position)."""
        orn = orn if orn is not None else self._down
        start = self._ee_pos()
        goal = np.asarray(goal, dtype=float)
        n = max(1, int(np.linalg.norm(goal - start) / speed * SIM_HZ))
        for i in range(1, n + 1):
            self._hold_arm(self._ik(start + (goal - start) * i / n, orn))
            self.env.step()
        for _ in range(int(1.0 * SIM_HZ)):  # settle
            self._hold_arm(self._ik(goal, orn))
            self.env.step()
            if np.linalg.norm(self._ee_pos() - goal) < REACH_TOLERANCE / 2:
                break
        ee = self._ee_pos()
        return bool(np.linalg.norm(ee - goal) < REACH_TOLERANCE), ee


def _dist_to_aabb(pt, lo, hi) -> float:
    d = np.maximum(np.maximum(lo - pt, 0), pt - hi)
    return float(np.linalg.norm(d))
