"""Structured task plan: what a demonstration shows, in simulator object names.

A Plan lists the scene objects, the pick-and-place steps, and the goal as relation
atoms. Success is judged from the goal atoms plus the global rule in
core_robot.verify.relations.check_success.

Destinations are object names or the literal "free_table_space"; raw coordinates
are rejected. Names must be simulator object names (e.g. red_block, blue_bin, mug).
"""

from __future__ import annotations

from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core_robot.sim.env import OBJECTS

FREE_TABLE_SPACE = "free_table_space"
SIM_OBJECTS = {o.name: o for o in OBJECTS}
RELATIONS = ("in", "on", "left_of", "right_of", "next_to", "swapped_with")
Relation = Literal["in", "on", "left_of", "right_of", "next_to", "swapped_with"]
Kind = Literal["block", "bin", "mug"]


def _known_name(v: str) -> str:
    if v not in SIM_OBJECTS:
        raise ValueError(f"unknown object name {v!r}; expected one of {sorted(SIM_OBJECTS)}")
    return v


class PlanObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    color: str
    kind: Kind

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _known_name(v)

    @model_validator(mode="after")
    def _matches_sim(self):
        spec = SIM_OBJECTS[self.name]
        if (self.color, self.kind) != (spec.color, spec.kind):
            raise ValueError(f"{self.name} is a {spec.color} {spec.kind}, not a {self.color} {self.kind}")
        return self


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    action: Literal["pick_and_place"]
    object: str
    destination: str  # an object name or "free_table_space"; coordinates are rejected

    @field_validator("object")
    @classmethod
    def _object(cls, v):
        _known_name(v)
        if SIM_OBJECTS[v].kind == "bin":
            raise ValueError(f"cannot pick a bin ({v})")
        return v

    @field_validator("destination", mode="before")
    @classmethod
    def _destination(cls, v):
        if not isinstance(v, str):
            raise ValueError(f"destination must be an object name or {FREE_TABLE_SPACE!r}, not {v!r}")
        if v != FREE_TABLE_SPACE:
            _known_name(v)
        return v


class GoalAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    relation: Relation
    target: str

    @field_validator("subject", "target")
    @classmethod
    def _names(cls, v):
        return _known_name(v)


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objects: list[PlanObject]
    steps: list[Step]
    goal: list[GoalAtom] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self):
        names = [o.name for o in self.objects]
        if len(set(names)) != len(names):
            raise ValueError("duplicate names in objects")
        listed = set(names)
        if [s.index for s in self.steps] != list(range(len(self.steps))):
            raise ValueError("step indices must be 0, 1, 2, ... in order")
        for s in self.steps:
            for n in (s.object, s.destination):
                if n != FREE_TABLE_SPACE and n not in listed:
                    raise ValueError(f"step {s.index} uses {n!r}, which is not in objects")
            if s.object == s.destination:
                raise ValueError(f"step {s.index} places {s.object} on itself")
        for g in self.goal:
            for n in (g.subject, g.target):
                if n not in listed:
                    raise ValueError(f"goal uses {n!r}, which is not in objects")
            if g.subject == g.target:
                raise ValueError(f"goal relates {g.subject} to itself")
        return self

    def check_scene(self, scene_names: Iterable[str]) -> None:
        """Raise ValueError if the plan names objects that are not in this scene."""
        missing = {o.name for o in self.objects} - set(scene_names)
        if missing:
            raise ValueError(f"plan names objects not in the scene: {sorted(missing)}")

    def moved_objects(self) -> set[str]:
        """Objects picked and placed by some step (the `object` of a step)."""
        return {s.object for s in self.steps}

    def named_in_steps(self) -> set[str]:
        """Objects that appear in any step, as the moved object or as a destination."""
        return {n for s in self.steps for n in (s.object, s.destination) if n != FREE_TABLE_SPACE}
