"""Custom workpiece, rack, and wall objects for the LIBERO hook-rack suite.

This module intentionally discovers its containing LIBERO namespace at runtime,
so the same file works in both ``libero.libero`` and ``liberopro.liberopro``.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from robosuite.models.objects import MujocoXMLObject


_CORE = __package__.split(".envs.objects", 1)[0]
register_object = importlib.import_module(f"{_CORE}.envs.base_object").register_object
_ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "hookrack"
_SIMULATION_MASS_KG = {"hook_obj4": 1.5}

def _category_name(cls) -> str:
    return "_".join(re.sub(r"([A-Z0-9])", r" \1", cls.__name__).split()).lower()


class _HookWorkpiece(MujocoXMLObject):
    def __init__(self, name: str, category: str):
        super().__init__(
            str(_ASSET_ROOT / f"{category}.xml"),
            name=name,
            # Small viscous damping represents bearing / air / contact losses
            # and lets the hanging pendulum settle without constraining it.
            joints=[dict(type="free", damping="0.02")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = category
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}
        report = json.loads(
            (_ASSET_ROOT / "meshes" / "collision" / "decomposition_report.json").read_text(encoding="utf-8")
        )[category]
        visual_mesh = _ASSET_ROOT / "meshes" / f"{category}.obj"
        if hashlib.sha256(visual_mesh.read_bytes()).hexdigest() != report["source_sha256"]:
            raise ValueError(f"CAD collision meshes for {category} do not match its visual mesh")
        self.estimated_physical_mass_kg = report["mass_from_304_density_kg"]
        self.simulation_mass_kg = _SIMULATION_MASS_KG.get(category, self.estimated_physical_mass_kg)


class _HookFixture(MujocoXMLObject):
    def __init__(self, name: str, category: str, joints=None, **_kwargs):
        super().__init__(
            str(_ASSET_ROOT / f"{category}.xml"),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = category
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
class HookObj1(_HookWorkpiece):
    def __init__(self, name="hook_obj1"):
        super().__init__(name, "hook_obj1")


@register_object
class HookObj2(_HookWorkpiece):
    def __init__(self, name="hook_obj2"):
        super().__init__(name, "hook_obj2")


@register_object
class HookObj3(_HookWorkpiece):
    def __init__(self, name="hook_obj3"):
        super().__init__(name, "hook_obj3")


@register_object
class HookObj4(_HookWorkpiece):
    def __init__(self, name="hook_obj4"):
        super().__init__(name, "hook_obj4")


@register_object
class HookShell1(_HookFixture):
    def __init__(self, name="hook_shell1", joints=None, **kwargs):
        super().__init__(name, "hook_shell1", joints=joints, **kwargs)


@register_object
class HookShell2(_HookFixture):
    def __init__(self, name="hook_shell2", joints=None, **kwargs):
        super().__init__(name, "hook_shell2", joints=joints, **kwargs)


@register_object
class HookShell3(_HookFixture):
    def __init__(self, name="hook_shell3", joints=None, **kwargs):
        super().__init__(name, "hook_shell3", joints=joints, **kwargs)


@register_object
class HookWall(_HookFixture):
    def __init__(self, name="hook_wall", joints=None, **kwargs):
        super().__init__(name, "hook_wall", joints=joints, **kwargs)


__all__ = [
    "HookObj1",
    "HookObj2",
    "HookObj3",
    "HookObj4",
    "HookShell1",
    "HookShell2",
    "HookShell3",
    "HookWall",
]
