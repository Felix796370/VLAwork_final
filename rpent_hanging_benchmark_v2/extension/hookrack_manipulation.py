"""LIBERO scene and physical success predicate for the hook-rack benchmark."""

from __future__ import annotations

import importlib
from functools import lru_cache

import numpy as np
import robosuite.utils.transform_utils as T


_CORE = __package__.split(".envs.problems", 1)[0]
_bddl = importlib.import_module(f"{_CORE}.envs.bddl_base_domain")
register_problem = _bddl.register_problem
# LIBERO's register_problem decorator intentionally does not return the class,
# so the decorated module symbol is None.  The real class lives in TASK_MAPPING.
LiberoTabletop = _bddl.TASK_MAPPING["libero_tabletop_manipulation"]


_INITIAL_POSES = {
    # Cap centre x/y; obj4 is kept closer to the Panda and away from the wall.
    "hook_obj1_1": (-0.06, -0.57),
    "hook_obj2_1": (-0.04, -0.43),
    "hook_obj3_1": (0.05, -0.21),
    "hook_obj4_1": (0.34, 0.00),
}

# All four objects use their CAD-derived horizontal resting poses.
_REST_QUATS_WXYZ = {
    "hook_obj1_1": (0.7344484479768253, 0.0000101881552009, 0.6786644422321144, 0.0002280579063929),
    "hook_obj2_1": (0.6879210082599873, 0.2118510988372695, 0.6488396608109005, 0.2467608009283447),
    "hook_obj3_1": (0.6709838222310821, 0.2007870813205274, 0.6655304182360187, 0.2579428632111015),
    # Same CAD-supported flat rest, yawed so the long axis runs along -X,
    # parallel to the other three workpieces (no support fixture).
    "hook_obj4_1": (0.6622799459556346, -0.2865117615557129, 0.6213080230640472, -0.3054056714461758),
}


@lru_cache(maxsize=4)
def _collision_vertices(category):
    root = importlib.import_module(f"{_CORE}.envs.objects.hookrack_objects")._ASSET_ROOT
    files = sorted((root / "meshes" / "collision").glob(f"{category}_collision_*.obj"))
    if not files:
        raise FileNotFoundError(f"No CAD collision pieces for {category}")
    vertices = []
    for path in files:
        with path.open(encoding="utf-8") as mesh:
            vertices.extend(tuple(map(float, line.split()[1:4])) for line in mesh if line.startswith("v "))
    return np.asarray(vertices)


def _resting_cap_z(table_z, category, quat):
    rotation = T.quat2mat(T.convert_quat(quat, to="xyzw"))
    lowest_relative_z = float(np.min(_collision_vertices(category) @ rotation[2]))
    return table_z - lowest_relative_z - 0.00006

_FIXTURE_POSES = {
    "hook_wall_1": (-0.45, 0.47062132, 0.950),
    # All three racks raised 50 mm: bearing heights 1.67, 1.30, 0.89 m.
    # Include the NEXT rack's
    # upright CAD rear wall in the clearance, not just the part's length.
    # CAD fixtures are otherwise
    # unchanged, so both their visible teeth and collision teeth move together.
    "hook_shell1_1": (-0.45, 0.390, 1.705),
    "hook_shell2_1": (-0.45, 0.390, 1.336),
    "hook_shell3_1": (-0.45, 0.360, 0.9345),
}


def _look_at_quat(camera_pos, target):
    """Return a MuJoCo wxyz camera quaternion looking along local -Z."""
    camera_pos = np.asarray(camera_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    backward = -forward
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack((right, up, backward))
    xyzw = T.mat2quat(rotation)
    return T.convert_quat(xyzw, to="wxyz")


@register_problem
class Hook_Rack_Manipulation(LiberoTabletop):
    """One scene, four language goals, and a contact-based hanging predicate."""

    stable_duration_s = 0.25

    def __init__(self, bddl_file_name, *args, **kwargs):
        self._stable_since = None
        self._success_latched = False
        self._last_success_diagnostics = {}
        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_model(self):
        # Use a narrower table so the side-wall hanging zone sits just beyond
        # the tabletop, allowing the 610 mm member to hang freely.
        self.table_full_size = (0.90, 0.80, 0.05)
        # Moving the table forward leaves a clear aisle before the wall rack.
        self.table_offset = (0.0, -0.35, 0.80)
        self.workspace_offset = self.table_offset
        super()._load_model()
        base = self.model.worldbody.find("body[@name='robot0_base']")
        if base is None:
            raise RuntimeError("Panda base body missing")
        base_position = np.fromstring(base.get("pos"), sep=" ")
        base_position[0] += .15
        base.set("pos", " ".join(map(str, base_position)))

        # Move the Panda wrist camera 120 mm back along its optical axis and
        # widen it.  At the reset pose this exposes all four table workpieces,
        # while the camera remains rigidly attached to the right-hand body.
        for camera in self.model.worldbody.findall(".//camera"):
            if camera.get("name", "").endswith("eye_in_hand"):
                camera.set("pos", "0.05 0 -0.12")
                camera.set("quat", "0 0.70710678 0.70710678 0")
                camera.set("fovy", "110")

    def _setup_camera(self, mujoco_arena):
        # Policy main view: face the tabletop and rack wall from the aisle,
        # close enough that workpiece profiles and all three tooth rows remain
        # legible at the 256 px resolution used by Pi0.5.
        agent_pos = [1.10, -1.80, 1.66]
        agent_target = [-0.25, 0.10, 1.10]
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=agent_pos,
            quat=_look_at_quat(agent_pos, agent_target),
        )
        front_pos = [-0.45, -2.10, 1.62]
        front_target = [-0.45, 0.48, 1.17]
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=front_pos,
            quat=_look_at_quat(front_pos, front_target),
        )
        for camera in mujoco_arena.worldbody.findall(".//camera"):
            if camera.get("name") == "agentview":
                camera.set("fovy", "52")
            elif camera.get("name") == "frontview":
                camera.set("fovy", "45")

    def _reset_internal(self):
        super()._reset_internal()
        self._stable_since = None
        self._success_latched = False
        self._last_success_diagnostics = {}

        # MujocoXMLObject deliberately strips the wrapper body's source pose;
        # fixtures therefore receive their world poses on every reset.
        for fixture_name, position in _FIXTURE_POSES.items():
            fixture = self.fixtures_dict[fixture_name]
            body_id = self.sim.model.body_name2id(fixture.root_body)
            self.sim.model.body_pos[body_id] = np.asarray(position, dtype=float)
            self.sim.model.body_quat[body_id] = np.array([1.0, 0.0, 0.0, 0.0])

        # Free-joint quaternions are MuJoCo wxyz. Put each real CAD profile
        # into its previously observed resting orientation with mild contact.
        table_z = float(self.table_offset[2])
        for object_name, (x, y) in _INITIAL_POSES.items():
            obj = self.objects_dict[object_name]
            jitter = np.random.uniform(low=[-0.012, -0.010], high=[0.012, 0.010])
            quat = np.asarray(_REST_QUATS_WXYZ[object_name], dtype=float)
            cap_z = _resting_cap_z(table_z, obj.category_name, quat)
            position = np.array([x + jitter[0], y + jitter[1], cap_z])
            self.sim.data.set_joint_qpos(
                obj.joints[-1], np.concatenate((position, quat))
            )
            self._zero_object_velocity(obj)
        self.sim.forward()

    def _zero_object_velocity(self, obj):
        try:
            self.sim.data.set_joint_qvel(obj.joints[-1], np.zeros(6))
        except (AttributeError, TypeError):
            address = self.sim.model.get_joint_qvel_addr(obj.joints[-1])
            if isinstance(address, tuple):
                self.sim.data.qvel[address[0] : address[1]] = 0
            else:
                self.sim.data.qvel[address : address + 6] = 0

    @property
    def target_object_name(self):
        return self.obj_of_interest[0]

    @property
    def target_rack_name(self):
        return self.obj_of_interest[1]

    def _prefixed_site(self, object_name, suffix):
        obj = self.get_object(object_name)
        return f"{obj.naming_prefix}{suffix}"

    def _target_contacts(self):
        target_obj = self.get_object(self.target_object_name)
        target_rack = self.get_object(self.target_rack_name)
        cap_name = f"{target_obj.naming_prefix}cap_collision"
        rack_prefix = target_rack.naming_prefix
        hits = set()
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            name1 = self.sim.model.geom_id2name(contact.geom1) or ""
            name2 = self.sim.model.geom_id2name(contact.geom2) or ""
            pairs = ((name1, name2), (name2, name1))
            for maybe_cap, maybe_rack in pairs:
                if (
                    maybe_cap == cap_name
                    and maybe_rack.startswith(rack_prefix)
                    and "target_" in maybe_rack
                    and "tooth_collision" in maybe_rack
                ):
                    hits.add(maybe_rack)
        return {
            "left": any("target_left_tooth_collision" in name for name in hits),
            "right": any("target_right_tooth_collision" in name for name in hits),
        }

    def _object_velocity(self, obj):
        address = self.sim.model.get_joint_qvel_addr(obj.joints[-1])
        if isinstance(address, tuple):
            qvel = np.asarray(self.sim.data.qvel[address[0] : address[1]])
        else:
            qvel = np.asarray(self.sim.data.qvel[address : address + 6])
        return float(np.linalg.norm(qvel[:3])), float(np.linalg.norm(qvel[3:6]))

    def _compute_success_diagnostics(self):
        target_obj = self.get_object(self.target_object_name)
        cap_site = self._prefixed_site(self.target_object_name, "cap_center_site")
        slot_site = self._prefixed_site(self.target_rack_name, "target_slot_site")
        cap_pos = np.asarray(self.sim.data.get_site_xpos(cap_site)).copy()
        slot_pos = np.asarray(self.sim.data.get_site_xpos(slot_site)).copy()
        delta = cap_pos - slot_pos

        body_id = self.obj_body_id[self.target_object_name]
        body_rotation = np.asarray(self.sim.data.body_xmat[body_id]).reshape(3, 3)
        vertical_alignment = float(body_rotation[:, 2].dot(np.array([0.0, 0.0, 1.0])))
        linear_speed, angular_speed = self._object_velocity(target_obj)
        try:
            grasped = bool(
                self._check_grasp(self.robots[0].gripper, target_obj.contact_geoms)
            )
        except (AttributeError, TypeError, ValueError):
            # Conservative: if an incompatible robosuite build cannot determine
            # release, do not report success.
            grasped = True

        contacts = self._target_contacts()
        diagnostics = {
            "cap_position": cap_pos,
            "target_position": slot_pos,
            "position_delta": delta,
            "position_ok": bool(
                abs(delta[0]) <= 0.025
                and abs(delta[1]) <= (0.003 if self.target_object_name == 'hook_obj4_1' else 0.008)
                and abs(delta[2]) <= 0.003
            ),
            "vertical_alignment": vertical_alignment,
            "orientation_ok": vertical_alignment >= 0.999,
            "target_left_contact": contacts["left"],
            "target_right_contact": contacts["right"],
            # A rigid circular cap can settle with an imperceptible tilt and a
            # single active contact manifold.  Restrict contact to the two
            # teeth bordering the correct slot, but do not require both
            # manifolds to be active in the same solver step.
            "target_contact": contacts["left"] or contacts["right"],
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "stable_speed": linear_speed <= 0.08 and angular_speed <= 0.60,
            "released": not grasped,
        }
        diagnostics["instant_valid"] = bool(
            diagnostics["position_ok"]
            and diagnostics["orientation_ok"]
            and diagnostics["target_contact"]
            and diagnostics["stable_speed"]
            and diagnostics["released"]
        )
        return diagnostics

    def _check_success(self):
        if self._success_latched:
            return True
        diagnostics = self._compute_success_diagnostics()
        now = float(self.sim.data.time)
        if diagnostics["instant_valid"]:
            if self._stable_since is None:
                self._stable_since = now
            diagnostics["stable_duration"] = now - self._stable_since
            if diagnostics["stable_duration"] >= self.stable_duration_s:
                self._success_latched = True
        else:
            self._stable_since = None
            diagnostics["stable_duration"] = 0.0
        diagnostics["success"] = self._success_latched
        self._last_success_diagnostics = diagnostics
        return self._success_latched

    def get_hookrack_success_diagnostics(self):
        """Return JSON-friendly evidence for the most recent success check."""
        diagnostics = self._compute_success_diagnostics()
        output = {}
        for key, value in diagnostics.items():
            output[key] = value.tolist() if isinstance(value, np.ndarray) else value
        output["stable_since"] = self._stable_since
        output["success_latched"] = self._success_latched
        return output

    def place_target_in_goal_for_test(self):
        """Teleport helper used only by smoke_test.py to validate the predicate."""
        target_obj = self.get_object(self.target_object_name)
        slot_site = self._prefixed_site(self.target_rack_name, "target_slot_site")
        position = np.asarray(self.sim.data.get_site_xpos(slot_site)).copy()
        hanging_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos(
            target_obj.joints[-1], np.concatenate((position, hanging_quat))
        )
        self._zero_object_velocity(target_obj)
        self._stable_since = None
        self._success_latched = False
        self.sim.forward()

    def drop_target_above_goal_for_test(self, drop_height=0.12):
        """Release the target above the real slot for gravity / retention tests."""
        target_obj = self.get_object(self.target_object_name)
        slot_site = self._prefixed_site(self.target_rack_name, "target_slot_site")
        position = np.asarray(self.sim.data.get_site_xpos(slot_site)).copy()
        position[2] += float(drop_height)
        hanging_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos(
            target_obj.joints[-1], np.concatenate((position, hanging_quat))
        )
        self._zero_object_velocity(target_obj)
        self._stable_since = None
        self._success_latched = False
        self.sim.forward()


__all__ = ["Hook_Rack_Manipulation"]
