#!/usr/bin/env python3
"""Action-only Panda expert for one v2 hanging task; no mid-episode pose writes.

This is a privileged simulator expert, not a learned policy. A generated
trajectory is usable for training only if report.json says accepted=true.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


PROFILES = {
    0: {"depth": .085, "local_x": -.010, "local_y": -.018, "lift": .18,
        "yaw": 55., "x_offset": -.006, "y_offset": -.002,
        "entry_clearance": .03, "pose_gain": 1., "rotation_gain": .8,
        "prompt": "Pick up the short L-shaped metal part with a round head and hang it by its round head on an empty slot of the top rack. Release it."},
    1: {"depth": .065, "local_x": -.010, "local_y": .015, "lift": .18,
        "yaw": 135., "x_offset": -.004, "y_offset": .003,
        "entry_clearance": .018, "transit_z": 1.52,
        "pose_gain": 1., "rotation_gain": .8,
        "prompt": "Pick up the short T-shaped metal part with a round head and hang it by its round head on an empty slot of the top rack. Release it."},
    2: {"depth": .120, "local_x": -.010, "local_y": .018, "lift": .18,
        "yaw": 135., "x_offset": 0., "y_offset": 0., "z_offset": 0.,
        "pose_gain": 1., "rotation_gain": .8,
        "transit_z": 1.48, "transit_y": -.10,
        "prompt": "Pick up the medium T-shaped metal part with a round head and hang it by its round head on an empty slot of the middle rack. Release it."},
    3: {"depth": .230, "local_x": .015, "local_y": -.030, "lift": .30,
        "yaw": 5., "travel_yaw": 5., "x_offset": 0., "y_offset": 0.,
        "grasp_yaw": -2.333, "side_x": 0., "side_y": .20,
        "transit_z": 1.55, "transit_y": .10,
        "prompt": "Pick up the long metal part lying on the table and hang it by its round head deep inside an empty slot of the bottom rack. Release it."},
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int, choices=range(4), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--record", action="store_true", help="Store both aligned 256px RGB streams and video")
    p.add_argument("--render-backend", choices=("egl", "osmesa", "glfw"), help="Rendering backend; defaults to MUJOCO_GL or the platform default")
    p.add_argument("--yaw", type=float)
    p.add_argument("--grasp-depth", type=float)
    p.add_argument("--lift-height", type=float)
    p.add_argument("--slot-x-offset", type=float)
    p.add_argument("--slot-y-offset", type=float)
    p.add_argument("--slot-z-offset", type=float)
    p.add_argument("--transit-z", type=float)
    p.add_argument("--transit-y", type=float)
    p.add_argument("--side-x", type=float, help="Obj4 route x outside the tabletop")
    p.add_argument("--side-y", type=float, help="Obj4 route y before descending")
    p.add_argument("--travel-yaw", type=float, help="Obj4 orientation until the lower-side aisle")
    p.add_argument("--fixed-grasp", action="store_true", help="Plan each phase with a measured rigid grasp transform")
    p.add_argument("--grasp-only", action="store_true", help="Diagnostic lift only; never accepted as hanging data")
    p.add_argument("--grasp-local-x", type=float)
    p.add_argument("--grasp-local-y", type=float)
    p.add_argument("--grasp-yaw", type=float, help="World-Z rotation of the initial downward gripper")
    p.add_argument("--pose-gain", type=float)
    p.add_argument("--rotation-gain", type=float)
    p.add_argument("--entry-clearance", type=float, help="Cap height above the bearing while entering the top comb")
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    backend = args.render_backend or os.environ.get("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    if args.record and backend not in ("egl", "osmesa", "glfw"):
        p.error("Recording requires --render-backend osmesa/egl/glfw (MUJOCO_GL cannot be disable)")
    os.environ["MUJOCO_GL"] = backend if args.record else "disable"
    if args.record and backend in ("egl", "osmesa"):
        os.environ["PYOPENGL_PLATFORM"] = backend

    from mujoco import mj_contactForce
    from robosuite.utils.transform_utils import quat2axisangle
    from hanging_camera_profile import apply_profile
    from state_provenance import runtime_fingerprint

    core = "liberopro.liberopro"
    wrapper = importlib.import_module(core + ".envs.env_wrapper")
    suite = importlib.import_module(core + ".benchmark").get_benchmark("libero_hookrack")()
    profile = dict(PROFILES[args.task])
    for key, fallback in [('side_x',-.4),('side_y',.25),('grasp_yaw',0.),
                          ('pose_gain',.4),('rotation_gain',.35),('entry_clearance',.10)]:
        if getattr(args,key) is None:
            setattr(args,key,profile.get(key,fallback))
    if args.yaw is not None:
        profile["yaw"] = args.yaw
    if args.grasp_depth is not None:
        profile["depth"] = args.grasp_depth
    if args.grasp_local_x is not None:
        profile["local_x"] = args.grasp_local_x
    if args.grasp_local_y is not None:
        profile["local_y"] = args.grasp_local_y
    if args.lift_height is not None:
        profile["lift"] = args.lift_height
    if args.slot_x_offset is not None:
        profile["x_offset"] = args.slot_x_offset
    if args.slot_y_offset is not None:
        profile["y_offset"] = args.slot_y_offset
    if args.slot_z_offset is not None:
        profile["z_offset"] = args.slot_z_offset
    if args.transit_z is not None:
        profile["transit_z"] = args.transit_z
    if args.transit_y is not None:
        profile["transit_y"] = args.transit_y
    if args.travel_yaw is not None:
        profile["travel_yaw"] = args.travel_yaw
    env = wrapper.ControlEnv(
        bddl_file_name=suite.get_task_bddl_file_path(args.task), robots=["Panda"],
        use_camera_obs=args.record, has_renderer=False, has_offscreen_renderer=args.record,
        camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=256,
        camera_widths=256, control_freq=20, horizon=10000,
        ignore_done=True, hard_reset=True,
    )
    actions, states_before, states_after, observations, joint_states, traces = [], [], [], [], [], []
    images = {"agentview": [], "robot0_eye_in_hand": []}
    report = {"accepted": False, "task": args.task, "seed": args.seed,
              "prompt": profile["prompt"],
              "profile": profile, "fps": 20, "recorded_images": args.record,
              "route": {"side_x": args.side_x, "side_y": args.side_y, "fixed_grasp": args.fixed_grasp, "grasp_yaw": args.grasp_yaw,
                        "pose_gain": args.pose_gain, "rotation_gain": args.rotation_gain,
                        "entry_clearance": args.entry_clearance},
              "method": "sim-state feedback + normalized OSC actions; no object pose writes or attachment",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        env.seed(args.seed)
        obs = env.reset()
        raw, sim = env.env, env.env.sim
        report["runtime"] = runtime_fingerprint(core)
        report["camera_profile"] = apply_profile(sim)
        obs = raw._get_observations(force_update=True)
        robot = raw.robots[0]
        tcp = robot.eef_site_id
        tcp = next(iter(tcp.values())) if isinstance(tcp, dict) else tcp
        ctrl = getattr(robot, "controller", None)
        if ctrl is None:
            ctrl = robot.part_controllers["right"]
        if getattr(ctrl, "input_ref_frame", "world") != "world" or getattr(ctrl, "input_type", "delta") != "delta":
            raise RuntimeError("Expert requires world-frame delta OSC")
        scale = (np.asarray(ctrl.output_max) - np.asarray(ctrl.output_min)) / 2
        if scale.shape != (6,):
            raise RuntimeError(f"Expected six OSC components, got {scale}")
        obj = raw.get_object(raw.target_object_name)
        rack = raw.get_object(raw.target_rack_name)
        bid = sim.model.body_name2id(obj.root_body)
        gids = {sim.model.geom_name2id(name) for name in obj.contact_geoms}
        capsite = raw._prefixed_site(raw.target_object_name, "cap_center_site")
        slotsite = raw._prefixed_site(raw.target_rack_name, "target_slot_site")
        slot = sim.data.get_site_xpos(slotsite).copy()
        report.update(target_object=raw.target_object_name, target_rack=raw.target_rack_name,
                      slot=slot.tolist(), action_scale=scale.tolist(),
                      object_mass_kg=float(sim.model.body_mass[bid]),
                      gravity=sim.model.opt.gravity.tolist(), model_equalities=int(sim.model.neq))
        if sim.model.neq != 0 or not np.allclose(sim.model.opt.gravity, [0, 0, -9.81]):
            raise RuntimeError("Unexpected weld or gravity configuration")

        def tcp_pose():
            return sim.data.site_xpos[tcp].copy(), sim.data.site_xmat[tcp].reshape(3, 3).copy()

        def cap_pose():
            return sim.data.get_site_xpos(capsite).copy(), sim.data.body_xmat[bid].reshape(3, 3).copy()

        def step(action, phase):
            nonlocal obs
            # Execute the very float32 command stored for policy/replay use.
            # Rounding only during replay can change thin-edge contacts.
            action = np.asarray(action, dtype=np.float32)
            if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.000001:
                raise RuntimeError("Invalid normalized expert action")
            states_before.append(sim.get_state().flatten().copy())
            observations.append(np.r_[obs["robot0_eef_pos"],
                                      quat2axisangle(obs["robot0_eef_quat"].copy()),
                                      obs["robot0_gripper_qpos"]].astype(np.float32))
            joint_states.append(np.r_[sim.data.qpos[robot._ref_joint_pos_indexes],
                                      sim.data.qvel[robot._ref_joint_vel_indexes]].copy())
            if args.record:
                for name in images:
                    images[name].append(obs[name + "_image"].copy())
            actions.append(action.copy())
            obs, _, _, _ = env.step(action)
            states_after.append(sim.get_state().flatten().copy())
            contacts = []
            for ci in range(sim.data.ncon):
                c = sim.data.contact[ci]
                if c.geom1 not in gids and c.geom2 not in gids:
                    continue
                force = np.zeros(6)
                mj_contactForce(sim.model._model, sim.data._data, ci, force)
                if force[0] <= .01:
                    continue
                own = c.geom1 if c.geom1 in gids else c.geom2
                other = c.geom2 if c.geom1 in gids else c.geom1
                contacts.append({"object_geom": sim.model.geom_id2name(own),
                                 "other": sim.model.geom_id2name(other),
                                 "force": float(force[0]), "distance": float(c.dist)})
            cap, rot = cap_pose()
            traces.append({"step": len(actions) - 1, "time": float(sim.data.time),
                           "phase": phase, "cap": cap.tolist(), "upright": float(rot[2, 2]),
                           "tcp": tcp_pose()[0].tolist(), "contacts": contacts,
                           "diagnostics": raw.get_hookrack_success_diagnostics()})

        def move(target, rot, grip, phase, n=120, ramp=True):
            origin, r0 = tcp_pose()
            slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([r0, rot])))
            for i in range(n):
                t = min(1., (i + 1) / (n * .75)) if ramp else 1.
                wp = origin + (target - origin) * t
                wr = slerp(t).as_matrix()
                pp, rr = tcp_pose()
                dp = (wp - pp) * .65
                dp *= min(1., .04 / max(np.linalg.norm(dp), 1e-9))
                dr = Rotation.from_matrix(wr @ rr.T).as_rotvec() * .5
                dr *= min(1., .04 / max(np.linalg.norm(dr), 1e-9))
                step(np.r_[np.clip(np.r_[dp, dr] / scale, -1, 1), grip], phase)
            error = float(np.linalg.norm(tcp_pose()[0] - target))
            print(f"task={args.task} seed={args.seed} {phase}: tcp_error={error:.4f}", flush=True)
            return error

        def move_cap(target, rot, phase, n=160, limit=.015):
            origin, r0 = cap_pose()
            initial_tcp, initial_tcp_rot = tcp_pose()
            local_p = r0.T @ (initial_tcp - origin)
            local_r = r0.T @ initial_tcp_rot
            slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([r0, rot])))
            for i in range(n):
                t = min(1., (i + 1) / (n * .70))
                wp = origin + (target - origin) * t
                wr = slerp(t).as_matrix()
                cp, cr = cap_pose()
                pp, rr = tcp_pose()
                desired = wp + wr @ (local_p if args.fixed_grasp else cr.T @ (pp - cp))
                dp = (desired - pp) * args.pose_gain
                dp *= min(1., limit / max(np.linalg.norm(dp), 1e-9))
                rotation_error = wr @ local_r @ rr.T if args.fixed_grasp else wr @ cr.T
                dr = Rotation.from_matrix(rotation_error).as_rotvec() * args.rotation_gain
                dr *= min(1., .025 / max(np.linalg.norm(dr), 1e-9))
                step(np.r_[np.clip(np.r_[dp, dr] / scale, -1, 1), 1.], phase)
            error = float(np.linalg.norm(cap_pose()[0] - target))
            print(f"task={args.task} seed={args.seed} {phase}: cap_error={error:.4f}", flush=True)
            return error

        for _ in range(30):
            step(np.r_[np.zeros(6), -1.], "settle")
        cp, cr = cap_pose()
        grasp = cp + cr @ [profile["local_x"], profile["local_y"], -profile["depth"]]
        pp, rr = tcp_pose()
        rr = Rotation.from_euler("z", args.grasp_yaw, degrees=True).as_matrix() @ rr
        move(np.r_[grasp[:2], pp[2]], rr, -1, "approach", 160)
        move(grasp, rr, -1, "descend", 130)
        move(grasp, rr, 1, "close", 35, False)
        lift_delta = np.array([0, 0, profile["lift"]])
        move(grasp + lift_delta, rr, 1, "lift", 150)
        move(grasp + lift_delta, rr, 1, "grasp_hold", 60, False)
        held = [r for r in traces if r["phase"] == "grasp_hold"]
        lift_height = min(r["cap"][2] for r in held) - traces[29]["cap"][2]
        bilateral = sum(any("finger1" in c["other"] for c in r["contacts"])
                        and any("finger2" in c["other"] for c in r["contacts"])
                        for r in held) / len(held)
        external = sum(any(not c["other"].startswith("gripper") for c in r["contacts"]) for r in held)
        report.update(lift_height_m=lift_height, bilateral_hold_fraction=bilateral,
                      unsupported_hold_frames=external)
        report["lift_verified"] = lift_height > .10 and bilateral >= .95 and external == 0
        if not report["lift_verified"]:
            raise RuntimeError("Sustained bilateral grasp/lift failed")
        if args.grasp_only:
            report["grasp_only"] = True
            return 0

        cp, cr = cap_pose()
        pp, rr = tcp_pose()
        relp = cr.T @ (pp - cp)
        relr = cr.T @ rr
        goalr = Rotation.from_euler("z", profile["yaw"], degrees=True).as_matrix()
        travelr = (Rotation.from_euler("z", profile.get("travel_yaw", profile["yaw"]), degrees=True).as_matrix()
                   if args.task == 3 else goalr)
        goalcap = slot + [profile["x_offset"], profile["y_offset"], profile.get("z_offset", 0.)]
        transit_y = profile.get("transit_y", .15 if args.task < 2 else -.10)
        transit_z = profile.get("transit_z", 1.40 if args.task < 2 else (1.48 if args.task == 2 else 1.65))
        transitcap = np.array([goalcap[0], transit_y, transit_z])
        if args.task == 3:
            # Rotate near the initial workpiece before translating to the
            # side slot; a long simultaneous reach loses the grasp.
            transitcap[0] = .15
        move_cap(transitcap, travelr, "orient_front", 500)
        if args.task < 2:
            move_cap(goalcap + [0, -.14, args.entry_clearance], goalr, "prehang", 260)
            move_cap(goalcap + [0, 0, args.entry_clearance], goalr, "insert", 500)
        elif args.task == 2:
            # The raised middle rack now has enough clearance for a level
            # approach; this avoids sweeping obj3 through the bottom teeth.
            move_cap(goalcap + [0, -.14, .12], goalr, "prehang", 220)
            move_cap(goalcap + [0, 0, .12], goalr, "insert", 800)
        elif args.task == 3:
            # Descending through the middle comb traps the long cap.  First
            # rotate near the proven transit pose, then move sideways while
            # elevated before lowering outside the table's left edge.
            side_x = args.side_x
            side_y = args.side_y
            move_cap(np.array([side_x, side_y, transit_z]), travelr, "move_side_high", 320)
            move_cap(np.array([side_x, side_y, goalcap[2]+.10]), travelr, "lower_outside_table", 400)
            move_cap(np.array([side_x, side_y, goalcap[2]+.10]), goalr, "align_for_slot", 260)
            move_cap(np.array([goalcap[0], side_y, goalcap[2]+.10]), goalr, "enter_bottom_aisle", 320)
            move_cap(goalcap + [0, 0, .10], goalr, "insert", 500)
        move_cap(goalcap, goalr, "seat", 180)
        pp, rr = tcp_pose()
        move(pp, rr, -1, "release", 40, False)
        if args.task == 1 or (args.task == 3 and profile["depth"] >= .03):
            # At the raised top rack, the open hand is a little left of the
            # cap.  Move further left before withdrawing from the wall.
            pp, rr = tcp_pose()
            move(pp + [-.13, 0, 0], rr, -1, "clear_release", 120)
        pp, rr = tcp_pose()
        move(pp + [0, -.18, 0], rr, -1, "retreat", 140)
        pp, rr = tcp_pose()
        move(pp, rr, -1, "verify_hang", 100, False)
        verified = [r for r in traces if r["phase"] == "verify_hang"]
        target_teeth = {f"{raw.target_rack_name}_target_left_tooth_collision",
                        f"{raw.target_rack_name}_target_right_tooth_collision"}
        valid = [r["diagnostics"]["instant_valid"] and len(r["contacts"]) >= 1
                 and all(c["object_geom"].endswith("cap_collision")
                         and c["other"].startswith(tuple(target_teeth)) for c in r["contacts"])
                 for r in verified]
        max_penetration = max((-c["distance"] for r in traces for c in r["contacts"]), default=0.)
        report.update(stable_check_seconds=len(verified) / 20,
                      valid_verify_frames=sum(valid),
                      max_object_penetration_m=float(max_penetration),
                      final_diagnostics=verified[-1]["diagnostics"],
                      cap_drift_during_verification_m=float(np.max(np.linalg.norm(
                          np.asarray([r["cap"] for r in verified]) - verified[0]["cap"], axis=1))))
        report["accepted"] = bool(all(valid) and report["lift_verified"]
                                  and report["max_object_penetration_m"] < .002
                                  and env.check_success())
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        report["steps"] = len(actions)
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output / "trace.json").write_text(json.dumps(traces), encoding="utf-8")
        if actions:
            np.savez_compressed(args.output / "trajectory.npz",
                                actions=np.asarray(actions), states_before=np.asarray(states_before),
                                states_after=np.asarray(states_after), observation_state=np.asarray(observations),
                                robot_joint_state=np.asarray(joint_states),
                                timestamps=np.arange(len(actions)) / 20)
        if args.record and images["agentview"]:
            np.savez_compressed(args.output / "images_raw.npz",
                                **{key: np.asarray(value) for key, value in images.items()})
            import imageio.v2 as imageio
            with imageio.get_writer(args.output / "main_wrist.mp4", fps=20) as video:
                for a, b in zip(images["agentview"], images["robot0_eye_in_hand"]):
                    video.append_data(np.concatenate([a[::-1], b[::-1]], axis=1))
        env.close()
    print(json.dumps({key: report.get(key) for key in
                      ("task", "seed", "accepted", "lift_verified", "valid_verify_frames", "steps", "error")}), flush=True)
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
