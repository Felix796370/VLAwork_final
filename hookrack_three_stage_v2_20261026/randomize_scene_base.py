"""Opt-in v2 reset randomization. Never changes the installed scene source.

Simulator pose writes occur only while constructing an episode, before sensing
or executing the grasp. Conservative full-CAD convex footprints are used.
"""
from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, box


def scene_constants(raw):
    module=importlib.import_module(type(raw).__module__)
    return module._INITIAL_POSES,module._REST_QUATS_WXYZ,module._collision_vertices


def rest_rotation(quat_wxyz):
    q=np.asarray(quat_wxyz)
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def hull(vertices,rotation,position):
    return MultiPoint((vertices@rotation.T+position)[:,:2]).convex_hull


def randomize_scene(env,seed=0,translation_m=.035,yaw_deg=10.,gap_m=.025,
                    table_margin_m=.008,max_attempts=20000):
    """Called immediately after env.reset, with no subsequent pose teleporting.

    This is bounded local perturbation, not a demonstrated full-table/360deg
    sampler. Failure to find a safe layout raises instead of relaxing bounds.
    """
    if not (0<=translation_m<=.10 and 0<=yaw_deg<=25 and gap_m>=.015):
        raise ValueError('Supported engineering limits: translation<=0.10m, yaw<=25deg, gap>=0.015m')
    raw=env.env
    sim=raw.sim
    poses,quats,collision_vertices=scene_constants(raw)
    center=np.asarray(raw.table_offset)
    size=np.asarray(raw.table_full_size)
    table=box(center[0]-size[0]/2+table_margin_m,center[1]-size[1]/2+table_margin_m,
              center[0]+size[0]/2-table_margin_m,center[1]+size[1]/2-table_margin_m)
    # Longest first leaves room for small pieces. Existing category lanes are
    # priors for safe workspace sampling, never fed to the vision estimator.
    from collect_expert_hanging import PROFILES
    rng=np.random.default_rng(seed)
    chosen={}
    attempts=0
    for index in (4,3,2,1):
        name=f'hook_obj{index}_1'
        obj=raw.get_object(name)
        vertices=collision_vertices(obj.category_name)
        base=rest_rotation(quats[name])
        # A yaw rotation preserves the flat projected convex hull. Restrict
        # candidate checks to its vertices instead of thousands of CAD points.
        flat=vertices@base.T
        reduced=vertices[ConvexHull(flat[:,:2]).vertices]
        anchor=np.asarray(poses[name],dtype=float).copy()
        # Original obj4 cap circle slightly overhangs the rear edge. The new
        # strict fully-on-table sampler starts farther inside, without changing
        # the original scene file or the object's supported resting attitude.
        anchor[1]={1:-.650,2:-.500,3:-.265,4:-.055}[index]
        profile=PROFILES[index-1]
        for _ in range(max_attempts):
            attempts+=1
            theta=float(rng.uniform(-yaw_deg,yaw_deg))
            rot=Rotation.from_euler('z',theta,degrees=True).as_matrix()@base
            xy=anchor+rng.uniform(-translation_m,translation_m,2)
            z=float(center[2]-np.min(vertices@rot[2])+.00015)
            p=np.r_[xy,z]
            footprint=hull(reduced,rot,p)
            grasp=p+rot@np.array([profile['local_x'],profile['local_y'],-profile['depth']])
            corridor=Point(*grasp[:2]).buffer(.055)
            if not table.covers(footprint):continue
            if any(footprint.distance(v['footprint'])<gap_m or
                   corridor.intersects(v['footprint'].buffer(.010)) or
                   v['corridor'].intersects(footprint.buffer(.010)) for v in chosen.values()):continue
            q=Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]
            chosen[name]={'position':p,'quaternion_wxyz':q,'yaw_delta_deg':theta,
                          'footprint':footprint,'corridor':corridor,'rotation':rot,'vertices':vertices}
            break
        else:
            raise RuntimeError(f'No safe layout for {name}; keep scene unchanged, do not run a grasp')
        print(f'Reset sampled {name}: xy={xy.round(4)} yaw={theta:.2f}deg',flush=True)
    for name,v in chosen.items():
        obj=raw.get_object(name)
        sim.data.set_joint_qpos(obj.joints[-1],np.r_[v['position'],v['quaternion_wxyz']])
        raw._zero_object_velocity(obj)
    sim.forward()
    before={n:sim.data.get_body_xpos(raw.get_object(n).root_body).copy() for n in chosen}
    # Only actions from this point onward, including settling.
    for _ in range(30):env.step(np.r_[np.zeros(6),-1.])
    geometries={}
    drifts={}
    for name,v in chosen.items():
        obj=raw.get_object(name)
        bid=sim.model.body_name2id(obj.root_body)
        pos=sim.data.body_xpos[bid].copy()
        rot=sim.data.body_xmat[bid].reshape(3,3)
        geometries[name]=hull(v['vertices'],rot,pos)
        drifts[name]=float(np.linalg.norm(pos-before[name]))
        if drifts[name]>.004:raise RuntimeError(f'{name} unstable after reset: {drifts[name]:.5f}m')
        if not table.covers(geometries[name]):raise RuntimeError(f'{name} outside table after settling')
    minimum_gap=min(geometries[a].distance(geometries[b]) for a in chosen for b in chosen if a<b)
    if minimum_gap<gap_m-.002:raise RuntimeError('Reset settling consumed object safety gap')
    for c in sim.data.contact[:sim.data.ncon]:
        a,b=(sim.model.geom_id2name(int(g)) or '' for g in (c.geom1,c.geom2))
        if c.dist<-.0001 and any(a.startswith(n) and b.startswith(m) for n in chosen for m in chosen if n!=m):
            raise RuntimeError('Inter-object contact at reset')
        if c.dist<-.0001 and ((a.startswith('hook_obj') and b.startswith(('robot0','gripper'))) or
                              (b.startswith('hook_obj') and a.startswith(('robot0','gripper')))):
            raise RuntimeError('Object intersects robot at reset')
    return {'seed':seed,'translation_m':translation_m,'yaw_limit_deg':yaw_deg,
            'minimum_cad_footprint_gap_m':float(minimum_gap),'settle_drift_m':drifts,
            'sampled_poses_for_audit_only':{n:{'position':v['position'].tolist(),
                'quaternion_wxyz':v['quaternion_wxyz'].tolist(),'yaw_delta_deg':v['yaw_delta_deg']} for n,v in chosen.items()},
            'pose_writes':'reset construction only','attempts':attempts}
