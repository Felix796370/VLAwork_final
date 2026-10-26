"""Four physical CAD instances, repeated classes and random order; reset only."""
import importlib
import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import box, Point
from randomize_scene_base import scene_constants, rest_rotation, hull

KINDS = ('obj1','obj2','obj3','obj4')

def grasp_profiles():
    """Keep the long T member's grip near its CAD centre of mass."""
    from collect_expert_hanging import PROFILES
    profiles={i:dict(p) for i,p in PROFILES.items()}
    # obj3 COM lies 185mm below its cap. A 175mm grip avoids the large
    # cantilever torque of the old 120mm grasp during free-space reorientation.
    profiles[2]['depth']=.175
    return profiles

def sample_layout(raw, kinds, seed, translation_m=.035, yaw_deg=10., gap=.025,reach_test=None):
    if len(kinds)!=4 or any(k not in KINDS for k in kinds):
        raise ValueError('Exactly four instances with classes obj1..obj4 required')
    if not 0<=yaw_deg<=25 or not 0<=translation_m<=.10:
        raise ValueError('yaw must be 0..25 degrees; translation 0..0.1m')
    _,quats,vertices_fn=scene_constants(raw)
    PROFILES=grasp_profiles()
    rng=np.random.default_rng(seed)
    center=np.asarray(raw.table_offset); size=np.asarray(raw.table_full_size)
    low=center[:2]-size[:2]/2+.008; high=center[:2]+size[:2]/2-.008
    table=box(*low,*high)
    robot_base=raw.sim.data.get_body_xpos('robot0_base').copy() if hasattr(raw,'sim') else None
    specs=[]
    for slot,kind in enumerate(kinds,1):
        vertices=vertices_fn('hook_'+kind)
        specs.append((f'hook_obj{slot}_1',kind,vertices,rest_rotation(quats[f'hook_{kind}_1'])))
    # Randomise yaw, then pack complete footprints along the table's Y axis.
    # Instance aliases encode slot identity only; they NEVER encode true class.
    for attempt in range(20000):
        prepared=[]
        for name,kind,v,base in specs:
            theta=float(rng.uniform(-yaw_deg,yaw_deg))
            rot=Rotation.from_euler('z',theta,degrees=True).as_matrix()@base
            vv=v@rot.T; mn=vv.min(0); mx=vv.max(0)
            prepared.append((name,kind,v,rot,theta,mn,mx))
        widths=np.array([p[6][1]-p[5][1] for p in prepared])
        slack=high[1]-low[1]-widths.sum()-gap*3
        if slack<0:continue
        spaces=rng.dirichlet(np.ones(5))*slack
        cursor=low[1]+spaces[0];chosen={}
        for index,(name,kind,v,rot,theta,mn,mx) in enumerate(prepared):
            # Repeated categories must not be tied to their old category lane.
            # Sample across the safe table width; actual robot IK filters reach.
            px=float(rng.uniform(low[0]-mn[0],high[0]-mx[0]))
            px=float(np.clip(px+rng.uniform(-translation_m,translation_m),low[0]-mn[0],high[0]-mx[0]))
            pos=np.array([px,cursor-mn[1],center[2]-mn[2]+.00015])
            footprint=hull(v,rot,pos)
            if robot_base is not None and footprint.intersects(Point(*robot_base[:2]).buffer(.13)):break
            profile=PROFILES[int(kind[-1])-1]
            grasp=pos+rot@np.array([profile['local_x'],profile['local_y'],-profile['depth']])
            corridor=Point(*grasp[:2]).buffer(.055)
            if reach_test is not None and not reach_test(grasp,theta,kind):break
            if any(corridor.intersects(s['footprint'].buffer(.01)) or
                   s['corridor'].intersects(footprint.buffer(.01)) for s in chosen.values()):break
            chosen[name]={'kind':kind,'position':pos,'rotation':rot,'yaw_delta_deg':theta,
                          'vertices':v,'footprint':footprint,'corridor':corridor}
            cursor+=mx[1]-mn[1]+gap+spaces[index+1]
        if len(chosen)==4:return chosen
    raise RuntimeError('This composition cannot be safely packed on the original table; choose another layout')

def random_composition(seed):
    """Reproducible class sampling; reject compositions too wide for table."""
    rng=np.random.default_rng(seed)
    width={'obj1':.080,'obj2':.102,'obj3':.251,'obj4':.150}
    for _ in range(1000):
        kinds=list(rng.choice(KINDS,4))
        if sum(width[k] for k in kinds)+.025*3<.75:return kinds
    raise RuntimeError('Could not sample feasible composition')

def randomize_scene(env,kinds,seed=0,translation_m=.035,yaw_deg=10.):
    raw=env.env;sim=raw.sim
    from types import SimpleNamespace
    from kinematic_screen import PandaIK
    robot=raw.robots[0];tcp=robot.eef_site_id
    if isinstance(tcp,dict):tcp=next(iter(tcp.values()))
    ik=PandaIK(SimpleNamespace(raw=raw,sim=sim,tcp=tcp))
    home_r=sim.data.site_xmat[tcp].reshape(3,3).copy()
    high=max(float(sim.data.site_xpos[tcp][2]),float(raw.table_offset[2])+.28)
    profiles=grasp_profiles();checks=0
    base=sim.data.get_body_xpos('robot0_base')
    def reachable(grasp,yaw,kind):
        nonlocal checks
        checks+=1
        if np.linalg.norm(grasp[:2]-base[:2])>.76:return False
        rr=Rotation.from_euler('z',yaw+profiles[int(kind[-1])-1].get('grasp_yaw',0),degrees=True).as_matrix()@home_r
        q=ik.solve(grasp,rr,attempts=3)
        if q is None:return False
        return ik.solve(np.r_[grasp[:2],max(high,grasp[2]+.18)],rr,seed=q,attempts=2) is not None
    chosen=sample_layout(raw,kinds,seed,translation_m,yaw_deg,reach_test=reachable)
    for name,v in chosen.items():
        obj=raw.get_object(name)
        quat=Rotation.from_matrix(v['rotation']).as_quat()[[3,0,1,2]]
        sim.data.set_joint_qpos(obj.joints[-1],np.r_[v['position'],quat])
        raw._zero_object_velocity(obj)
    sim.forward()
    before={n:sim.data.get_body_xpos(raw.get_object(n).root_body).copy() for n in chosen}
    for _ in range(30):env.step(np.r_[np.zeros(6),-1.])
    footprints={};drifts={}
    c=np.asarray(raw.table_offset);s=np.asarray(raw.table_full_size)
    table=box(c[0]-s[0]/2+.008,c[1]-s[1]/2+.008,c[0]+s[0]/2-.008,c[1]+s[1]/2-.008)
    for name,v in chosen.items():
        bid=sim.model.body_name2id(raw.get_object(name).root_body)
        pos=sim.data.body_xpos[bid].copy();rot=sim.data.body_xmat[bid].reshape(3,3)
        drifts[name]=float(np.linalg.norm(pos-before[name]))
        footprints[name]=hull(v['vertices'],rot,pos)
        if drifts[name]>.004 or not table.buffer(.0002).covers(footprints[name]):
            raise RuntimeError(f'Unsafe settled pose: {name}, drift={drifts[name]}')
    min_gap=min(footprints[a].distance(footprints[b]) for a in chosen for b in chosen if a<b)
    if min_gap<.023:raise RuntimeError('Settled CAD gap less than 23mm')
    for contact in sim.data.contact[:sim.data.ncon]:
        names=[sim.model.geom_id2name(int(g)) or '' for g in (contact.geom1,contact.geom2)]
        if contact.dist<-.0001 and all(n.startswith('hook_obj') for n in names) and names[0].split('_1_')[0]!=names[1].split('_1_')[0]:
            raise RuntimeError('Inter-object collision at reset')
    return {'classes_in_table_order':kinds,'seed':seed,'yaw_limit_deg':yaw_deg,
            'grasp_and_pregrasp_ik_verified_for_all_instances':True,'reach_checks':checks,
            'x_sampling':'full CAD-safe table width; robot reach and pedestal filtered',
            'minimum_cad_gap_m':min_gap,'settle_drift_m':drifts,
            'instances_for_audit_only':{n:{'kind':v['kind'],'position':v['position'].tolist(),
                'yaw_delta_deg':v['yaw_delta_deg']} for n,v in chosen.items()},'pose_writes':'reset only'}

def write_episode_bddl(path,kinds):
    objects='\n'.join('    '+' '.join(f'hook_obj{j}_1' for j,k in enumerate(kinds,1) if k==kind)+f' - hook_obj_{kind[-1]}'
                      for kind in KINDS if kind in kinds)
    path.write_text('''(define (problem Hook_Rack_Manipulation)
  (:domain robosuite)
  (:language Pick up a requested metal part then face an empty slot of its rack)
  (:regions)
  (:fixtures main_table - table
    hook_wall_1 - hook_wall
    hook_shell1_1 - hook_shell_1
    hook_shell2_1 - hook_shell_2
    hook_shell3_1 - hook_shell_3)
  (:objects
'''+objects+''')
  (:obj_of_interest hook_obj1_1 hook_shell1_1)
  (:init)
  (:goal (And (True hook_obj1_1))))
''',encoding='utf-8')
