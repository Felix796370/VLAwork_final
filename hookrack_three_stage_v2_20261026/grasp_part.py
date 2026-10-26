#!/usr/bin/env python3
"""Persistent action-only expert: random repeated instances -> grasp -> rack view.

This is a simulator expert, not Pi0.5 inference. No pose writes after reset.
"""
from __future__ import annotations
import argparse,importlib,importlib.metadata,json,os,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import linear_sum_assignment
CODE_FILES=('grasp_part.py','randomize_scene.py','randomize_scene_base.py','perceive_parts.py',
            'perceive_parts_base.py','face_rack.py','occupied_scene.py','kinematic_screen.py','joint_motion.py','hang_part.py','hanging_references.json')

def code_hashes():
    return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in CODE_FILES}

def close_on_initialization_error(function):
    """Preserve failure evidence and release an initialized renderer/env."""
    from functools import wraps
    @wraps(function)
    def wrapped(self,*args,**kwargs):
        try:return function(self,*args,**kwargs)
        except Exception as error:
            if hasattr(self,'report'):
                self.report.update(success=False,error=repr(error))
                try:self.close()
                except Exception as cleanup:
                    self.report['cleanup_error']=repr(cleanup)
                    (self.out/'report.json').write_text(json.dumps(self.report,indent=2),encoding='utf-8')
            raise
    return wrapped

class HookrackSession:
    @close_on_initialization_error
    def __init__(self,*,benchmark_root,output,seed=0,layout=None,yaw_deg=10.,
                 translation_m=.035,render_backend=None,video_stride=8,occupied='',occupied_layer=1):
        self.out=Path(output).expanduser().resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.seed=seed;self.video_stride=video_stride;self.env=None;self.writer=None
        self.actions=[];self.states=[];self.obs_states=[];self.traces=[];self.held=None
        self.position_integral=np.zeros(3);self.rotation_integral=np.zeros(3)
        self.stage2_joint_target=None;self.joint_targets=[]
        self.report={'success':False,'train_ready':False,'server_tested':False,'seed':seed,
                     'task':'grasp and face correct rack, no insertion or release',
                     'perception':'Stage 1: calibrated RGB-D + known CAD. Stage 2: privileged simulator held-object transform feedback.',
                     'code_sha256':code_hashes()}
        import shutil
        snapshot=self.out/'source_snapshot';snapshot.mkdir()
        for name in CODE_FILES:shutil.copy2(Path(__file__).parent/name,snapshot/name)
        backend=render_backend or ('glfw' if os.name=='nt' else 'egl')
        os.environ['MUJOCO_GL']=backend
        if backend in ('egl','osmesa'):os.environ['PYOPENGL_PLATFORM']=backend
        root=Path(benchmark_root).expanduser().resolve();sys.path.insert(0,str(root/'scripts'))
        from collect_expert_hanging import PROFILES
        from randomize_scene import random_composition,randomize_scene,write_episode_bddl,KINDS,grasp_profiles
        from perceive_parts import configure_initial_wrist
        self.profiles=grasp_profiles()
        self.kinds=list(layout) if layout else random_composition(seed)
        if len(self.kinds)!=4 or any(k not in KINDS for k in self.kinds):raise ValueError('Invalid four-part layout')
        self.report['runtime_versions']={n:importlib.metadata.version(n) for n in ('mujoco','robosuite','numpy','scipy')}
        bddl=self.out/'episode.bddl';write_episode_bddl(bddl,self.kinds)
        wrapper=importlib.import_module('liberopro.liberopro.envs.env_wrapper')
        from occupied_scene import install_runtime_problem
        previous=install_runtime_problem(occupied,occupied_layer)
        try:
            self.env=wrapper.ControlEnv(bddl_file_name=str(bddl),robots=['Panda'],use_camera_obs=False,
                has_renderer=False,has_offscreen_renderer=True,camera_names=['agentview','robot0_eye_in_hand'],
                camera_heights=256,camera_widths=256,control_freq=20,horizon=30000,ignore_done=True,hard_reset=False)
        finally:
            domain=importlib.import_module('liberopro.liberopro.envs.bddl_base_domain')
            domain.TASK_MAPPING['hook_rack_manipulation']=previous
        self.env.seed(seed);self.env.reset();self.raw=self.env.env;self.sim=self.raw.sim
        self.report['mujoco_arena_bytes']=int(self.sim.model._model.narena)
        if occupied:
            self.report['preset_occupancy_for_audit_only']=self.raw.preset_occupied
        self.report['reset']=randomize_scene(self.env,self.kinds,seed,translation_m,yaw_deg)
        self.report['camera_profile']=configure_initial_wrist(self.raw)
        self.configure_gripper_preload()
        robot=self.raw.robots[0];self.tcp=robot.eef_site_id
        if isinstance(self.tcp,dict):self.tcp=next(iter(self.tcp.values()))
        ctrl=getattr(robot,'controller',None)
        if ctrl is None:ctrl=robot.part_controllers['right']
        self.controller=ctrl
        # robosuite 1.5 defaults to base-frame commands. Match the explicit
        # world/achieved delta semantics used by this expert across versions.
        if hasattr(ctrl,'input_ref_frame'):ctrl.input_ref_frame='world'
        if hasattr(ctrl,'_goal_update_mode'):ctrl._goal_update_mode='achieved'
        ctrl.update(force=True);ctrl.reset_goal()
        self.report['osc_convention']={'reference_frame':'world','goal_update':'achieved','input_type':'delta'}
        if getattr(ctrl,'input_ref_frame','world')!='world' or getattr(ctrl,'input_type','delta')!='delta':
            raise RuntimeError('Requires world-frame delta OSC')
        self.scale=(np.asarray(ctrl.output_max)-np.asarray(ctrl.output_min))/2
        if self.scale.shape!=(6,) or np.any(self.scale<=0):raise RuntimeError('Unsupported OSC scaling')
        if self.sim.model.neq or not np.allclose(self.sim.model.opt.gravity,[0,0,-9.81]):raise RuntimeError('Unexpected weld/gravity')
        self.home_p,self.home_r=self.tcp_pose()
        self.home_q=self.sim.data.qpos[np.asarray(robot._ref_joint_pos_indexes,int)].copy()
        self.initial={};self.ids={}
        for slot in range(1,5):
            name=f'hook_obj{slot}_1';obj=self.raw.get_object(name)
            bid=self.sim.model.body_name2id(obj.root_body)
            self.initial[name]=self.sim.data.body_xpos[bid].copy()
            self.ids[name]=(bid,{self.sim.model.geom_name2id(g) for g in obj.contact_geoms})
        if video_stride>0:
            import imageio.v2 as imageio
            self.writer=imageio.get_writer(self.out/'main_wrist.mp4',fps=20/video_stride)
        self.return_to_initial_pose()

    def tcp_pose(self):
        return self.sim.data.site_xpos[self.tcp].copy(),self.sim.data.site_xmat[self.tcp].reshape(3,3).copy()

    def configure_gripper_preload(self):
        """Command closing force even on thin metal, within original force caps.

        A zero-position servo setpoint provides almost no normal force for a
        1mm sheet. A virtual 20mm closing preload saturates at the unchanged
        finger motor force limit. Physical joint stroke and geometry stay fixed.
        """
        model=self.sim.model
        ids=[i for i in range(model.nu) if (model.actuator_id2name(i) or '').startswith('gripper') and 'finger' in (model.actuator_id2name(i) or '')]
        if len(ids)!=2:raise RuntimeError('Expected two Panda finger position actuators')
        original=model.actuator_ctrlrange[ids].copy()
        force=model.actuator_forcerange[ids].copy()
        for i in ids:
            low,high=model.actuator_ctrlrange[i]
            if abs(low)<1e-8 and high>0:model.actuator_ctrlrange[i,0]=-.020
            elif abs(high)<1e-8 and low<0:model.actuator_ctrlrange[i,1]=.020
            else:raise RuntimeError('Unrecognised Panda finger position-servo range')
        robot=self.raw.robots[0]
        for key,controller in getattr(robot,'part_controllers',{}).items():
            if 'gripper' in key:
                controller.actuator_min=model.actuator_ctrlrange[ids,0].copy()
                controller.actuator_max=model.actuator_ctrlrange[ids,1].copy()
        self.report['gripper_preload']={'closing_preload_m':.020,'original_ctrlrange':original.tolist(),
            'command_ctrlrange':model.actuator_ctrlrange[ids].tolist(),'unchanged_force_limits_N':force.tolist(),
            'physical_joint_stroke_changed':False,'friction_changed':False,'object_mass_changed':False}

    def step(self,action,phase):
        import mujoco
        from robosuite.utils.transform_utils import quat2axisangle
        action=np.asarray(action,np.float32)
        if not np.isfinite(action).all() or abs(action).max()>1.00001:raise RuntimeError('Invalid action')
        obs=self.raw._get_observations(force_update=True)
        self.obs_states.append(np.r_[obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']].astype(np.float32))
        self.states.append(self.sim.get_state().flatten().copy());self.actions.append(action)
        self.joint_targets.append(np.full(7,np.nan) if self.stage2_joint_target is None else self.stage2_joint_target.copy())
        self.env.step(action)
        arena=int(self.sim.data._data.maxuse_arena)
        if arena>self.sim.model._model.narena*.85:
            raise RuntimeError('MuJoCo arena has insufficient safety headroom; reject episode')
        if any(w.number for w in self.sim.data._data.warning):
            raise RuntimeError('MuJoCo numerical/allocation warning; reject episode')
        target=self.held['alias'] if self.held else getattr(self,'target_alias',None)
        contacts=[]
        gids=self.ids[target][1] if target else set()
        for ci,c in enumerate(self.sim.data.contact[:self.sim.data.ncon]):
            a,b=[self.sim.model.geom_id2name(int(g)) or '' for g in (c.geom1,c.geom2)]
            for own,other in ((a,b),(b,a)):
                if own.startswith('hook_obj') and (target is None or not own.startswith(target+'_')) and other.startswith(('robot0','gripper')) and c.dist<-.0001:
                    raise RuntimeError(f'Robot contacted non-target {own}')
                if own.startswith(('robot0','gripper')) and other.startswith(('hook_shell','preset_')) and c.dist<-.0002:
                    raise RuntimeError(f'Robot penetrated rack/pre-existing part: {other}')
            if c.geom1 not in gids and c.geom2 not in gids:continue
            other=int(c.geom2 if c.geom1 in gids else c.geom1)
            force=np.zeros(6);mujoco.mj_contactForce(self.sim.model._model,self.sim.data._data,ci,force)
            if force[0]>.01:
                own=int(c.geom1 if c.geom1 in gids else c.geom2)
                contact={'other':self.sim.model.geom_id2name(other) or '', 'object_geom':self.sim.model.geom_id2name(own) or '', 'force_N':float(force[0]),'distance_m':float(c.dist),'position_m':np.asarray(c.pos).tolist()}
                world_force=np.asarray(c.frame).reshape(3,3).T@force[:3]
                contact['vertical_force_N']=float(abs(world_force[2]))
                contact['normal_vertical']=float(abs(np.asarray(c.frame)[2]))
                contacts.append(contact)
                if contact['distance_m']<-.002:raise RuntimeError(f'Target penetration over 2mm in {phase}: {contact}')
                if contact['other'].startswith('hook_obj'):raise RuntimeError('Target touched another part')
                allowed_rack=getattr(self,'insertion',False) and contact['other'].startswith(f'hook_shell{dict(obj1=1,obj2=1,obj3=2,obj4=3)[self.held["kind"]]}_1_') if self.held else False
                if self.held and not contact['other'].startswith('gripper') and not allowed_rack:raise RuntimeError(f'Held part contacted external support/obstacle in {phase}: {contact}')
        row={'step':len(self.actions)-1,'phase':phase,'tcp_m':self.tcp_pose()[0].tolist(),'contacts':contacts}
        if target:
            row['target_cap_m']=self.sim.data.body_xpos[self.ids[target][0]].copy().tolist()
            obj=self.raw.get_object(target)
            jid=self.sim.model.joint_name2id(obj.joints[-1]);adr=int(self.sim.model.jnt_dofadr[jid])
            velocity=self.sim.data.qvel[adr:adr+6]
            row['target_linear_speed_m_s']=float(np.linalg.norm(velocity[:3]))
            row['target_angular_speed_rad_s']=float(np.linalg.norm(velocity[3:]))
            row['target_mass_kg']=float(self.sim.model._model.body_subtreemass[self.ids[target][0]])
        self.traces.append(row)
        for name,p in self.initial.items():
            if name==target:continue
            if np.linalg.norm(self.sim.data.body_xpos[self.ids[name][0]]-p)>.005:
                raise RuntimeError(f'Non-target moved >5mm: {name}')
        if self.held:
            if self.bilateral(row):self.lost_frames=0
            else:self.lost_frames+=1
            if self.lost_frames>5 and not getattr(self,'releasing',False):raise RuntimeError('Lost bilateral grip for >0.25s')
        if self.writer is not None and len(self.actions)%self.video_stride==0:
            from perceive_parts import CAMERAS
            self.writer.append_data(np.concatenate([self.sim.render(384,384,camera_name=c)[::-1] for c in CAMERAS],1))

    @staticmethod
    def bilateral(row):
        names=[c['other'] for c in row['contacts']]
        return any('finger1' in s or 'leftfinger' in s for s in names) and any('finger2' in s or 'rightfinger' in s for s in names)

    def move(self,target,rot,phase,n=180,grip=None):
        pp,rr=self.tcp_pose();target=np.asarray(target,float)
        slerp=Slerp([0,1],Rotation.from_matrix(np.stack([rr,rot])))
        grip=float(self.held is not None)*2-1 if grip is None else grip
        for i in range(n):
            t=min(1.,(i+1)/(n*.75));p,r=self.tcp_pose()
            loaded=self.held is not None or phase in ('lift','hold')
            ep=pp+(target-pp)*t-p
            er=Rotation.from_matrix(slerp(t).as_matrix()@r.T).as_rotvec()
            if loaded:
                # OSC compensates the robot's gravity, not the held payload.
                # Proprioceptive PI feedback removes payload-dependent static
                # pose error without reading hidden object poses or masses.
                self.position_integral+=ep*.10/20
                self.rotation_integral+=er*.12/20
                for value,limit in ((self.position_integral,.025),(self.rotation_integral,.08)):
                    value*=min(1.,limit/max(np.linalg.norm(value),1e-9))
            dp=ep*(1. if loaded else .65)+(self.position_integral if loaded else 0.)
            dp*=min(1.,.025/max(np.linalg.norm(dp),1e-9))
            dr=er*(.8 if loaded else .5)+(self.rotation_integral if loaded else 0.)
            dr*=min(1.,(.08 if loaded else .025)/max(np.linalg.norm(dr),1e-9))
            self.step(np.r_[np.clip(np.r_[dp,dr]/self.scale,-1,1),grip],phase)
        err=float(np.linalg.norm(self.tcp_pose()[0]-target))
        angle=float(np.rad2deg(Rotation.from_matrix(rot@self.tcp_pose()[1].T).magnitude()))
        print(f'{phase}: tcp_error={err:.5f}m angle={angle:.2f}deg',flush=True)
        if err>.018 or angle>8:raise RuntimeError(f'Pose tracking failed: {phase} ({err:.4f}m, {angle:.1f}deg)')

    def return_to_initial_pose(self):
        """No user arguments. Return to the stored overview pose, maintaining grip."""
        if self.held:
            raise RuntimeError('Overview return while holding requires completing/releasing the part first; grip retained')
        p,r=self.tcp_pose()
        self.move(np.r_[p[:2],max(p[2],self.home_p[2])],r,'home_raise',60)
        self.move(self.home_p,self.home_r,'home_overview',100)
        return {'tcp_m':self.tcp_pose()[0].tolist()}

    def perceive(self):
        from perceive_parts import capture_views,locate_parts
        views=capture_views(self.raw,self.out/'initial')
        self.estimates=locate_parts(self.raw,views,self.out/'initial')
        if any(min(e['visible_pixels'].values())<25 for e in self.estimates):
            raise RuntimeError('Initial wrist/main view does not show every instance')
        # Matching names to estimates below is acceptance auditing only. Object
        # aliases do not supply class/pose to the motion target.
        costs=np.array([[np.linalg.norm(np.asarray(e['cap_position_m'])-p) for p in self.initial.values()] for e in self.estimates])
        rr,cc=linear_sum_assignment(costs);aliases=list(self.initial)
        self.audit_alias={self.estimates[r]['instance_id']:aliases[c] for r,c in zip(rr,cc)}
        audit=[]
        for r,c in zip(rr,cc):
            est=self.estimates[r];alias=aliases[c];true_kind=self.kinds[int(alias[8])-1]
            audit.append({'visual_instance':est['instance_id'],'true_alias':alias,'predicted_kind':est['kind'],'true_kind':true_kind,'position_error_m':float(costs[r,c])})
            if costs[r,c]>.008 or est['kind']!=true_kind:raise RuntimeError('Perception audit rejected, abort expert')
        self.report['perception_audit']=audit
        return self.estimates

    def grasp_part(self,kind):
        """Pick any visible instance of requested class, then retain the grasp."""
        if self.held:raise RuntimeError('Already holding a part')
        self.position_integral[:]=0.;self.rotation_integral[:]=0.
        if not hasattr(self,'estimates'):self.perceive()
        candidates=[e for e in self.estimates if e['kind']==kind]
        if not candidates:raise ValueError(f'{kind} absent; visible classes: {[e["kind"] for e in self.estimates]}')
        rng=np.random.default_rng(self.seed+1907)
        est=candidates[int(rng.integers(len(candidates)))];self.target_alias=self.audit_alias[est['instance_id']]
        profile=self.profiles[int(kind[-1])-1]
        cap=np.asarray(est['cap_position_m']);rot=np.asarray(est['rotation_matrix'])
        grasp=cap+rot@np.array([profile['local_x'],profile['local_y'],-profile['depth']])
        yaw=est['yaw_delta_deg']+profile.get('grasp_yaw',0.)
        rr=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@self.home_r
        p,_=self.tcp_pose();high=max(p[2],self.raw.table_offset[2]+.28,grasp[2]+.18)
        self.move(np.r_[p[:2],high],self.home_r,'raise_clear',70,grip=-1)
        self.move(np.r_[grasp[:2],high],rr,'approach',160,grip=-1)
        self.move(grasp,rr,'descend',130,grip=-1)
        self.move(grasp,rr,'close',35,grip=1)
        lift=grasp+[0,0,profile['lift']]
        self.move(lift,rr,'lift',180,grip=1)
        self.move(lift,rr,'hold',60,grip=1)
        rows=self.traces[-60:];fraction=np.mean([self.bilateral(r) for r in rows])
        alias=self.target_alias
        rise=min(r['target_cap_m'][2] for r in rows)-self.initial[alias][2]
        drift=max(np.linalg.norm(np.asarray(r['target_cap_m'])-rows[0]['target_cap_m']) for r in rows)
        external=sum(any(not c['other'].startswith('gripper') for c in r['contacts']) for r in rows)
        metrics={'success':bool(fraction>=.95 and rise>.10 and drift<.01 and external==0),
                 'kind':kind,'visual_instance':est['instance_id'],'candidate_count':len(candidates),
                 'lift_height_m':float(rise),'bilateral_hold_fraction':float(fraction),'hold_drift_m':float(drift),'external_support_frames':external}
        self.report['grasp']=metrics
        self.report['prompt']=f"Pick up one {kind} from the table and keep holding it while facing an empty slot of the {dict(obj1='top',obj2='top',obj3='middle',obj4='bottom')[kind]} rack."
        if not metrics['success']:raise RuntimeError('Grasp physical acceptance failed')
        # Rigid held transform inferred from RGB-D estimate and commanded lift;
        # no current simulator object pose is used to correct the command.
        p,r=self.tcp_pose();estimated_cap=cap+(p-grasp)
        self.held={'kind':kind,'alias':alias,'cap_in_tcp':r.T@(estimated_cap-p),
                   'object_rotation_in_tcp':r.T@rot,'instance_id':est['instance_id']}
        self.lost_frames=0
        return metrics

    def face_rack(self,kind):
        """Keep the gripper closed and face the rack appropriate to held class."""
        from face_rack import face_rack
        result=face_rack(self,kind)
        self.report['success']=bool(self.report.get('grasp',{}).get('success') and result['success'])
        return result

    def refresh_held_transform(self):
        """Privileged simulation expert feedback; not a real-world estimator.

        Parts can pivot inside the closing fingers. Reusing their tabletop CAD
        rotation would give an incorrect swept volume and camera orientation.
        """
        if not self.held:raise RuntimeError('No held part')
        bid=self.ids[self.held['alias']][0]
        p,r=self.tcp_pose();cap=self.sim.data.body_xpos[bid].copy()
        object_r=self.sim.data.body_xmat[bid].reshape(3,3).copy()
        self.held['cap_in_tcp']=r.T@(cap-p)
        self.held['object_rotation_in_tcp']=r.T@object_r
        self.report['stage2_pose_source']='privileged simulator measured held-object / TCP transform; no pose writes or attachments'
        return cap,object_r

    def close(self):
        self.report['steps']=len(self.actions)
        if hasattr(self,'sim'):
            self.report['mujoco_arena_peak_bytes']=int(self.sim.data._data.maxuse_arena)
        self.report['code_sha256_at_finish']=code_hashes()
        self.report['code_changed_during_episode']=self.report['code_sha256']!=self.report['code_sha256_at_finish']
        if self.writer is not None:self.writer.close();self.writer=None
        (self.out/'report.json').write_text(json.dumps(self.report,indent=2),encoding='utf-8')
        (self.out/'trace.json').write_text(json.dumps(self.traces),encoding='utf-8')
        if self.actions:np.savez_compressed(self.out/'trajectory.npz',actions=self.actions,states_before=self.states,
            observation_state=self.obs_states,stage2_joint_target=self.joint_targets,timestamps=np.arange(len(self.actions))/20)
        if self.env is not None:self.env.close();self.env=None

    def __enter__(self):return self
    def __exit__(self,t,error,tb):
        if error:
            self.report['error']=repr(error);self.report['success']=False
        self.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=['obj1','obj2','obj3','obj4','random'],default='random')
    p.add_argument('--benchmark-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--layout',help='e.g. obj1,obj3,obj3,obj2 in front-to-back table order')
    p.add_argument('--yaw-deg',type=float,default=10.);p.add_argument('--translation-m',type=float,default=.035)
    p.add_argument('--render-backend',choices=['glfw','egl','osmesa']);p.add_argument('--video-stride',type=int,default=8)
    p.add_argument('--grasp-only',action='store_true');p.add_argument('--perception-only',action='store_true')
    p.add_argument('--occupied',default='',help='Test-only reset fixture occupancy, e.g. center or almost-full')
    args=p.parse_args();session=None
    try:
        if args.occupied and args.part=='random':raise ValueError('--occupied requires an explicit --part for its test rack layer')
        session=HookrackSession(benchmark_root=args.benchmark_root,output=args.output,seed=args.seed,
            layout=args.layout.split(',') if args.layout else None,yaw_deg=args.yaw_deg,
            translation_m=args.translation_m,render_backend=args.render_backend,video_stride=args.video_stride,
            occupied=args.occupied,occupied_layer={'obj1':1,'obj2':1,'obj3':2,'obj4':3}.get(args.part,1))
        estimates=session.perceive()
        if args.perception_only:
            session.report['perception_completed']=True;return 0
        present=sorted(set(e['kind'] for e in estimates))
        kind=present[int(np.random.default_rng(args.seed+2909).integers(len(present)))] if args.part=='random' else args.part
        session.grasp_part(kind)
        if not args.grasp_only:session.face_rack(kind)
        session.report['success']=True
        return 0
    except Exception as error:
        print(f'FAILED: {error}',flush=True)
        if session:
            session.report['error']=repr(error);session.report['success']=False
        elif args.output.exists() and not (args.output/'report.json').exists():
            (args.output/'initialization_failure.json').write_text(json.dumps({'success':False,'error':repr(error)},indent=2),encoding='utf-8')
        return 1
    finally:
        if session:
            session.close();print(json.dumps(session.report,indent=2),flush=True)

if __name__=='__main__':raise SystemExit(main())
