"""Stage 3 and multi-cycle supervisor. Simulation-state expert, not a VLA.

No live pose assignments, constraints, geometry or object-mass modifications.
Vision supplies class/initial grasp; measured simulated poses close insertion.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
from grasp_part import HookrackSession


class HangingSession(HookrackSession):
    def __init__(self,**kwargs):
        self.completed={};self.insertion=False;self.releasing=False;self.cycles=[];self.preset_anchors={}
        super().__init__(**kwargs)
        self.report.update(task='recognize, grasp, face rack, insert, release, verify, return; repeat',
                           cycles=self.cycles,train_ready=False)
        try:
            for entry in getattr(self.raw,'preset_occupied',[]):
                if not entry.get('free_dynamic_body'):continue
                bid=self.sim.model.body_name2id(entry['body_name'])
                self.preset_anchors[entry['name']]=dict(bid=bid,cap=self.sim.data.body_xpos[bid].copy(),layer=entry['layer'])
            self.audit_presets()
        except Exception as error:
            self.report.update(success=False,error=repr(error));self.close();raise

    def audit_presets(self):
        for name,entry in self.preset_anchors.items():
            bid=entry['bid'];cap=self.sim.data.body_xpos[bid]
            if np.linalg.norm(cap-entry['cap'])>.008 or self.sim.data.body_xmat[bid].reshape(3,3)[2,2]<.97:
                raise RuntimeError('Pre-existing free workpiece moved/dropped: '+name)
            support=False
            for c in self.sim.data.contact[:self.sim.data.ncon]:
                a,b=(self.sim.model.geom_id2name(int(g)) or '' for g in (c.geom1,c.geom2))
                for own,other in ((a,b),(b,a)):
                    if own.startswith(name+'_') and 'cap_collision' in own and other.startswith(f'hook_shell{entry["layer"]}_1_') and c.dist<.0001:
                        support=True
            if not support:raise RuntimeError('Pre-existing free workpiece lacks round-head rack support: '+name)

    def perceive(self):
        from perceive_parts import capture_views,locate_parts
        from scipy.optimize import linear_sum_assignment
        folder=self.out/f'table_scan_{len(self.cycles):02d}'
        views=capture_views(self.raw,folder)
        self.estimates=locate_parts(self.raw,views,folder)
        if any(min(e['visible_pixels'].values())<25 for e in self.estimates):
            raise RuntimeError('Overview does not show every detected part in both cameras')
        # True state below is a rejection-only audit, never supplies missing
        # detections, their classes or their approach targets.
        aliases=[a for a in self.initial if a not in self.completed]
        if len(self.estimates)!=len(aliases):
            raise RuntimeError('Visual inventory incomplete; refusing to declare table empty')
        self.audit_alias={};audit=[]
        if aliases:
            costs=np.array([[np.linalg.norm(np.array(e['cap_position_m'])-self.sim.data.body_xpos[self.ids[a][0]])
                             for a in aliases] for e in self.estimates])
            rr,cc=linear_sum_assignment(costs)
            for i,j in zip(rr,cc):
                est=self.estimates[i];alias=aliases[j]
                if costs[i,j]>.008 or est['kind']!=self.kinds[int(alias[8])-1]:
                    raise RuntimeError('Reject-only perception audit failed')
                self.audit_alias[est['instance_id']]=alias
                audit.append(dict(instance=est['instance_id'],alias=alias,error_m=float(costs[i,j])))
        self.report.setdefault('inventory_history',[]).append(dict(count=len(self.estimates),
            kinds=[e['kind'] for e in self.estimates],audit=audit))
        return self.estimates

    def step(self,action,phase):
        # Base safety checks remain active. Intentional rack contact and
        # finger release are enabled only in the explicitly delimited stages.
        super().step(action,phase)
        self.audit_presets()
        for alias,entry in self.completed.items():
            cap=self.sim.data.body_xpos[self.ids[alias][0]]
            if np.linalg.norm(cap-entry['cap'])>.008:
                raise RuntimeError(f'Previously hung part moved/dropped: {alias}')

    def return_to_initial_pose(self):
        if not getattr(self,'completed',{}):return super().return_to_initial_pose()
        if self.held:raise RuntimeError('Release and verify the part before returning home')
        from joint_motion import JointMotion
        motion=JointMotion(self,empty=True)
        motion.forced_goal_joint=self.home_q.copy()
        try:motion.move(self.home_p,self.home_r,'empty_hand_home')
        finally:motion.restore()
        return {'tcp_m':self.tcp_pose()[0].tolist()}

    def face_rack(self,kind):
        root=self.out;self.out=root/f'cycle_{len(self.cycles):02d}'
        self.out.mkdir(exist_ok=True)
        try:
            result=super().face_rack(kind)
            result['held_alias']=self.held['alias']
            return result
        finally:self.out=root

    def grasp_part(self,kind):
        result=super().grasp_part(kind)
        row=dict(obj1='top',obj2='top',obj3='middle',obj4='bottom')[kind]
        self.report['prompt']=f'Pick up one {kind} from the table, hang it by its round head in an empty slot of the {row} rack, release it, and return to the table overview pose.'
        return result

    def select_placement(self,kind,layer,frames,preferred):
        from face_rack import scan_slots
        from joint_motion import JointMotion
        origin_p,origin_r=self.tcp_pose();rejected=[]
        scans=[(0.,0.),(.14,0.),(-.14,0.),(.24,0.),(-.24,0.)]
        if kind=='obj4':scans=[(0.,0.),(0.,.30),(-.14,.30),(-.28,.30),(.14,.30)]+scans[1:]
        for attempt,(dx,dz) in enumerate(scans):
            if attempt:
                before=len(self.actions);motion=JointMotion(self)
                try:motion.move(origin_p+[dx,0,dz],origin_r,f'placement_lateral_scan_{attempt}')
                except RuntimeError:
                    if len(self.actions)!=before:raise
                    continue
                finally:motion.restore()
            folder=self.out/f'cycle_{len(self.cycles):02d}'/f'placement_scan_{attempt}'
            scan=scan_slots(self,layer,folder)
            radius={1:.04,2:.04,3:.075}[layer]
            xmin=-.95 if kind=='obj4' else -.65
            choices=[s for s in scan['slots'] if s['state']=='empty' and xmin<s['position_m'][0]<.25]
            choices=[s for s in choices if all(abs(s['position_m'][0]-o['position_m'][0])>radius*2+.008
                for o in scan['slots'] if o['state']=='occupied')]
            if kind=='obj4':
                center=np.asarray(self.raw.table_offset);size=np.asarray(self.raw.table_full_size)
                left=center[0]-size[0]/2-.075;right=center[0]+size[0]/2+.075
                if attempt<5:choices=[s for s in choices if not left<s['position_m'][0]<right]
                choices.sort(key=lambda s:(left<s['position_m'][0]<right,abs(s['position_m'][0]-preferred[0])))
            else:choices.sort(key=lambda s:abs(s['position_m'][0]-preferred[0]))
            self.refresh_held_transform();screen=JointMotion(self,placement_layer=layer);screen.refresh()
            for choice in choices:
                candidate=np.array(choice['position_m']);seed=self.sim.data.qpos[screen.qidx].copy()
                valid=True
                for frame in frames:
                    objr=np.array(frame['object_rotation']);rr=objr@self.held['object_rotation_in_tcp'].T
                    cap=candidate+frame['relative_cap_m'];pp=cap-rr@self.held['cap_in_tcp']
                    found=None
                    for trial in range(32 if kind=='obj4' else 8):
                        qs=seed if trial==0 else (screen.rng.uniform(screen.limits[:,0],screen.limits[:,1]) if kind=='obj4' and trial>=6 else np.clip(seed+screen.rng.normal(0,.8,7),screen.limits[:,0],screen.limits[:,1]))
                        q=screen.ik.solve(pp,rr,seed=qs,attempts=2)
                        if q is not None and screen.valid(q):found=q;break
                    if found is None:
                        valid=False;rejected.append(dict(scan=attempt,slot=choice['index'],phase=frame['phase'],reason=getattr(screen,'last_invalid',None)));break
                    seed=found
                if valid:
                    self.report['placement_slot']=choice
                    self.report['placement_rejected_slots']=rejected
                    return candidate
        self.report['placement_rejected_slots']=rejected
        raise RuntimeError('No physically usable observed empty slot: '+json.dumps(rejected))

    @staticmethod
    def placement_variants(cap,rotation):
        """Bounded corrections for compliance; no fixture/geometry changes."""
        yield np.asarray(cap),np.asarray(rotation),dict(delta_m=[0,0,0],yaw_deg=0.,pitch_deg=0.)
        shifts=([0,.004,0],[0,.008,0],[0,0,.004],[.004,0,0],[-.004,0,0],[0,-.004,0],[0,.004,.004])
        for delta in shifts:
            yield np.asarray(cap)+delta,np.asarray(rotation),dict(delta_m=list(delta),yaw_deg=0.,pitch_deg=0.)
        for yaw in (3.,-3.,6.,-6.,10.,-10.):
            rot=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rotation
            for delta in ([0,0,0],[0,.006,0],[0,0,.004]):
                yield np.asarray(cap)+delta,rot,dict(delta_m=list(delta),yaw_deg=yaw,pitch_deg=0.)
        for pitch in (3.,-3.,6.,-6.):
            rot=Rotation.from_euler('x',pitch,degrees=True).as_matrix()@rotation
            for delta in ([0,0,0],[0,.006,0]):
                yield np.asarray(cap)+delta,rot,dict(delta_m=list(delta),yaw_deg=0.,pitch_deg=pitch)

    def move_object_waypoint(self,cap,rotation,phase,layer):
        from joint_motion import JointMotion
        self.refresh_held_transform();failures=[]
        motion=getattr(self,'_placement_motion',None)
        if motion is None or motion.alias!=self.held['alias'] or motion.placement_layer!=layer:
            if motion is not None:motion.restore()
            motion=JointMotion(self,placement_layer=layer)
            self._placement_motion=motion
            self.report['placement_planner_builds']=self.report.get('placement_planner_builds',0)+1
        variants=list(self.placement_variants(cap,rotation))
        if self.held['kind']=='obj4':
            # Round-head bearing has no required yaw. Search additional wrist
            # headings when the long-piece grasp would put link6 in the table.
            for yaw in (-15.,15.,-25.,25.,-40.,40.,-60.,60.,-90.,90.,-120.,120.,180.):
                rr=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@rotation
                for delta in ([0,0,0],[0,0,.004],[0,.004,.004],[0,0,.008]):
                    variants.append((np.asarray(cap)+delta,rr,dict(delta_m=list(delta),yaw_deg=yaw,pitch_deg=0.)))
            for pitch in (-6.,-10.,6.,10.):
                rr=Rotation.from_euler('x',pitch,degrees=True).as_matrix()@rotation
                for delta in ([0,0,.004],[0,0,.008]):
                    variants.append((np.asarray(cap)+delta,rr,dict(delta_m=list(delta),yaw_deg=0.,pitch_deg=pitch)))
            variants.append((np.asarray(cap)+[0,0,.008],np.asarray(rotation),
                dict(delta_m=[0,0,.008],yaw_deg=0.,pitch_deg=0.)))
        for pp_obj,rr_obj,change in variants:
            rr=rr_obj@self.held['object_rotation_in_tcp'].T
            pp=pp_obj-rr@self.held['cap_in_tcp']
            motion.restore();motion.refresh()
            motion.rng=np.random.default_rng(self.seed+9401)
            motion.ik.initial=self.sim.data.qpos[motion.qidx].copy()
            motion.integral[:]=0.
            for key in ('gravity_start','gravity_goal','gravity_arc','last_invalid'):
                if hasattr(motion,key):delattr(motion,key)
            # Cheap geometric/IK screening before spending an RRT budget.
            q0=self.sim.data.qpos[motion.qidx].copy();feasible=False
            long_part=self.held['kind']=='obj4'
            for trial in range(24 if long_part else 8):
                seed=q0 if trial==0 else (motion.rng.uniform(motion.limits[:,0],motion.limits[:,1]) if long_part and trial>=6 else np.clip(q0+motion.rng.normal(0,.8,7),motion.limits[:,0],motion.limits[:,1]))
                q=motion.ik.solve(pp,rr,seed=seed,attempts=2)
                if q is not None and motion.valid(q):feasible=True;break
            if not feasible:continue
            before=len(self.actions)
            try:motion.move(pp,rr,phase)
            except RuntimeError as error:
                if len(self.actions)!=before:raise
                failures.append(str(error));continue
            finally:motion.restore()
            self.report.setdefault('online_placement_adjustments',[]).append(dict(phase=phase,**change,
                reference_cap_m=np.asarray(cap).tolist(),command_cap_m=pp_obj.tolist(),rejected_plans=failures))
            return
        raise RuntimeError('No collision-safe bounded placement correction in '+phase+': '+str(failures[-3:]))

    def close(self):
        motion=getattr(self,'_placement_motion',None)
        if motion is not None:motion.restore();self._placement_motion=None
        super().close()

    def hang_part(self,kind):
        if not self.held or self.held['kind']!=kind:raise ValueError('Wrong/no held part')
        facing=self.report.get('face_rack',{})
        if not facing.get('success') or facing['kind']!=kind or facing.get('held_alias')!=self.held['alias']:raise RuntimeError('Run face_rack for the currently held instance first')
        from joint_motion import JointMotion
        alias=self.held['alias'];layer=facing['layer']
        slot=np.array(facing['selected_slot']['position_m'],float)
        profile=self.profiles[int(kind[-1])-1]
        goal=slot+np.array([profile.get('x_offset',0),profile.get('y_offset',0),profile.get('z_offset',0)])
        goal_r=Rotation.from_euler('z',profile['yaw'],degrees=True).as_matrix()
        start=len(self.traces)
        self.refresh_held_transform()
        # The old validated recipes supply upright yaw and over-tip approach;
        # transform these goals through the current measured grasp, not time.
        references=json.loads((Path(__file__).parent/'hanging_references.json').read_text())[kind]['frames']
        pre=[f for f in references if f['phase'] in ('prehang','align_for_slot','enter_bottom_aisle')][-1:]
        frames=pre+[f for f in references if f['phase']=='insert'][1:]+[f for f in references if f['phase']=='seat'][1:]
        # Reject visually empty but mechanically unusable cells before moving.
        slot=self.select_placement(kind,layer,frames,slot)
        goal=slot+[profile.get('x_offset',0),profile.get('y_offset',0),profile.get('z_offset',0)]
        self.insertion=True
        try:
            if kind=='obj4':
                # The long body must clear both the tabletop and the middle
                # comb before lowering. This is the old successful high/side
                # route, parameterized by the newly observed empty cell.
                cap,current_r=self.refresh_held_transform()
                route=[(np.r_[cap[:2],1.55],current_r,'raise_flat_clear'),
                       (np.array([.15,.10,1.55]),goal_r,'orient_high'),
                       (np.array([slot[0],.15,1.55]),goal_r,'side_high'),
                       (np.array([slot[0],.15,1.13]),goal_r,'lower_front_intermediate'),
                       (np.array([slot[0],.27,1.13]),goal_r,'move_behind_table')]
                for cap,objr,label in route:
                    self.refresh_held_transform()
                    rr=objr@self.held['object_rotation_in_tcp'].T
                    pp=cap-rr@self.held['cap_in_tcp']
                    motion=JointMotion(self,placement_layer=layer)
                    try:motion.move(pp,rr,'hang_'+label)
                    finally:motion.restore()
            for i,frame in enumerate(frames):
                self.move_object_waypoint(slot+frame['relative_cap_m'],np.array(frame['object_rotation']),
                    f"hang_reference_{i:02d}_{frame['phase']}",layer)
            # The last measured reference already seats the cap. Do not switch
            # back to unconstrained Cartesian OSC next to the comb: that can
            # drift an elbow into an adjacent rack even if TCP is stationary.
            self.releasing=True
            hold=JointMotion(self,placement_layer=layer)
            hold.activate()
            try:
                for _ in range(60):
                    self.stage2_joint_target=hold.target.copy()
                    self.step(np.r_[np.zeros(6),-1.],'hang_release')
            finally:hold.restore()
            self.held=None
            p,r=self.tcp_pose()
            withdraw=JointMotion(self,empty=True,placement_layer=layer)
            try:
                withdraw.move(p+[0,-.18,0],r,'hang_withdraw')
                for _ in range(100):
                    self.stage2_joint_target=withdraw.target.copy()
                    self.step(np.r_[np.zeros(6),-1.],'hang_verify')
            finally:withdraw.restore()
            metrics=self.verify_hanging(alias,layer,goal,self.traces[-100:])
            metrics['stage3_max_penetration_m']=max((max(0,-c['distance_m'])
                for row in self.traces[start:] for c in row['contacts']),default=0.)
            metrics['success']=metrics['success'] and metrics['stage3_max_penetration_m']<.001
            self.report['hang_part']=metrics
            if not metrics['success']:raise RuntimeError('Physical hanging verification failed: '+json.dumps(metrics))
            self.completed[alias]={'cap':self.sim.data.body_xpos[self.ids[alias][0]].copy(),'kind':kind,'layer':layer}
            # Non-target stationarity baseline is now its physically seated
            # position; no state writes or body relocation are involved.
            self.initial[alias]=self.completed[alias]['cap'].copy()
            self.target_alias=None
            import copy
            self.cycles.append(dict(kind=kind,alias=alias,hang=metrics,stage3_steps=len(self.traces)-start,
                grasp=copy.deepcopy(self.report['grasp']),face=copy.deepcopy(facing),
                placement_slot=copy.deepcopy(self.report['placement_slot']),prompt=self.report['prompt']))
            self.return_to_initial_pose()
            self.cycles[-1]['returned_home']=True
            (self.out/'progress.json').write_text(json.dumps(dict(completed=self.cycles,
                table_empty_visually_verified=False,complete=False),indent=2),encoding='utf-8')
            if hasattr(self,'estimates'):del self.estimates
            self.report['success']=True
            return metrics
        finally:self.insertion=False;self.releasing=False

    def move_cap(self,target,rotation,phase,n):
        # Same closed-loop object-space law as the old validated expert;
        # current cap feedback corrects compliance, rather than replaying time.
        target=np.asarray(target);cap0,r0=self.refresh_held_transform()
        from scipy.spatial.transform import Slerp
        interp=Slerp([0,1],Rotation.from_matrix(np.stack([r0,rotation])))
        for i in range(n):
            t=min(1.,(i+1)/(n*.70))
            cap=cap0+(target-cap0)*t;rot=interp(t).as_matrix()
            cp,cr=self.refresh_held_transform()
            ep=cap-cp;er=Rotation.from_matrix(rot@cr.T).as_rotvec()
            self.position_integral=np.clip(self.position_integral+ep*.10/20,-.015,.015)
            self.rotation_integral=np.clip(self.rotation_integral+er*.12/20,-.05,.05)
            dp=ep+self.position_integral;dr=er*.8+self.rotation_integral
            dp*=min(1.,.010/max(np.linalg.norm(dp),1e-9))
            dr*=min(1.,.025/max(np.linalg.norm(dr),1e-9))
            self.step(np.r_[np.clip(np.r_[dp,dr]/self.scale,-1,1),1.],phase)
        bid=self.ids[self.held['alias']][0]
        error=float(np.linalg.norm(self.sim.data.body_xpos[bid]-target))
        if error>.012:raise RuntimeError(f'Cap tracking failed {phase}: {error}')

    def verify_hanging(self,alias,layer,goal,rows):
        caps=np.array([r['target_cap_m'] for r in rows]);prefix=f'hook_shell{layer}_1_'
        contact_frames=[];penetrations=[];cap_load=[];shaft_load=[]
        for row in rows:
            contacts=row['contacts'];penetrations.extend(max(0,-c['distance_m']) for c in contacts)
            valid=bool(contacts) and all(c['other'].startswith(prefix) and ('tooth' in c['other'] or 'cell_' in c['other']) for c in contacts)
            # A neck can brush a slot side at tiny force without supporting
            # the part. Check actual vertical load, not merely geom names.
            weight=row.get('target_mass_kg',float('inf'))*9.81
            bearing=[c for c in contacts if 'cap_collision' in c.get('object_geom','')
                     and c.get('normal_vertical',0)>.75]
            other=[c for c in contacts if 'cap_collision' not in c.get('object_geom','')]
            head=sum(c.get('vertical_force_N',0) for c in bearing)/weight
            shaft=sum(c.get('vertical_force_N',float('inf')) for c in other)/weight
            lateral=sum(c['force_N'] for c in other)/weight
            valid=valid and head>=.90 and shaft<.05 and lateral<.25
            cap_load.append(head);shaft_load.append(shaft)
            contact_frames.append(valid)
        bid=self.ids[alias][0];upright=float(self.sim.data.body_xmat[bid].reshape(3,3)[2,2])
        drift=float(np.max(np.linalg.norm(caps-caps[0],axis=1)))
        err=float(np.linalg.norm(caps[-1]-goal))
        penetration=max(penetrations,default=0)
        linear=max((r.get('target_linear_speed_m_s',float('inf')) for r in rows),default=float('inf'))
        angular=max((r.get('target_angular_speed_rad_s',float('inf')) for r in rows),default=float('inf'))
        good=bool(all(contact_frames) and upright>.97 and drift<.003 and err<.012 and penetration<.001 and linear<.01 and angular<.05)
        return dict(success=good,alias=alias,layer=layer,goal_m=goal.tolist(),cap_m=caps[-1].tolist(),
                    valid_contact_frames=sum(contact_frames),frames=len(rows),duration_s=len(rows)/20,
                    upright=upright,drift_m=drift,position_error_m=err,max_penetration_m=penetration,
                    max_linear_speed_m_s=linear,max_angular_speed_rad_s=angular,
                    min_cap_vertical_load_fraction=min(cap_load,default=0),max_shaft_vertical_load_fraction=max(shaft_load,default=float('inf')),
                    gripper_released=True,free_dynamic_body=True)

    def run_all(self):
        for _ in range(5):
            inventory=self.perceive()
            if not inventory:
                self.report.update(success=True,table_empty_visually_verified=True)
                (self.out/'progress.json').write_text(json.dumps(dict(completed=self.cycles,
                    table_empty_visually_verified=True,complete=True),indent=2),encoding='utf-8')
                return self.cycles
            # Select from visual inventory only; repeat recognition each cycle.
            kind=sorted(inventory,key=lambda e:e['cap_position_m'][1])[0]['kind']
            self.grasp_part(kind);self.face_rack(kind);self.hang_part(kind)
        raise RuntimeError('Cycle limit reached; table not verified empty')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--part',choices=['obj1','obj2','obj3','obj4','random'],default='random')
    p.add_argument('--all',action='store_true');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--layout');p.add_argument('--occupied',default='')
    p.add_argument('--render-backend',default='egl');p.add_argument('--video-stride',type=int,default=8)
    a=p.parse_args()
    with HangingSession(benchmark_root=a.benchmark_root,output=a.output,seed=a.seed,
        layout=a.layout.split(',') if a.layout else None,occupied=a.occupied,
        occupied_layer={'obj1':1,'obj2':1,'obj3':2,'obj4':3}.get(a.part,1),
        render_backend=a.render_backend,video_stride=a.video_stride) as arm:
        if a.all:arm.run_all()
        else:
            inventory=arm.perceive()
            kind=inventory[int(np.random.default_rng(a.seed+2909).integers(len(inventory)))]['kind'] if a.part=='random' else a.part
            arm.grasp_part(kind);arm.face_rack(kind);arm.hang_part(kind)
    return 0

if __name__=='__main__':raise SystemExit(main())
