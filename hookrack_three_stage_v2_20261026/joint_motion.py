"""Privileged simulator motion planner + physically actuated joint PID.

Collision and IK samples use independent model/data copies. Live joint/object
qpos are never assigned. Actual motion uses the existing Panda torque actuators
and closed gripper. Stage-2 commands are recorded as joint targets, not passed
off as Pi0.5 Cartesian training actions.
"""
import copy,time
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from kinematic_screen import PandaIK

class JointMotion:
    def __init__(self,session,empty=False,placement_layer=None):
        import mujoco
        self.s=session;self.mj=mujoco;self.robot=session.raw.robots[0]
        self.model=copy.copy(session.sim.model._model)
        self.data=mujoco.MjData(self.model)
        self.data.qpos[:]=session.sim.data.qpos;self.data.qvel[:]=0
        self.snapshot=self.data.qpos.copy()
        self.qidx=np.array(self.robot._ref_joint_pos_indexes,int)
        self.vidx=np.array(self.robot._ref_joint_vel_indexes,int)
        self.ik=PandaIK(session)
        self.limits=self.ik.limits
        joints=[int(np.flatnonzero(self.model.jnt_qposadr==i)[0]) for i in self.qidx]
        self.hard_limits=self.model.jnt_range[joints].copy()
        self.site=session.tcp
        self.empty=empty;self.grip=-1. if empty else 1.
        self.placement_layer=placement_layer
        if placement_layer is not None:
            # A centimetre-scale view pose tolerates millimetre IK error; a
            # metal neck in a comb does not. Do not mistake IK error for CAD
            # collision or allow it to erase the narrow tooth clearance.
            self.ik.position_tolerance=.0001
            self.ik.rotation_tolerance=np.deg2rad(.1)
        self.alias=None if empty else session.held['alias']
        self.gids=set() if empty else session.ids[self.alias][1]
        if not empty:
            obj=session.raw.get_object(self.alias)
            jid=session.sim.model.joint_name2id(obj.joints[-1]);self.object_qadr=int(self.model.jnt_qposadr[jid])
        self.names=[self.model.geom(i).name or '' for i in range(self.model.ngeom)]
        # A few mm clearance in the PLANNING COPY. Live collision geometry and
        # contact settings remain unchanged.
        for g,name in enumerate(self.names):
            if g in self.gids:self.model.geom_margin[g]+=.030
            elif not name.startswith(('robot0','gripper')):self.model.geom_margin[g]+=.015
            else:self.model.geom_margin[g]+=.010
        self.cache={};self.queries=0;self.rng=np.random.default_rng(session.seed+9401)
        self.ctrl=session.controller
        self.old_run=None;self.target=session.sim.data.qpos[self.qidx].copy()
        self.empty_start_q=self.target.copy()
        self.integral=np.zeros(7)

    def refresh(self):
        if self.empty:self.cp=np.zeros(3);self.cr=np.eye(3)
        else:
            self.s.refresh_held_transform()
            self.cp=self.s.held['cap_in_tcp'].copy()
            self.cr=self.s.held['object_rotation_in_tcp'].copy()
        self.snapshot=self.s.sim.data.qpos.copy();self.cache={}
        self.start_q=self.s.sim.data.qpos[self.qidx].copy()

    def valid(self,q):
        key=tuple(np.round(q,5))
        if key in self.cache:return self.cache[key]
        if np.any(q<self.hard_limits[:,0]-.003) or np.any(q>self.hard_limits[:,1]+.003):
            self.last_invalid={'reason':'joint_limit','q':q.tolist()};return False
        self.queries+=1;self.data.qpos[:]=self.snapshot;self.data.qpos[self.qidx]=q
        self.mj.mj_kinematics(self.model,self.data)
        p=self.data.site_xpos[self.site];r=self.data.site_xmat[self.site].reshape(3,3)
        cap=p+r@self.cp;rr=r@self.cr;quat=Rotation.from_matrix(rr).as_quat()[[3,0,1,2]]
        if hasattr(self,'gravity_start') and not self.empty:
            gravity=rr.T@np.array([0.,0.,1.])
            arc=sum(np.arccos(np.clip(gravity@g,-1,1)) for g in (self.gravity_start,self.gravity_goal))
            if arc>self.gravity_arc+np.deg2rad(16):
                self.cache[key]=False;return False
        if not self.empty:self.data.qpos[self.object_qadr:self.object_qadr+7]=np.r_[cap,quat]
        self.mj.mj_forward(self.model,self.data)
        ok=True
        for c in self.data.contact[:self.data.ncon]:
            g1,g2=int(c.geom1),int(c.geom2);a,b=self.names[g1],self.names[g2]
            target=g1 in self.gids or g2 in self.gids
            robot_a=a.startswith(('robot0','gripper'));robot_b=b.startswith(('robot0','gripper'))
            if target:
                other=b if g1 in self.gids else a
                # The measured grasp transform fixes the part relative to the
                # entire hand; an existing near-palm grasp is not an obstacle.
                if other.startswith('gripper'):continue
                threshold=.025
                if self.placement_layer is not None:
                    threshold=.001
                    own=a if g1 in self.gids else b
                    if other.startswith(f'hook_shell{self.placement_layer}_1_'):
                        threshold=-.0005
                if c.dist<threshold:
                    self.last_invalid={'pair':[a,b],'distance_m':float(c.dist)};ok=False;break
            elif robot_a or robot_b:
                if a.startswith('gripper') and b.startswith('gripper'):continue
                if robot_a and robot_b:
                    ba,bb=int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
                    # Joint-adjacent housings are designed to meet. Positive
                    # safety margins must not turn that assembly into a false
                    # self-collision. Non-adjacent links remain checked.
                    if ba==bb or self.model.body_parentid[ba]==bb or self.model.body_parentid[bb]==ba:continue
                # Mounted base contact is a known fixed installation, not arm motion.
                if ('robot0_link0' in a or 'robot0_link0' in b or 'robot0_base' in a or 'robot0_base' in b) and 'table' in a+b:continue
                threshold=.002 if self.placement_layer is not None else .010
                if robot_a and robot_b:
                    # Allow leaving a physically clear pose inside the 10mm
                    # planning buffer after servo compliance. Still require
                    # positive 1mm self-clearance; recover 10mm over 0.03 rad.
                    threshold=min(.010,.001+.30*np.linalg.norm(q-self.start_q))
                if self.placement_layer is not None and (a.startswith('gripper') or b.startswith('gripper')):
                    # Insertion necessarily comes within a centimetre of the
                    # comb. Keep positive 1mm hand clearance, NOT penetration.
                    threshold=.001
                    alias=getattr(self.s,'target_alias',None)
                    other=b if a.startswith('gripper') else a
                    if self.empty and alias and other.startswith(alias+'_'):
                        # Just-opened hand can be sub-mm from the released
                        # stem. Require NON-PENETRATION, not an impossible
                        # positive-clearance starting pose before withdrawal.
                        threshold=-.00015*max(0.,1.-np.linalg.norm(q-self.empty_start_q)/.025)
                if c.dist<threshold:
                    self.last_invalid={'pair':[a,b],'distance_m':float(c.dist)};ok=False;break
        self.cache[key]=ok
        return ok

    def edge(self,a,b,resolution=.025):
        n=max(1,int(np.ceil(np.max(abs(b-a))/resolution)))
        return all(self.valid(a+(b-a)*t) for t in np.linspace(1/n,1,n))

    def plan(self,position,rotation):
        self.refresh()
        start=self.s.sim.data.qpos[self.qidx].copy()
        p0,r0=self.s.tcp_pose();object_r0=r0@self.cr;object_r1=rotation@self.cr
        self.gravity_start=object_r0.T@np.array([0.,0.,1.]);self.gravity_goal=object_r1.T@np.array([0.,0.,1.])
        self.gravity_arc=float(np.arccos(np.clip(self.gravity_start@self.gravity_goal,-1,1)))
        if not self.valid(start):
            raise RuntimeError('Current held configuration lacks planning clearance: '+str(getattr(self,'last_invalid',None)))
        goals=[]
        forced=getattr(self,'forced_goal_joint',None)
        if forced is not None:
            if not self.valid(forced):raise RuntimeError('Stored home joint pose is obstructed: '+str(getattr(self,'last_invalid',None)))
            if self.edge(start,forced):return [start,forced.copy()]
            goals=[forced.copy()]
        for attempt in range(0 if forced is not None else 12):
            seed=start if attempt==0 else np.clip(start+self.rng.normal(0,.8,7),self.limits[:,0],self.limits[:,1])
            if not self.empty and self.s.held['kind']=='obj4' and attempt>=4:
                seed=self.rng.uniform(self.limits[:,0],self.limits[:,1])
            q=self.ik.solve(position,rotation,seed=seed,attempts=2)
            if q is not None and self.valid(q):
                if self.edge(start,q):return [start,q]
                goals.append(q)
        if not goals:raise RuntimeError('No collision-free joint IK: '+str(getattr(self,'last_invalid',None)))
        structured=self.task_path(start,goals,position,rotation) if forced is None else None
        if structured is not None:return structured
        # Bidirectional RRT-Connect; both trees retain their root identity.
        nodes_a=[start];parents_a=[-1];nodes_b=goals.copy();parents_b=[-1]*len(goals)
        root_a='start';root_b='goal'
        started=time.monotonic()
        def extend(nodes,parents,target):
            nearest=int(np.argmin([np.linalg.norm(q-target) for q in nodes]))
            old=nodes[nearest];distance=np.linalg.norm(target-old)
            new=old+(target-old)*min(1.,.22/max(distance,1e-9))
            if not self.edge(old,new):return None,False
            nodes.append(new);parents.append(nearest)
            return len(nodes)-1,distance<=.22
        def branch(nodes,parents,index):
            result=[]
            while index!=-1:result.append(nodes[index]);index=parents[index]
            return result[::-1]
        for iteration in range(2400):
            if time.monotonic()-started>90:break
            target=nodes_b[int(self.rng.integers(len(nodes_b)))] if self.rng.random()<.25 else self.rng.uniform(self.limits[:,0],self.limits[:,1])
            ia,_=extend(nodes_a,parents_a,target)
            if ia is not None:
                reached=False;ib=None
                for _ in range(35):
                    ib,reached=extend(nodes_b,parents_b,nodes_a[ia])
                    if ib is None or reached:break
                if reached:
                    aa=branch(nodes_a,parents_a,ia);bb=branch(nodes_b,parents_b,ib)
                    path=aa+bb[-2::-1] if root_a=='start' else bb+aa[-2::-1]
                    # Shortcut only through collision-checked joint edges.
                    for _ in range(60):
                        if len(path)<3:break
                        i,j=sorted(self.rng.choice(len(path),2,replace=False))
                        if j>i+1 and self.edge(path[i],path[j]):path=path[:i+1]+path[j:]
                    print(f'Joint planner: {len(path)} nodes, {self.queries} collision checks',flush=True)
                    return path
            nodes_a,nodes_b=nodes_b,nodes_a;parents_a,parents_b=parents_b,parents_a;root_a,root_b=root_b,root_a
        raise RuntimeError(f'Joint planner timed out ({self.queries} collision checks); grasp retained')

    def task_path(self,start,goals,position,rotation):
        """RRT on cap XYZ + object-orientation progress, lifted through IK.

        Joint-space shortcuts can roll the load inside the fingers. Restricting
        object orientation to a smooth start->goal family keeps the gravity
        direction between its two validated endpoints.
        """
        p0,r0=self.s.tcp_pose();cap0=p0+r0@self.cp;cap1=position+rotation@self.cp
        slerp=Slerp([0.,1.],Rotation.from_matrix(np.stack([r0@self.cr,rotation@self.cr])))
        start_state=np.r_[cap0,0.];goal_state=np.r_[cap1,1.]
        def pose(state):
            objr=slerp(float(np.clip(state[3],0,1))).as_matrix();tcp_r=objr@self.cr.T
            return state[:3]-tcp_r@self.cp,tcp_r
        def solve(state,seed):
            pp,rr=pose(state)
            q=self.ik.solve(pp,rr,seed=seed,attempts=1)
            return q if q is not None and self.valid(q) and self.edge(seed,q) else None
        # First test a smooth Cartesian object path, not an unconstrained wrist flip.
        direct=[start];q=start
        for t in np.linspace(1/35,1,35):
            candidate=solve(start_state+(goal_state-start_state)*t,q)
            if candidate is None:break
            direct.append(candidate);q=candidate
        else:
            print(f'Object-space planner: direct {len(direct)} nodes',flush=True)
            return direct
        states_a=[start_state];qs_a=[start];parents_a=[-1]
        states_b=[goal_state];qs_b=[goals[0]];parents_b=[-1]
        root_a='start';started=time.monotonic()
        metric=np.array([1.,1.,1.,.35])
        def extend(states,qs,parents,target):
            i=int(np.argmin([np.linalg.norm((s-target)*metric) for s in states]))
            delta=target-states[i];norm=np.linalg.norm(delta*metric)
            new=states[i]+delta*min(1.,.065/max(norm,1e-9))
            q=solve(new,qs[i])
            if q is None:return None,False
            states.append(new);qs.append(q);parents.append(i)
            return len(states)-1,norm<=.065
        def branch(qs,parents,i):
            path=[]
            while i!=-1:path.append(qs[i]);i=parents[i]
            return path[::-1]
        for iteration in range(700):
            if time.monotonic()-started>60:break
            if self.rng.random()<.35:target=states_b[int(self.rng.integers(len(states_b)))]
            else:target=np.r_[self.rng.uniform([-.95,-.70,.98],[.35,.28,1.78]),self.rng.random()]
            ia,_=extend(states_a,qs_a,parents_a,target)
            if ia is not None:
                for _ in range(25):
                    ib,reached=extend(states_b,qs_b,parents_b,states_a[ia])
                    if ib is None:break
                    if reached:
                        if not self.edge(qs_a[ia],qs_b[ib]):break
                        aa=branch(qs_a,parents_a,ia);bb=branch(qs_b,parents_b,ib)
                        path=aa+bb[::-1] if root_a=='start' else bb+aa[::-1]
                        print(f'Object-space planner: RRT {len(path)} nodes, {self.queries} checks',flush=True)
                        return path
            states_a,states_b=states_b,states_a;qs_a,qs_b=qs_b,qs_a;parents_a,parents_b=parents_b,parents_a
            root_a='goal' if root_a=='start' else 'start'
        return None

    def activate(self):
        if self.old_run is not None:return
        self.old_run=self.ctrl.run_controller
        self.s.report['stage2_controller']='collision-checked joint path + torque PID, Panda actuator limits; NOT Cartesian Pi0.5 actions'
        def run_controller():
            self.ctrl.update(force=True)
            q=self.s.sim.data.qpos[self.qidx];v=self.s.sim.data.qvel[self.vidx]
            error=self.target-q
            self.integral=np.clip(self.integral+error*50*self.s.sim.model.opt.timestep,-8,8)
            kp=np.array([500,500,400,350,180,150,100.]);kd=np.array([45,45,35,35,12,12,10.])
            payload=np.zeros(7)
            if not self.empty and self.s.held and not getattr(self.s,'releasing',False):
                bid=self.s.ids[self.s.held['alias']][0]
                model=self.s.sim.model._model;data=self.s.sim.data._data
                jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
                self.mj.mj_jacSite(model,data,jp,jr,self.site)
                lever=data.subtree_com[bid]-data.site_xpos[self.site]
                skew=np.array([[0,-lever[2],lever[1]],[lever[2],0,-lever[0]],[-lever[1],lever[0],0]])
                jac=(jp-skew@jr)[:,self.vidx]
                payload=jac.T@(-model.opt.gravity*model.body_subtreemass[bid])
            tau=kp*error-kd*v+self.integral+self.ctrl.torque_compensation+payload
            tau=np.clip(tau,self.ctrl.actuator_min,self.ctrl.actuator_max)
            self.ctrl.torques=tau
            return tau
        self.ctrl.run_controller=run_controller

    def move(self,position,rotation,phase):
        preplanned=getattr(self,'preplanned',None)
        if preplanned is not None and np.allclose(preplanned[0],position) and np.allclose(preplanned[1],rotation):
            path=preplanned[2];self.preplanned=None
        else:path=self.plan(np.asarray(position),np.asarray(rotation))
        self.activate()
        self.s.report.setdefault('joint_plans',[]).append({'phase':phase,'nodes':[q.tolist() for q in path],
            'collision_queries':self.queries,'live_pose_writes':False})
        for a,b in zip(path,path[1:]):
            # At most 0.2 rad/s; dense execution with physical substeps.
            n=max(12,int(np.ceil(np.max(abs(b-a))/.01)))
            for t in np.linspace(1/n,1,n):
                self.target=a+(b-a)*t
                self.s.stage2_joint_target=self.target.copy()
                self.s.step(np.r_[np.zeros(6),self.grip],phase)
        self.target=path[-1].copy()
        for _ in range(30):
            self.s.stage2_joint_target=self.target.copy();self.s.step(np.r_[np.zeros(6),self.grip],phase)
        p,r=self.s.tcp_pose()
        ep=float(np.linalg.norm(p-position));er=float(np.rad2deg(Rotation.from_matrix(rotation@r.T).magnitude()))
        print(f'{phase} joint tracking: {ep:.4f}m {er:.2f}deg',flush=True)
        if ep>.018 or er>8:raise RuntimeError(f'Joint path tracking failed: {phase}: {ep:.4f}m, {er:.1f}deg')

    def restore(self):
        if self.old_run is not None:
            self.ctrl.run_controller=self.old_run;self.old_run=None
            self.ctrl.update(force=True);self.ctrl.reset_goal()
        self.s.stage2_joint_target=None
