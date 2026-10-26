"""Panda joint-limit IK checks in isolated MjData; never writes live qpos.

Only robot FK/Jacobians enter the IK objective. Scene object poses do not enter
the solver. A positive result is kinematic feasibility, not collision clearance.
"""
import numpy as np
from scipy.spatial.transform import Rotation

class PandaIK:
    def __init__(self,session):
        import mujoco
        self.mj=mujoco;self.model=session.sim.model._model
        self.data=mujoco.MjData(self.model)
        self.data.qpos[:]=session.sim.data.qpos
        robot=session.raw.robots[0]
        self.qidx=np.asarray(robot._ref_joint_pos_indexes,int)
        self.vidx=np.asarray(robot._ref_joint_vel_indexes,int)
        self.site=session.tcp
        joints=[int(np.flatnonzero(self.model.jnt_qposadr==i)[0]) for i in self.qidx]
        self.limits=self.model.jnt_range[joints].copy();self.limits[:,0]+=.015;self.limits[:,1]-=.015
        self.initial=session.sim.data.qpos[self.qidx].copy()
        self.position_tolerance=.003;self.rotation_tolerance=np.deg2rad(2.)

    def solve(self,position,rotation,seed=None,attempts=4):
        rng=np.random.default_rng(57)
        initial=self.initial if seed is None else np.asarray(seed)
        seeds=[initial,self.initial]+[initial+rng.normal(0,.5,len(initial)) for _ in range(max(0,attempts-2))]
        for q in seeds[:attempts]:
            q=np.clip(q,self.limits[:,0],self.limits[:,1])
            for _ in range(100):
                self.data.qpos[self.qidx]=q
                self.mj.mj_kinematics(self.model,self.data);self.mj.mj_comPos(self.model,self.data)
                p=self.data.site_xpos[self.site];r=self.data.site_xmat[self.site].reshape(3,3)
                ep=np.asarray(position)-p;er=Rotation.from_matrix(rotation@r.T).as_rotvec()
                if np.linalg.norm(ep)<self.position_tolerance and np.linalg.norm(er)<self.rotation_tolerance:return q.copy()
                jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
                self.mj.mj_jacSite(self.model,self.data,jp,jr,self.site)
                j=np.vstack([jp[:,self.vidx],.3*jr[:,self.vidx]])
                error=np.r_[ep,.3*er]
                dq=j.T@np.linalg.solve(j@j.T+np.eye(6)*.00008,error)
                dq*=min(1.,.12/max(np.linalg.norm(dq),1e-9))
                q=np.clip(q+dq,self.limits[:,0],self.limits[:,1])
        return None
