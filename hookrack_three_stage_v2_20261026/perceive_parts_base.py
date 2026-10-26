"""Training-free tabletop RGB-D geometry + CAD registration baseline.

No object world poses / qpos / reset sampled poses enter the estimator.
Default identity: table-plane removal, connected components, CAD dimensions.
Simulator instance segmentation is an explicitly named debug/oracle option.
Camera calibration and flat-rest CAD orientation are known priors.
"""
from __future__ import annotations
import importlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,linear_sum_assignment
from scipy.spatial import cKDTree,ConvexHull,distance
from scipy.spatial.transform import Rotation
from randomize_scene_base import scene_constants,rest_rotation

CAMERAS=('agentview','robot0_eye_in_hand')


def configure_initial_wrist(raw):
    """Runtime-only rigid wrist mount aimed at the known TABLE, not any part.

    The orientation is set once at reset and then follows the wrist rigidly.
    No per-frame look-at tracking, object-pose access, or physical edits.
    """
    sim=raw.sim;cid=sim.model.camera_name2id('robot0_eye_in_hand')
    parent=int(sim.model.cam_bodyid[cid])
    parent_r=sim.data.body_xmat[parent].reshape(3,3)
    sim.model.cam_pos[cid]=[.10,0.,-.06]
    position=sim.data.body_xpos[parent]+parent_r@sim.model.cam_pos[cid]
    target=np.asarray(raw.table_offset)+[0.,-.02,.04]
    back=position-target;back/=np.linalg.norm(back)
    right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right)
    up=np.cross(back,right)
    world_r=np.column_stack([right,up,back])
    quat=Rotation.from_matrix(parent_r.T@world_r).as_quat()[[3,0,1,2]]
    sim.model.cam_quat[cid]=quat
    sim.model.cam_fovy[cid]=110.
    sim.forward()
    return {'name':'random_grasp_table_facing_wrist_v1','camera':'robot0_eye_in_hand',
            'pos':[.10,0.,-.06],'quat_wxyz':quat.tolist(),'fovy_degrees':110.,
            'fixed_at_reset':True,'target':'known table center, no object pose'}


def surface_samples(path,count=60000):
    import trimesh
    mesh=trimesh.load_mesh(path,process=False)
    rng=np.random.default_rng(627)
    triangles=np.asarray(mesh.triangles)
    area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    picked=triangles[rng.choice(len(triangles),count,p=area/area.sum())]
    uv=rng.random((count,2))
    flip=uv.sum(axis=1)>1
    uv[flip]=1-uv[flip]
    return picked[:,0]+uv[:,0,None]*(picked[:,1]-picked[:,0])+uv[:,1,None]*(picked[:,2]-picked[:,0]),np.asarray(mesh.vertices)


def capture_views(raw,output,size=640,instance_oracle=False):
    import imageio.v2 as imageio
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix,get_camera_extrinsic_matrix,get_real_depth_map
    sim=raw.sim
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    views={}
    for camera in CAMERAS:
        rgb,depth=sim.render(size,size,camera_name=camera,depth=True)
        seg=sim.render(size,size,camera_name=camera,segmentation=True) if instance_oracle else np.full((size,size,2),-1,dtype=np.int32)
        # MuJoCo raw buffers are bottom-up. Keep all modalities aligned.
        rgb,depth,seg=rgb[::-1].copy(),depth[::-1].copy(),seg[::-1].copy()
        K=get_camera_intrinsic_matrix(sim,camera,size,size)
        T=get_camera_extrinsic_matrix(sim,camera).copy()
        depth=get_real_depth_map(sim,depth)
        imageio.imwrite(output/f'{camera}_initial.png',rgb)
        views[camera]={'rgb':rgb,'depth_m':depth,'segmentation':seg,'K':K,'camera_to_world':T}
    np.savez_compressed(output/'initial_rgbd_segmentation.npz',**{
        f'{camera}_{key}':value for camera,v in views.items() for key,value in v.items()})
    return views


def pixel_points(view,mask):
    yy,xx=np.where(mask)
    d=view['depth_m'][yy,xx];K=view['K'];T=view['camera_to_world']
    cp=np.stack(((xx+.5-K[0,2])*d/K[0,0],(yy+.5-K[1,2])*d/K[1,1],d),axis=1)
    return cp@T[:3,:3].T+T[:3,3],yy,xx


def diameter(points):
    xy=points[:,:2]
    if len(xy)>4000:xy=xy[::max(1,len(xy)//4000)]
    h=xy[ConvexHull(xy).vertices]
    return float(distance.pdist(h).max())


def geometry_instances(raw,views,assets,quats):
    """Plane removal + XY connected components + known CAD size matching.

    No simulator segmentation ids, object poses, or sampled reset coordinates.
    Requires calibrated depth and a known table plane. Rejects missing pieces.
    """
    import cv2
    center=np.asarray(raw.table_offset);size=np.asarray(raw.table_full_size)
    low=center[:2]-size[:2]/2;high=center[:2]+size[:2]/2
    clouds=[];pixels={}
    for camera,v in views.items():
        points,yy,xx=pixel_points(v,np.ones(v['depth_m'].shape,dtype=bool))
        inside=(points[:,0]>low[0]+.004)&(points[:,0]<high[0]-.004)&(points[:,1]>low[1]+.004)&(points[:,1]<high[1]-.004)
        inside&=(points[:,2]>center[2]+.003)&(points[:,2]<center[2]+.185)&np.isfinite(points).all(axis=1)
        pixels[camera]=(points[inside],yy[inside],xx[inside])
        clouds.append(points[inside])
    cloud=np.concatenate(clouds)
    resolution=.002
    shape=np.ceil((high-low)/resolution).astype(int)+1
    occupancy=np.zeros((shape[1],shape[0]),np.uint8)
    def grid(points):return np.clip(((points[:,:2]-low)/resolution).astype(int),0,shape-1)
    cells=grid(cloud);occupancy[cells[:,1],cells[:,0]]=1
    connected=cv2.dilate(occupancy,np.ones((7,7),np.uint8))
    count,labels=cv2.connectedComponents(connected,connectivity=8)
    point_labels=labels[cells[:,1],cells[:,0]]
    candidates=[]
    for label in range(1,count):
        points=cloud[point_labels==label]
        if len(points)<100:continue
        length=diameter(points)
        if .10<length<.72:candidates.append((label,points,length))
    if len(candidates)<4:raise RuntimeError(f'Only {len(candidates)} separated depth components; require all four')
    priors={}
    for i in range(1,5):
        samples,vertices=surface_samples(assets/'meshes'/f'hook_obj{i}.obj')
        base=rest_rotation(quats[f'hook_obj{i}_1'])
        priors[i]=(samples,vertices,base,diameter(vertices@base.T))
    cost=np.asarray([[abs(c[2]-priors[i][3]) for c in candidates] for i in range(1,5)])
    rows,cols=linear_sum_assignment(cost)
    clouds_by_class={};masks={c:{} for c in views}
    for row,col in zip(rows,cols):
        if cost[row,col]>.035:raise RuntimeError('Depth component does not match CAD dimensions within 35mm')
        name=f'hook_obj{row+1}_1';label,points,length=candidates[col]
        clouds_by_class[name]=points
        for camera,(pts,yy,xx) in pixels.items():
            cells=grid(pts);keep=labels[cells[:,1],cells[:,0]]==label
            mask=np.zeros(views[camera]['depth_m'].shape,bool);mask[yy[keep],xx[keep]]=True
            masks[camera][name]=mask
    return clouds_by_class,masks,priors


def fit_cad(points,local_samples,vertices,base_rotation,table_z,yaw_limit_deg=25):
    """3-DoF tabletop ICP; no world-pose prior from the sampler."""
    if len(points)<80:raise RuntimeError('Insufficient visible depth pixels')
    rng=np.random.default_rng(0)
    if len(points)>3500:points=points[rng.choice(len(points),3500,replace=False)]
    tree=cKDTree(local_samples)
    z=float(table_z-np.min(vertices@base_rotation[2]))
    def rotation(theta):return Rotation.from_euler('z',theta).as_matrix()@base_rotation
    def residual(q):
        r=rotation(q[2]);local=(points-np.array([q[0],q[1],z]))@r
        _,idx=tree.query(local)
        return (local-local_samples[idx]).ravel()
    candidates=[]
    for deg in np.linspace(-yaw_limit_deg,yaw_limit_deg,7):
        r=rotation(np.deg2rad(deg))
        ref=vertices@r.T
        xy=(points[:,:2].min(axis=0)+points[:,:2].max(axis=0)-ref[:,:2].min(axis=0)-ref[:,:2].max(axis=0))/2
        result=least_squares(residual,np.r_[xy,np.deg2rad(deg)],
            bounds=([-.60,-.85,-np.deg2rad(yaw_limit_deg+2)],[.60,.15,np.deg2rad(yaw_limit_deg+2)]),
            loss='soft_l1',f_scale=.002,max_nfev=100,xtol=1e-10,ftol=1e-10,gtol=1e-10,
            x_scale=[.1,.1,.2])
        distances=np.linalg.norm(residual(result.x).reshape(-1,3),axis=1)
        candidates.append((float(np.mean(np.minimum(distances,.015)**2)),result.x,distances))
    _,q,d=min(candidates,key=lambda c:c[0])
    if np.sqrt(np.mean(d*d))>.0045 or np.mean(d<.004)<.90:
        raise RuntimeError(f'CAD registration rejected: RMSE={np.sqrt(np.mean(d*d)):.5f}m')
    return {'cap_position_m':[float(q[0]),float(q[1]),z],
            'rotation_matrix':rotation(q[2]).tolist(),'yaw_delta_deg':float(np.rad2deg(q[2])),
            'rmse_m':float(np.sqrt(np.mean(d*d))),'inlier_fraction_4mm':float(np.mean(d<.004)),
            'points_used':len(points)}


def locate_parts(raw,views,output,yaw_limit_deg=25,mode='rgbd'):
    import mujoco
    import cv2
    _,quats,_=scene_constants(raw)
    core=type(raw).__module__.split('.envs.',1)[0]
    assets=importlib.import_module(core+'.envs.objects.hookrack_objects')._ASSET_ROOT
    estimates={}
    masks={camera:{} for camera in views}
    if mode=='rgbd':
        geometry_clouds,masks,priors=geometry_instances(raw,views,assets,quats)
    for index in range(1,5):
        name=f'hook_obj{index}_1'
        prefix=raw.get_object(name).naming_prefix
        ids=[g for g in range(raw.sim.model.ngeom) if (raw.sim.model.geom_id2name(g) or '').startswith(prefix)] if mode=='instance_oracle' else []
        clouds=[];counts={}
        for camera,v in views.items():
            seg=v['segmentation']
            mask=((seg[:,:,0]==int(mujoco.mjtObj.mjOBJ_GEOM))&np.isin(seg[:,:,1],ids)) if mode=='instance_oracle' else masks[camera][name]
            # Discard rasterized boundary pixels with mixed foreground depth.
            if mode=='instance_oracle':
                mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            masks[camera][name]=mask
            yy,xx=np.where(mask)
            counts[camera]=len(xx)
            if not len(xx):continue
            d=v['depth_m'][yy,xx];K=v['K'];T=v['camera_to_world']
            cp=np.stack(((xx+.5-K[0,2])*d/K[0,0],(yy+.5-K[1,2])*d/K[1,1],d),axis=1)
            world=cp@T[:3,:3].T+T[:3,3]
            clouds.append(world[np.isfinite(world).all(axis=1)])
        if not clouds:raise RuntimeError(f'{name} not visible in either camera')
        samples,vertices=surface_samples(assets/'meshes'/f'hook_obj{index}.obj')
        estimate=fit_cad(np.concatenate(clouds),samples,vertices,rest_rotation(quats[name]),float(raw.table_offset[2]),yaw_limit_deg)
        estimate.update(visible_pixels=counts,identity_source=('simulator instance segmentation oracle' if mode=='instance_oracle' else 'depth table-plane removal + connected components + CAD size assignment'),
                        pose_source='two-camera calibrated depth + CAD registration; no object pose read')
        estimates[f'obj{index}']=estimate
    for camera,v in views.items():
        image=v['rgb'].copy()
        for index in range(1,5):
            mask=masks[camera][f'hook_obj{index}_1'];yy,xx=np.where(mask)
            if len(xx):
                cv2.rectangle(image,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(200,60,190),2)
                cv2.putText(image,f'obj{index}',(int(xx.min()),max(18,int(yy.min())-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,(200,60,190),2)
        import imageio.v2 as imageio
        imageio.imwrite(Path(output)/f'{camera}_detections.png',image)
    (Path(output)/'detections.json').write_text(json.dumps(estimates,indent=2),encoding='utf-8')
    return estimates
