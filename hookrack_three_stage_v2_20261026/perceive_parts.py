"""Repeated-instance RGB-D/CAD recognition without category-count priors."""
from pathlib import Path
import importlib,json
import numpy as np
from perceive_parts_base import (CAMERAS,capture_views,configure_initial_wrist as _configure_initial_wrist,
    pixel_points,diameter,surface_samples,fit_cad)
from randomize_scene_base import scene_constants,rest_rotation

def configure_initial_wrist(raw):
    """Rigid side mount at reset: reduce hand occlusion without wrist twisting."""
    from scipy.spatial.transform import Rotation
    profile=_configure_initial_wrist(raw)
    cid=raw.sim.model.camera_name2id('robot0_eye_in_hand')
    q=raw.sim.model.cam_quat[cid]
    rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    raw.sim.model.cam_pos[cid]+=rotation@np.array([-.14,0.,0.])
    raw.sim.model.cam_fovy[cid]=120.
    raw.sim.forward()
    profile.update(name='two_stage_side_wrist_v2',pos=raw.sim.model.cam_pos[cid].tolist(),
                   optical_side_extension_m=-.14,fovy_degrees=120.)
    return profile

def locate_parts(raw,views,output,yaw_limit_deg=25,expected_count=None):
    import cv2,imageio.v2 as imageio
    center=np.asarray(raw.table_offset);size=np.asarray(raw.table_full_size)
    low=center[:2]-size[:2]/2;high=center[:2]+size[:2]/2
    pixels={};clouds=[]
    for camera,view in views.items():
        pts,yy,xx=pixel_points(view,np.ones(view['depth_m'].shape,bool))
        keep=(pts[:,0]>low[0]+.004)&(pts[:,0]<high[0]-.004)&(pts[:,1]>low[1]+.004)&(pts[:,1]<high[1]-.004)
        keep&=(pts[:,2]>center[2]+.003)&(pts[:,2]<center[2]+.185)&np.isfinite(pts).all(1)
        pixels[camera]=(pts[keep],yy[keep],xx[keep]);clouds.append(pts[keep])
    cloud=np.concatenate(clouds);res=.002
    shape=np.ceil((high-low)/res).astype(int)+1
    def grid(pts):return np.clip(((pts[:,:2]-low)/res).astype(int),0,shape-1)
    cells=grid(cloud);occupancy=np.zeros((shape[1],shape[0]),np.uint8)
    occupancy[cells[:,1],cells[:,0]]=1
    count,labels=cv2.connectedComponents(cv2.dilate(occupancy,np.ones((7,7),np.uint8)),connectivity=8)
    ids=labels[cells[:,1],cells[:,0]]
    _,quats,_=scene_constants(raw)
    core=type(raw).__module__.split('.envs.',1)[0]
    assets=importlib.import_module(core+'.envs.objects.hookrack_objects')._ASSET_ROOT
    priors={}
    for i in range(1,5):
        samples,vertices=surface_samples(assets/'meshes'/f'hook_obj{i}.obj')
        base=rest_rotation(quats[f'hook_obj{i}_1'])
        priors[f'obj{i}']=(samples,vertices,base,diameter(vertices@base.T))
    estimates=[]
    for label in range(1,count):
        points=cloud[ids==label]
        if len(points)<100:continue
        length=diameter(points)
        if not .10<length<.72:continue
        print(f'Fitting depth component {label}: length={length:.4f}m points={len(points)}',flush=True)
        np.savez_compressed(Path(output)/f'component_{label}.npz',points=points)
        fits=[]
        for kind,(samples,vertices,base,expected) in priors.items():
            if abs(expected-length)>.035:continue
            height=np.ptp((vertices@base.T)[:,2])
            if np.quantile(points[:,2],.995)>center[2]+height+.006:continue
            try:est=fit_cad(points,samples,vertices,base,float(center[2]),yaw_limit_deg)
            except RuntimeError as error:
                print(f'  rejected {kind}: {error}',flush=True)
                continue
            fits.append((est['rmse_m'],kind,est))
        if not fits:
            print(f'Ignoring non-CAD component {label}; it is not counted as a workpiece',flush=True)
            continue
        fits.sort(key=lambda x:x[0]);_,kind,est=fits[0]
        if len(fits)>1 and fits[1][0]<fits[0][0]*1.15:
            raise RuntimeError('Ambiguous CAD identity; do not grasp')
        masks={};counts={}
        for camera,(pts,yy,xx) in pixels.items():
            cc=grid(pts);keep=labels[cc[:,1],cc[:,0]]==label
            mask=np.zeros(views[camera]['depth_m'].shape,bool);mask[yy[keep],xx[keep]]=True
            masks[camera]=mask;counts[camera]=int(keep.sum())
        est.update(instance_id=f'visual_{len(estimates)+1}',kind=kind,visible_pixels=counts,
                   identity_source='independent depth component and CAD fit; repeated classes permitted')
        print(f"Detected {est['instance_id']}: {kind}, yaw={est['yaw_delta_deg']:.2f}deg RMSE={est['rmse_m']:.5f}m",flush=True)
        estimates.append((est,masks))
    if expected_count is not None and len(estimates)!=expected_count:
        raise RuntimeError(f'Expected {expected_count} separated visible instances, found {len(estimates)}')
    for camera,view in views.items():
        image=view['rgb'].copy()
        for est,masks in estimates:
            yy,xx=np.where(masks[camera])
            if len(xx):
                cv2.rectangle(image,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(200,60,190),2)
                cv2.putText(image,f"{est['instance_id']} {est['kind']}",(int(xx.min()),max(18,int(yy.min())-4)),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,60,190),1)
        imageio.imwrite(Path(output)/f'{camera}_detections.png',image)
    result=[e for e,_ in estimates]
    (Path(output)/'detections.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
