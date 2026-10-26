"""Closed-grip rack observation and conservative RGB-D empty-slot search."""
from pathlib import Path
import importlib,json
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation
from perceive_parts import capture_views,pixel_points

def look_rotation(position,target,up=np.array([0.,0.,1.])):
    back=np.asarray(position)-np.asarray(target);back/=np.linalg.norm(back)
    right=np.cross(up,back)
    if np.linalg.norm(right)<1e-5:raise RuntimeError('Degenerate camera look direction')
    right/=np.linalg.norm(right)
    return np.column_stack([right,np.cross(back,right),back])

def rack_slots(session,layer):
    """Known fixture CAD dimensions, not workpiece poses or occupancy labels."""
    raw=session.raw;sim=session.sim;rack=raw.get_object(f'hook_shell{layer}_1')
    slot=sim.data.get_site_xpos(f'{rack.naming_prefix}target_slot_site').copy()
    core=type(raw).__module__.split('.envs.',1)[0]
    assets=importlib.import_module(core+'.envs.objects.hookrack_objects')._ASSET_ROOT
    data=json.loads((assets/'meshes'/'rack_collision'/f'shell{layer}_decomposition.json').read_text())
    # Each interior cell contains one full tooth pitch. The designated slot
    # determines the pitch phase; derive pitch from median CAD cell spacing.
    centers=sorted((c['x_interval'][0]+c['x_interval'][1])/2 for c in data['cells'])
    pitch=float(np.median(np.diff(centers)))
    bid=sim.model.body_name2id(rack.root_body);rx=float(sim.data.body_xpos[bid][0])
    xmin=rx+min(c['x_interval'][0] for c in data['cells']);xmax=rx+max(c['x_interval'][1] for c in data['cells'])
    xs=[slot[0]+k*pitch for k in range(-40,41) if xmin+.05<slot[0]+k*pitch<xmax-.05]
    return np.array([[x,slot[1],slot[2]] for x in xs]),pitch

def count_head_depth_points(cloud,slot,pitch):
    """Capture the real plate top (near local z=0), not only raised bosses.

    Bearing tops lie 1--2.5mm BELOW the nominal cap-centre site. A narrow
    near-centre XY crop also excludes the continuous rear rack spine.
    """
    near=(abs(cloud[:,0]-slot[0])<min(.020,pitch*.30))&(abs(cloud[:,1]-slot[1])<.018)
    near&=(cloud[:,2]>slot[2]-.0007)&(cloud[:,2]<slot[2]+.012)
    return int(near.sum())

def scan_slots(session,layer,folder):
    """Observe occupancy below each bearing using measured two-view depth.

    A clear result needs actual background depth behind the expected workpiece
    volume. Missing/occluded pixels mean UNKNOWN, never empty by default.
    """
    views=capture_views(session.raw,folder)
    slots,pitch=rack_slots(session,layer)
    measured_cloud=np.concatenate([pixel_points(v,np.ones(v['depth_m'].shape,bool))[0] for v in views.values()])
    rows=[]
    for idx,slot in enumerate(slots):
        blocking=0;clear_views=0;visible_views=[];details={}
        # Thin round heads above the bearing also count as occupied. This
        # catches a narrow stem whose centre misses the below-row probe rays.
        cap_points=count_head_depth_points(measured_cloud,slot,pitch)
        for camera,v in views.items():
            T=v['camera_to_world'];local=(slot-T[:3,3])@T[:3,:3]
            # Test three sample points well below comb metal, above the next row.
            samples=np.array([slot+[0,-.025,-h] for h in (.055,.085,.115)])
            cc=(samples-T[:3,3])@T[:3,:3]
            uv=cc[:,:2]/cc[:,2,None]@v['K'][:2,:2].T+v['K'][:2,2]
            inview=(cc[:,2]>.01)&(uv[:,0]>4)&(uv[:,0]<v['rgb'].shape[1]-5)&(uv[:,1]>4)&(uv[:,1]<v['rgb'].shape[0]-5)
            if inview.all():visible_views.append(camera)
            clear=0;blocked=0
            for point,pixel,depth,ok in zip(samples,uv,cc[:,2],inview):
                if not ok:continue
                x,y=np.round(pixel).astype(int)
                measured=float(np.median(v['depth_m'][y-2:y+3,x-2:x+3]))
                # Expected air sample +5mm must be visible in front of depth.
                if measured>depth+.005:clear+=1
                elif measured<depth-.015:blocked+=1
            details[camera]={'clear_samples':clear,'blocked_samples':blocked}
            clear_views+=int(clear==3);blocking+=blocked
        # One camera with actual background behind all three probes is a
        # positive free-space witness. An occluded second camera is not a
        # contradictory occupancy measurement (it may see the held part).
        state='occupied' if cap_points>=2 else ('unknown' if cap_points else ('empty' if clear_views>=1 else ('occupied_or_occluded' if blocking else 'unknown')))
        rows.append({'index':idx,'position_m':slot.tolist(),'state':state,'visible_views':visible_views,
                     'cap_depth_points':cap_points,'depth_evidence':details})
    visibility=tooth_visibility(session,layer,views)
    result={'layer':layer,'pitch_m':pitch,'slots':rows,'tooth_visibility':visibility}
    import cv2,imageio.v2 as imageio
    for camera,v in views.items():
        image=v['rgb'].copy();T=v['camera_to_world']
        for row in rows:
            point=np.array(row['position_m'])+[0,0,-.06]
            cp=(point-T[:3,3])@T[:3,:3]
            if cp[2]<=0:continue
            uv=v['K']@(cp/cp[2]);x,y=np.round(uv[:2]).astype(int)
            if 0<=x<image.shape[1] and 0<=y<image.shape[0]:
                color=(40,210,70) if row['state']=='empty' else (240,80,40)
                cv2.circle(image,(x,y),4,color,-1)
                cv2.putText(image,str(row['index']),(x+3,y-5),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        imageio.imwrite(Path(folder)/f'{camera}_slots.png',image)
    Path(folder).joinpath('rack_scan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

@lru_cache(maxsize=3)
def tooth_markers(asset_root,layer):
    asset_root=Path(asset_root)
    report=json.loads((asset_root/'meshes/rack_collision'/f'shell{layer}_decomposition.json').read_text())
    markers=[]
    import xml.etree.ElementTree as ET
    site=ET.parse(asset_root/f'hook_shell{layer}.xml').find('.//site[@name="target_slot_site"]')
    bearing=np.fromstring(site.get('pos'),sep=' ')[2]
    for cell in report['cells']:
        vertices=[]
        for part in cell['parts']:
            p=asset_root/'meshes/rack_collision'/part['file']
            vertices.extend(tuple(map(float,line.split()[1:4])) for line in p.read_text().splitlines() if line.startswith('v '))
        v=np.unique(np.array(vertices),axis=0)
        visible_surface=v[(v[:,1]<0)&(abs(v[:,2]-bearing)<.02)]
        if len(visible_surface)<3:continue
        center=(cell['x_interval'][0]+cell['x_interval'][1])/2
        probes=[]
        for y in (-.055,-.035,-.015):
            ideal=np.array([center,y,bearing])
            probes.append(visible_surface[np.argmin(np.linalg.norm(visible_surface-ideal,axis=1))])
        markers.append(probes)
    return np.asarray(markers)

def tooth_visibility(session,layer,views):
    core=type(session.raw).__module__.split('.envs.',1)[0]
    assets=importlib.import_module(core+'.envs.objects.hookrack_objects')._ASSET_ROOT
    rack=session.raw.get_object(f'hook_shell{layer}_1');sim=session.sim
    bid=sim.model.body_name2id(rack.root_body)
    markers=tooth_markers(str(assets),layer)@sim.data.body_xmat[bid].reshape(3,3).T+sim.data.body_xpos[bid]
    result={}
    for camera,v in views.items():
        T=v['camera_to_world'];flat=markers.reshape(-1,3)
        cp=(flat-T[:3,3])@T[:3,:3]
        uv=(cp/cp[:,2,None])@v['K'].T;seen=[]
        for p,pix in zip(cp,uv):
            x,y=np.round(pix[:2]).astype(int)
            valid=p[2]>.01 and 2<=x<v['rgb'].shape[1]-2 and 2<=y<v['rgb'].shape[0]-2
            depth=float(np.median(v['depth_m'][y-1:y+2,x-1:x+2])) if valid else -99
            seen.append(bool(valid and abs(depth-p[2])<.025))
        flags=np.asarray(seen).reshape(-1,3).any(1)
        result[camera]={'visible_cells':int(flags.sum()),'total_cells':len(flags),'fraction':float(flags.mean())}
    return result

def observer_pose(session,center,layer,cam_local_p,cam_local_r,preferred_object_rotation=None,exact_aim=False):
    """Search upright, yaw-only held poses with a rigid side-mounted camera."""
    from randomize_scene_base import scene_constants
    from shapely.geometry import MultiPoint,box
    held=session.held;_,_,get_vertices=scene_constants(session.raw)
    vertices=get_vertices('hook_'+held['kind'])
    # Boundary vertices suffice for these convex keep-out screens.
    from scipy.spatial import ConvexHull
    vertices=vertices[ConvexHull(vertices).vertices]
    forward_in_object=held['object_rotation_in_tcp'].T@cam_local_r@np.array([0,0,-1.])
    preferred=np.eye(3) if preferred_object_rotation is None else preferred_object_rotation
    forward_preferred=preferred@forward_in_object
    north_yaw=np.pi/2-np.arctan2(forward_preferred[1],forward_preferred[0])
    variants=[(preferred,north_yaw,0.)]
    if exact_aim:
        axis=np.cross([0.,0.,1.],forward_preferred)
        axis/=max(np.linalg.norm(axis),1e-9)
        for degrees in (-10.,10.,-20.,20.,-30.,30.):
            pref=Rotation.from_rotvec(axis*np.deg2rad(degrees)).as_matrix()@preferred
            forward=pref@forward_in_object
            variants.append((pref,np.pi/2-np.arctan2(forward[1],forward[0]),degrees))
    c=np.asarray(session.raw.table_offset);size=np.asarray(session.raw.table_full_size)
    base=session.sim.data.get_body_xpos('robot0_base').copy()
    session.report['robot_base_world_m']=base.tolist()
    table=box(c[0]-size[0]/2-.015,c[1]-size[1]/2-.015,c[0]+size[0]/2+.015,c[1]+size[1]/2+.015)
    slots,_=rack_slots(session,layer)
    cid=session.sim.model.camera_name2id('robot0_eye_in_hand')
    tangent=np.tan(np.deg2rad(session.sim.model.cam_fovy[cid]/2))
    rows=[rack_slots(session,j)[0][0,2] for j in (1,2,3)]
    candidates=[]
    for shift in ((0.,) if exact_aim else (0.,.15,-.15,.30)):
        settings=[(heading,pref,north,tilt) for pref,north,tilt in variants
                  for heading in ((0.,-10.,10.,-20.,20.,-35.,35.,-45.,45.) if exact_aim else (0.,-10.,10.,-20.,20.))]
        for heading,pref,north,extra_tilt in settings:
            vr=Rotation.from_euler('z',north+np.deg2rad(heading)).as_matrix()@pref
            rr=vr@held['object_rotation_in_tcp'].T
            direction=-rr@cam_local_r[:,2]
            for distance in ((.40,.45,.50,.55,.60,.65,.70,.80,.90) if exact_aim else (.40,.45,.50,.55,.60,.65,.70)):
                aim=center+np.array([shift,0.,-.04]);cam=aim-distance*direction
                pp=cam-rr@cam_local_p
                cap=pp+rr@held['cap_in_tcp'];vr=rr@held['object_rotation_in_tcp']
                points=vertices@vr.T+cap
                footprint=MultiPoint(points[:,:2]).convex_hull
                if points[:,2].min()<c[2]+.025 and footprint.intersects(table):continue
                if np.any((np.linalg.norm(points[:,:2]-base[:2],axis=1)<.17)&(points[:,2]<base[2]+.50)):continue
                if any(np.any((points[:,1]>.285)&(abs(points[:,2]-z)<.055)) for z in rows):continue
                if np.linalg.norm(pp-base)>.82 or not .60<pp[2]<1.61:continue
                camera_r=rr@cam_local_r
                cc=(slots-cam)@camera_r
                in_frame=(cc[:,2]<-.01)&(abs(cc[:,0])<-cc[:,2]*tangent*.92)&(abs(cc[:,1])<-cc[:,2]*tangent*.92)
                fraction=float(in_frame.mean())
                if fraction<.65:continue
                tilt=np.rad2deg(np.arccos(np.clip(vr[2,2],-1,1)))
                score=abs(heading)*.004+abs(extra_tilt)*.003+abs(shift)*.8+abs(distance-(.55 if layer==3 else .45))*.8-fraction*.25
                candidates.append((score,cam,rr,pp,{'estimated_part_tilt_deg':float(tilt),'projected_slot_fraction':fraction,'aim_shift_m':shift,
                    'estimated_cap_m':cap.tolist(),'estimated_min_cad_z_m':float(points[:,2].min())}))
    if not candidates:raise RuntimeError('No conservative camera/held-CAD observation pose found; grip retained')
    candidates.sort(key=lambda x:x[0])
    session._observation_candidates=candidates
    from kinematic_screen import PandaIK
    ik=PandaIK(session)
    for selected_index,(_,cam,rr,pp,evidence) in enumerate(candidates):
        joints=ik.solve(pp,rr,attempts=4)
        if joints is not None:
            evidence['joint_limit_ik_verified']=True
            break
    else:raise RuntimeError('No joint-limit-feasible observation pose; grip retained')
    session._observation_candidates=[candidates[selected_index]]+candidates[:selected_index]+candidates[selected_index+1:]
    session.report['observation_pose_plan']=evidence
    print('Observation pose:',cam.round(3).tolist(),'TCP',pp.round(3).tolist(),evidence,flush=True)
    return cam,rr,pp

def observe_and_select(session,kind,layer,initial,center,desired_cam_pos,desired_cam_r,cam_local_p,cam_local_r):
    sim=session.sim;cid=sim.model.camera_name2id('robot0_eye_in_hand')
    scan=scan_slots(session,layer,session.out/'rack_initial')
    # Only reachable, visible empty slots may be selected. The full CAD rack
    # extends beyond Panda reach; a distant empty slot is not a usable slot.
    def usable(s):
        radius={1:.04,2:.04,3:.075}[layer]
        neighboring_heads=[r for r in scan['slots'] if r['state']=='occupied']
        separated=all(abs(s['position_m'][0]-r['position_m'][0])>2*radius+.008 for r in neighboring_heads)
        return s['state']=='empty' and separated and -.65<s['position_m'][0]<.25 and 'robot0_eye_in_hand' in s['visible_views']
    choices=[s for s in scan['slots'] if usable(s)]
    scans=[scan]
    if not choices or sum(s['state']=='occupied' for s in scan['slots'])>len(scan['slots'])*.5 or scan['tooth_visibility']['robot0_eye_in_hand']['fraction']<=.5:
        scan_origin_p,scan_origin_r=session.tcp_pose()
        for j,dx in enumerate((.14,-.14,.24,-.24),1):
            before=len(session.actions)
            # Translate without twisting the loaded wrist; the subsequent
            # dedicated recentering step chooses the final optical direction.
            try:session.move(scan_origin_p+[dx,0,0],scan_origin_r,f'rack_lateral_scan_{j}',220)
            except RuntimeError:
                if len(session.actions)!=before:raise
                continue
            scan=scan_slots(session,layer,session.out/f'rack_scan_{j}');scans.append(scan)
            choices=[s for s in scan['slots'] if usable(s)]
            if choices and scan['tooth_visibility']['robot0_eye_in_hand']['fraction']>.5:break
    if not choices:raise RuntimeError('No observed reachable empty slot; grip retained, no insertion attempted')
    def angle_to_slot(slot):
        optical=-sim.data.cam_xmat[cid].reshape(3,3)[:,2]
        direction=np.array(slot['position_m'])+[0,0,-.04]-sim.data.cam_xpos[cid]
        direction/=np.linalg.norm(direction)
        return float(np.rad2deg(np.arccos(np.clip(optical@direction,-1,1))))
    for chosen in sorted(choices,key=angle_to_slot):
        point=np.array(chosen['position_m'])
        if angle_to_slot(chosen)<=6.:break
        _,current_object_r=session.refresh_held_transform()
        try:observer_pose(session,point,layer,cam_local_p,cam_local_r,current_object_r,exact_aim=True)
        except RuntimeError:continue
        reached=False
        for _,cam,rr,pp,evidence in session._observation_candidates[:35]:
            before=len(session.actions)
            try:session.move(pp,rr,'face_empty_slot',260)
            except RuntimeError:
                if len(session.actions)!=before:raise
                continue
            session.report['empty_slot_camera_plan']=evidence;reached=True
            break
        if reached:break
    else:raise RuntimeError('No collision-free camera pose for any observed reachable empty slot')
    session.move(*session.tcp_pose(),'rack_final_hold',60)
    final=scan_slots(session,layer,session.out/'rack_final')
    final_slot=next(s for s in final['slots'] if s['index']==chosen['index'])
    rows=session.traces[-60:];bilateral=float(np.mean([session.bilateral(r) for r in rows]))
    in_frame=sum('robot0_eye_in_hand' in s['visible_views'] for s in final['slots'])/len(final['slots'])
    visible=final['tooth_visibility']['robot0_eye_in_hand']['fraction']
    optical=-sim.data.cam_xmat[cid].reshape(3,3)[:,2]
    direction=point+[0,0,-.04]-sim.data.cam_xpos[cid];direction/=np.linalg.norm(direction)
    angle=float(np.rad2deg(np.arccos(np.clip(optical@direction,-1,1))))
    drift=max(np.linalg.norm(np.array(r['target_cap_m'])-rows[0]['target_cap_m']) for r in rows)
    report={'success':bool(bilateral>=.95 and visible>.5 and angle<8 and drift<.01 and final_slot['state']=='empty'),
        'kind':kind,'layer':layer,'selected_slot':final_slot,'wrist_visible_slot_fraction':visible,
        'wrist_tooth_visibility':final['tooth_visibility']['robot0_eye_in_hand'],'wrist_in_frame_slot_fraction':in_frame,
        'camera_target_angle_deg':angle,'bilateral_final_hold_fraction':bilateral,
        'final_hold_cap_drift_m':float(drift),'final_hold_duration_s':3.,
        'stage_steps':len(session.traces)-initial,'scan_count':len(scans),'released':False}
    session.report['face_rack']=report
    if not report['success']:raise RuntimeError('Rack-view acceptance failed: '+json.dumps(report))
    return report

def face_rack(session,kind):
    """Plan an actual collision-free arm/held-part path to the correct rack."""
    if not session.held or session.held['kind']!=kind:raise ValueError('Held class does not match requested rack class')
    from joint_motion import JointMotion
    layer={'obj1':1,'obj2':1,'obj3':2,'obj4':3}[kind]
    initial=len(session.traces)
    _,object_rotation=session.refresh_held_transform()
    sim=session.sim;cid=sim.model.camera_name2id('robot0_eye_in_hand')
    p,r=session.tcp_pose();local_r=r.T@sim.data.cam_xmat[cid].reshape(3,3)
    local_p=r.T@(sim.data.cam_xpos[cid]-p)
    slots,_=rack_slots(session,layer);center=slots[np.argmin(abs(slots[:,0]+.41))]
    observer_pose(session,center,layer,local_p,local_r,object_rotation if kind=='obj4' else None)
    motion=JointMotion(session);old_move=session.move
    try:
        errors=[]
        for _,cam,rr,pp,evidence in session._observation_candidates[:30]:
            try:
                path=motion.plan(pp,rr)
            except RuntimeError as error:
                errors.append(str(error));continue
            session.report['observation_pose_plan']=evidence
            session.report['planning_rejected_candidates']=errors
            break
        else:raise RuntimeError('No collision-free observation path: '+str(errors[-3:]))
        # Execute the path just checked instead of replanning after movement.
        motion.preplanned=(np.asarray(pp),np.asarray(rr),path)
        def joint_move(target,rotation,phase,n=180,grip=None):
            if grip is not None and grip<0:raise RuntimeError('Stage 2 cannot release the gripper')
            if phase=='rack_final_hold':
                for _ in range(n):
                    session.stage2_joint_target=motion.target.copy()
                    session.step(np.r_[np.zeros(6),1.],phase)
            else:motion.move(np.asarray(target),np.asarray(rotation),phase)
        session.move=joint_move
        motion.move(np.asarray(pp),np.asarray(rr),'face_correct_rack')
        return observe_and_select(session,kind,layer,initial,center,cam,rr@local_r,local_p,local_r)
    finally:
        session.move=old_move;motion.restore()
