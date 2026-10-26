"""Recompute ALL sampled contact distances, including zero-force contacts.

Only independent MjData is assigned recorded states. No live robot execution
or trajectory repair occurs. This is a 20Hz sampled audit, not a substep proof.
"""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from hang_part import HangingSession
from grasp_part import code_hashes
p=argparse.ArgumentParser();p.add_argument('episode',type=Path);p.add_argument('scratch')
p.add_argument('--benchmark-root',default='../rpent_hanging_benchmark_v2')
p.add_argument('--render-backend',default='glfw');a=p.parse_args()
r=json.loads((a.episode/'report.json').read_text())
assert r['success'] and not r['code_changed_during_episode']
assert r['code_sha256']==code_hashes()
presets=r.get('preset_occupancy_for_audit_only',[])
pattern='free-sparse' if any(x.get('free_dynamic_body') for x in presets) else ('sparse' if presets else '')
layer=presets[0]['layer'] if presets else 1
states=np.load(a.episode/'trajectory.npz')['states_before']
worst={}
with HangingSession(benchmark_root=a.benchmark_root,output=a.scratch,seed=r['seed'],
    layout=r['reset']['classes_in_table_order'],occupied=pattern,occupied_layer=layer,render_backend=a.render_backend,video_stride=0) as s:
    model=s.sim.model._model;data=mujoco.MjData(model)
    names=[model.geom(i).name or '' for i in range(model.ngeom)]
    for index,state in enumerate(states):
        data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:1+model.nq+model.nv]
        data.time=state[0];mujoco.mj_forward(model,data)
        for c in data.contact[:data.ncon]:
            g1,g2=int(c.geom1),int(c.geom2);left,right=names[g1],names[g2]
            robot=lambda n:n.startswith(('robot0','gripper'))
            part=lambda n:n.startswith(('hook_obj','preset_'))
            relevant=part(left) or part(right)
            if robot(left) and robot(right):
                ba,bb=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
                adjacent=ba==bb or model.body_parentid[ba]==bb or model.body_parentid[bb]==ba
                relevant=not adjacent and not (left.startswith('gripper') and right.startswith('gripper'))
            elif robot(left) or robot(right):relevant=True
            if not relevant or c.dist>=0:continue
            key=' | '.join(sorted((left,right)))
            depth=float(-c.dist)
            if depth>worst.get(key,{}).get('penetration_m',0):worst[key]=dict(pair=key,penetration_m=depth,step=index)
        if index%1000==0:print(f'audit {index}/{len(states)}',flush=True)
    ordered=sorted(worst.values(),key=lambda x:-x['penetration_m'])
    maximum=ordered[0]['penetration_m'] if ordered else 0.
    result=dict(accepted=maximum<.001,sampled_frames=len(states),sampling_hz=20,
        include_zero_force_contacts=True,max_penetration_m=maximum,worst_pairs=ordered[:30],
        code_sha256=code_hashes(),method='recorded qpos/qvel -> independent MjData -> mj_forward, no integration or pose repair')
    destination=a.episode/'contact_audit.json'
    if destination.exists():raise FileExistsError(destination)
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('worst_pairs','code_sha256')},indent=2))
