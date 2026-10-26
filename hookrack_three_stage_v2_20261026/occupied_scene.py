"""Optional reset-only fixed CAD props for testing crowded-rack observation.

These are pre-existing mounted scene objects, NOT a hanging demonstration. The
picked target always remains free and is never welded or repositioned.
"""
import importlib,json
import numpy as np

def install_runtime_problem(pattern,layer):
    core='liberopro.liberopro'
    scene=importlib.import_module(core+'.envs.problems.hookrack_manipulation')
    domain=importlib.import_module(core+'.envs.bddl_base_domain')
    objects=importlib.import_module(core+'.envs.objects.hookrack_objects')
    base=domain.TASK_MAPPING['hook_rack_manipulation']
    class HookrackCrowded(base):
        def _load_model(self):
            super()._load_model()
            # Resource budget only: same geometry, solver and physics. The
            # legacy njmax/nconmax=5000 default reserves multi-GB arenas for
            # every MjData used by IK/collision checking on Windows.
            import xml.etree.ElementTree as ET
            size=self.model.root.find('size')
            if size is None:size=ET.SubElement(self.model.root,'size')
            for key in ('njmax','nconmax','nstack','memory'):size.attrib.pop(key,None)
            size.set('memory','128M')
            self.preset_occupied=[]
            if not pattern:return
            data=json.loads((objects._ASSET_ROOT/'meshes'/'rack_collision'/f'shell{layer}_decomposition.json').read_text())
            centers=sorted((c['x_interval'][0]+c['x_interval'][1])/2 for c in data['cells'])
            pitch=float(np.median(np.diff(centers)))
            origin=np.asarray(scene._FIXTURE_POSES[f'hook_shell{layer}_1'])
            # All rack slot sites in the supplied v2 have local x=.04,y=.001;
            # query the source XML site rather than hard-code its bearing Z.
            tree=ET.parse(objects._ASSET_ROOT/f'hook_shell{layer}.xml')
            local=np.fromstring(tree.find('.//site[@name="target_slot_site"]').get('pos'),sep=' ')
            slot=origin+local
            self.preset_occupied=[]
            indices=range(-40,41) if pattern=='almost-full' else ((-8,3) if pattern=='free-sparse' else ((-6,3) if pattern=='sparse' else range(-1,2)))
            xmin=origin[0]+min(c['x_interval'][0] for c in data['cells'])+.05
            xmax=origin[0]+max(c['x_interval'][1] for c in data['cells'])-.05
            for j,k in enumerate(indices):
                # Leave a reachable far-right region to test lateral discovery.
                x=slot[0]+k*pitch
                if not xmin<x<xmax or x>.10:continue
                kind={1:'hook_obj2',2:'hook_obj3',3:'hook_obj4'}[layer]
                free=pattern=='free-sparse'
                name=f'preset_{layer}_{j}'
                prop=(objects._HookWorkpiece if free else objects._HookFixture)(name,kind)
                body=prop.get_obj();position=slot.copy();position[0]=x
                quat=np.array([1.,0.,0.,0.])
                if free:
                    from collect_expert_hanging import PROFILES
                    from scipy.spatial.transform import Rotation
                    profile=PROFILES[{1:1,2:2,3:3}[layer]]
                    position+=np.array([profile.get('x_offset',0),profile.get('y_offset',0),.0001])
                    quat=Rotation.from_euler('z',profile['yaw'],degrees=True).as_quat()[[3,0,1,2]]
                body.set('pos',' '.join(map(str,position)));body.set('quat',' '.join(map(str,quat)))
                self.model.merge_assets(prop);self.model.worldbody.append(body)
                self.preset_occupied.append({'x':x,'layer':layer,'kind':kind,'fixed_scene_prop':not free,
                    'name':name,'body_name':prop.root_body,'free_dynamic_body':free,'reset_position_m':position.tolist()})
    HookrackCrowded.__module__=base.__module__
    HookrackCrowded.__name__=base.__name__
    domain.TASK_MAPPING['hook_rack_manipulation']=HookrackCrowded
    return base
