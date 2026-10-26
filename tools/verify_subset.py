"""Verify copied runtime hashes and the complete custom-MJCF file closure."""
import ast,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
audit=json.loads((ROOT/'DEPENDENCY_AUDIT.json').read_text(encoding='utf-8'))
for item in audit['files']:
    p=ROOT/item['path']
    assert p.is_file(),item['path']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],item['path']
expert=ROOT/'hookrack_three_stage_v2_20261026'
scripts=ROOT/'rpent_hanging_benchmark_v2/scripts'
external=set()
for name in audit['runtime_code_files']:
    if not name.endswith('.py'):continue
    for node in ast.walk(ast.parse((expert/name).read_text(encoding='utf-8'))):
        if isinstance(node,ast.ImportFrom) and node.module:
            mod=node.module.split('.')[0]
            if not (expert/(mod+'.py')).is_file() and not (scripts/(mod+'.py')).is_file():external.add(mod)
asset=ROOT/'rpent_hanging_benchmark_v2/assets/hookrack';refs=0
for p in asset.glob('*.xml'):
    for node in ET.parse(p).getroot().iter():
        if node.get('file'):
            assert (asset/node.get('file')).is_file(),(p,node.get('file'));refs+=1
for case in ('verify_all_final_a','verify_obj2_final_a'):
    r=json.loads((expert/'tests'/case/'report.json').read_text(encoding='utf-8'))
    assert r['success'] and not r['code_changed_during_episode']
    for name,h in r['code_sha256'].items():assert hashlib.sha256((expert/name).read_bytes()).hexdigest()==h,(case,name)
print(json.dumps({'source_files_verified':len(audit['files']),'custom_MJCF_file_references_verified':refs,
                  'external_import_roots':sorted(external),'accepted_recordings_match_current_sources':True}))
