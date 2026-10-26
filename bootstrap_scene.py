"""Install only the 2026-10-26 expert's scene dependencies; no saved states.

Default is a preview. Explicit --apply backs up every differing target first.
Use a dedicated venv/Conda environment; no system Python or recursive deletion.
"""
import argparse,hashlib,importlib,json,os,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');a=parser.parse_args()
    if sys.prefix==sys.base_prefix:parser.error('Activate a dedicated venv or Conda environment first')
    os.environ.setdefault('LIBERO_CONFIG_PATH',str(ROOT/'.runtime/libero-config'))
    os.environ['MUJOCO_GL']='disable'
    core=importlib.import_module('liberopro.liberopro');package=Path(core.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):parser.error('LIBERO is outside the active environment')
    for key in ('assets','bddl_files','init_states'):
        if not Path(core.get_libero_path(key)).resolve().is_relative_to(package):
            parser.error('Use a new LIBERO_CONFIG_PATH dedicated to this environment')
    audit=json.loads((ROOT/'DEPENDENCY_AUDIT.json').read_text(encoding='utf-8'))
    for item in audit['files']:
        if sha(ROOT/item['path'])!=item['sha256']:raise RuntimeError('Bundled source changed: '+item['path'])
    base=ROOT/'hookrack_v2_standalone_20261019';bench=ROOT/'rpent_hanging_benchmark_v2'
    manifest=json.loads((base/'BASE_ASSETS.json').read_text(encoding='utf-8'))
    pairs=[]
    for rel,expected in manifest['files_sha256'].items():
        source=base/'base_assets'/rel
        if sha(source)!=expected:raise RuntimeError('Base resource changed: '+rel)
        pairs.append((source,package/'assets'/rel))
    for source in (bench/'assets/hookrack').rglob('*'):
        if source.is_file():pairs.append((source,package/'assets/hookrack'/source.relative_to(bench/'assets/hookrack')))
    for source in (bench/'bddl').glob('*.bddl'):pairs.append((source,package/'bddl_files/libero_hookrack'/source.name))
    pairs.extend([(bench/'extension/hookrack_objects.py',package/'envs/objects/hookrack_objects.py'),
                  (bench/'extension/hookrack_manipulation.py',package/'envs/problems/hookrack_manipulation.py')])
    registrations=[package/p for p in ('envs/objects/__init__.py','envs/problems/__init__.py','benchmark/__init__.py','benchmark/libero_suite_task_map.py')]
    targets=[p for source,p in pairs if p.exists() and sha(source)!=sha(p)]+registrations
    print(f'LIBERO package: {package}\nBundle files: {len(pairs)}\nBackup existing differing targets: {len(set(targets))}')
    if not a.apply:
        print('Preview only. Add --apply to install. No saved reset states are required by the current expert.')
        return
    backup=ROOT/'.deployment_backups'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');backup.mkdir(parents=True,exist_ok=False)
    saved=[]
    for i,p in enumerate(sorted(set(targets))):
        if not p.is_file():continue
        dest=backup/f'{i:05d}.bin';shutil.copy2(p,dest);saved.append({'target':str(p),'backup':dest.name})
    (backup/'restore_manifest.json').write_text(json.dumps({'files':saved},indent=2),encoding='utf-8')
    for source,target in pairs[:len(manifest['files_sha256'])]:
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    subprocess.run([sys.executable,str(bench/'install.py'),'--variant','pro','--preserve-config'],check=True)
    print(f'Installed; no files deleted. Backup: {backup}')

if __name__=='__main__':main()
