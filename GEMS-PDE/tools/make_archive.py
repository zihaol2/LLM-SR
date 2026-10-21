"""Create a clean source/reproduction archive; never include credentials/cache."""
from pathlib import Path
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT.parent/(ROOT.name+'.zip')
EXCLUDE={'.git','.venv','venv','__pycache__','build','dist','runs','.pytest_cache'}
files=[]
for path in sorted(ROOT.rglob('*')):
    if not path.is_file():continue
    rel=path.relative_to(ROOT)
    if any(p in EXCLUDE or p.endswith('.egg-info') for p in rel.parts):continue
    if path.name.startswith('.env') and path.name!='.env.example':continue
    if path.suffix in ('.pyc','.pyo','.tmp'):continue
    files.append(path)
with zipfile.ZipFile(OUTPUT,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for path in files:archive.write(path,ROOT.name+'/'+path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(OUTPUT) as archive:
    if archive.testzip() is not None:raise RuntimeError('Archive CRC check failed')
print('Archive:',OUTPUT.name)
print('Files:',len(files),'bytes:',OUTPUT.stat().st_size)
print('SHA256:',hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
