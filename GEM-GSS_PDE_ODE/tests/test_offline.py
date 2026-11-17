"""Offline export/profile checks; no benchmark results or network required."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def child(code):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', GEM_GSS_API_KEY='', OPENAI_API_KEY='')
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, env=env,
                            capture_output=True, text=True, encoding='utf8', errors='replace')
    if result.returncode:
        raise AssertionError(result.stdout+'\n'+result.stderr)
    return result.stdout


class OfflineRelease(unittest.TestCase):
    def test_release_hashes_and_file_count(self):
        child('from gem_gss.entry import check_release; check_release()')
        records = json.loads((ROOT/'gem_gss/assets/release_manifest.json').read_text(encoding='utf8'))
        self.assertLessEqual(records['source_file_count'], 50)
        self.assertFalse(records['results_included'])
        self.assertFalse(records['observation_data_included'])

    def test_archived_math_sources(self):
        from gem_gss.entry import manifest, source_path, digest
        record = manifest()
        packaging = {'cli.py', '__main__.py', 'chain_numerics.py'}
        for profile, sources in record['original_source_sha256'].items():
            for name, expected in sources.items():
                if name not in packaging:
                    self.assertEqual(digest(source_path(name, profile)), expected, (profile, name))

    def test_all_eight_frozen_graphs(self):
        child('''
import hashlib,json
from gem_gss.entry import configure,manifest
C=configure('pde049','unused-data')
graphs=C.bundled_graphs()
for case in manifest()['cases']:
 assert hashlib.sha256(json.dumps(graphs[case['id']],sort_keys=True).encode()).hexdigest()==case['graph_sha256']
assert len(graphs)==8
''')

    def test_profile_selection(self):
        for profile, version in [('pde047','0.4.7'),('pde049','0.4.9')]:
            code='''
from gem_gss.entry import configure,active_source_hashes,manifest
import gem_gss
C=configure(PROFILE,'unused-data')
assert gem_gss.__version__.startswith(VERSION)
assert active_source_hashes()==manifest()['source_profiles'][PROFILE]
assert C.SETTINGS['history_fits_per_round']==25
try:
 configure('pde049' if PROFILE=='pde047' else 'pde047','unused-data')
except RuntimeError:
 pass
else:
 raise AssertionError('Mixing profiles must fail')
'''.replace('PROFILE',repr(profile)).replace('VERSION',repr(version))
            child(code)

    def test_twenty_round_loop_without_http(self):
        for profile in ['pde047','pde049']:
            code='''
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch
from gem_gss.entry import configure
C=configure(PROFILE,'unused-data')
from gem_gss.builder import Graph
from gem_gss import search
from gem_gss.final_selection import ASSESSMENT_KEYS
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);x=np.linspace(0,1,9);t=np.linspace(0,1,9)
 path=root/'observations.npz';np.savez(path,x=x,t=t,u=1+.25*x[:,None]+2*t[None,:])
 graph=Graph(['u'],['x'])
 for expression in ['u','1']:
  graph.add_expr(graph.language.parse(expression),'offline_test')
 gp=root/'graph.json';C.dump(gp,dict(nodes=graph.nodes,edges=list(graph.edges.values())))
 case=dict(id='OFFLINE',fields=['u'],coordinates=['x'],prior='Discover the observed temporal rate.',
           data_path=str(path),data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),graph_path=str(gp),metadata={})
 C.SETTINGS.update(rounds=20,history_fits_per_round=1,coverage_schedule='global_1pct',minimum_test_fraction=.1,compact_prompt=True)
 calls=[]
 def fake_once(prompt,folder,number):
  calls.append(number)
  policy={}
  if number==20:
   cards=C.read(folder.parent/'checkpoint.json')['pending']['cards']
   chosen={f:min(rows,key=lambda a:a['normalized_MSE'])['candidate'] for f,rows in cards.items()}
   policy=dict(selected=chosen,assessment={f:{k:'Offline mock assessment.' for k in ASSESSMENT_KEYS} for f in cards})
  record=dict(status='accepted',round=number,http_attempts=0,policy=policy)
  C.dump(folder/('round%02d_record.json'%number),record)
  return policy,record
 search.once=fake_once
 with patch('requests.sessions.Session.request',side_effect=AssertionError('Network is disabled')):
  result=search.run(case,root/'run')
  assert result['completed_rounds']==20 and result['total_API_calls']==0
  assert result['equations']['u']['MSE']<1e-20
  assert len(calls)==21
  assert search.run(case,root/'run')==result
  assert len(calls)==21
'''.replace('PROFILE',repr(profile))
            child(code)

    def test_compiled_derivative_matches_reference(self):
        child('''
import hashlib,tempfile
from pathlib import Path
import numpy as np
from gem_gss.entry import configure,validate_backend
C=configure('pde047','unused-data');validate_backend('compiled_chain_2d')
from gem_gss.chain_numerics import ChainData
with tempfile.TemporaryDirectory() as temp:
 path=Path(temp)/'data.npz';x=np.linspace(-1,1,9);y=np.linspace(-1,1,9);t=np.linspace(0,1,8)
 u=.5+.1*x[:,None,None]+.03*y[None,:,None]**2+.02*t[None,None,:]
 np.savez(path,x=x,y=y,t=t,u=u)
 case=dict(id='OFFLINE2D',fields=['u'],coordinates=['x','y'],data_path=str(path),
           data_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 data=ChainData(case,prepare=False)
 node=dict(expr='PD((a*(u*u)),(1,0))',params=['a'])
 a=data.feature(node,[.7]);b=data.reference.feature(node,[.7])
 assert np.linalg.norm(a-b)/np.linalg.norm(b)<1e-9
''')


if __name__ == '__main__':
    unittest.main()
