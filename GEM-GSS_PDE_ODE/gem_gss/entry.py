"""Portable packaging front end for the two archived PDE source profiles.

Select a profile before importing algorithm modules. Six original 0.4.7 files
override their 0.4.9 counterparts; the 25 shared modules remain single copies.
No fitting, graph-construction or search-selection rule is implemented here.
"""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / 'assets/release_manifest.json'
_profile = None


def manifest():
    return json.loads(MANIFEST.read_text(encoding='utf8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_name():
    return _profile or 'pde049'


def source_path(name, profile=None):
    profile = profile or profile_name()
    override = PACKAGE / 'versions/pde047' / name
    return override if profile == 'pde047' and override.is_file() else PACKAGE / name


def active_source_hashes():
    names = manifest()['source_profiles'][profile_name()]
    return {name: digest(source_path(name)) for name in names}


def configure(profile, data_dir):
    global _profile
    if profile not in ('pde047', 'pde049'):
        raise ValueError('Unknown historical source profile')
    already = sys.modules.get('gem_gss.core')
    if already is not None and Path(already.__file__).resolve() != source_path('core.py', profile).resolve():
        raise RuntimeError('Select the source profile before importing gem_gss.core; use a fresh process to switch')
    package = importlib.import_module('gem_gss')
    package.__path__[:] = ([str(PACKAGE / 'versions/pde047'), str(PACKAGE)]
                          if profile == 'pde047' else [str(PACKAGE)])
    for statement in ast.parse(source_path('__init__.py', profile).read_text(encoding='utf8')).body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__version__' for t in statement.targets):
            package.__version__ = ast.literal_eval(statement.value)
    _profile = profile
    core = importlib.import_module('gem_gss.core')
    core.ROOT = PACKAGE
    core.ASSETS = PACKAGE / 'assets'
    core.WORKSPACE = Path(data_dir).expanduser().resolve()
    return core


def cache_directory():
    root = Path(os.environ.get('GEM_GSS_CACHE_DIR', str(Path.cwd() / '.gem_gss_cache')))
    return root.expanduser().resolve() / profile_name()


def validate_backend(backend):
    if backend == 'reference':
        return
    if backend != 'compiled_chain_2d' or profile_name() != 'pde047':
        raise ValueError('The archived compiled backend belongs to T2D / pde047; use reference for other historical profiles')
    expected = manifest()['source_profiles'][profile_name()]
    numerical = ['core.py', 'fitting.py', 'chain_numerics.py', 'derivative_program.py', 'numerics.py']
    if any(digest(source_path(name)) != expected[name] for name in numerical):
        raise RuntimeError('Numerical source differs from the frozen release manifest')


def check_release():
    record = manifest()
    source_tree = (PACKAGE.parent / 'pyproject.toml').is_file()
    checked, missing, modified = 0, [], []
    for relative, expected in record['files'].items():
        if not source_tree and not relative.startswith('gem_gss/'):
            continue
        path = PACKAGE.parent / relative
        checked += 1
        if not path.is_file():
            missing.append(relative)
        elif digest(path) != expected:
            modified.append(relative)
    report = dict(mode='source' if source_tree else 'installed', source_file_count=record['source_file_count'],
                  maximum_files=50, checked_hashes=checked, missing=missing, modified=modified,
                  results_included=False, observation_data_included=False, API_calls=0)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing or modified or record['source_file_count'] > 50:
        raise SystemExit(1)


def doctor(core, selected):
    cases = core.read(core.ASSETS / 'cases.json')
    rows = []
    for case in cases:
        if selected and case['id'] != selected:
            continue
        path = core.WORKSPACE / case['data_path']
        row = dict(case=case['id'], data_path=str(path), data_available=path.is_file())
        if path.is_file():
            data = core.Data(case)
            graph = core.initial_graph(case)
            row.update(observation_hash_verified=True, shape=list(data.shape),
                       candidates=sum(bool(n['candidate']) for n in graph['nodes']))
        rows.append(row)
    if not rows:
        raise ValueError('Unknown case')
    print(json.dumps(dict(source_profile=profile_name(), cases=rows, API_calls=0, fits=0), ensure_ascii=False, indent=2))
    if any(not row['data_available'] for row in rows):
        raise SystemExit(2)


def option(arguments, flag):
    for i, value in enumerate(arguments):
        if value == flag and i+1 < len(arguments):
            return arguments[i+1]
        if value.startswith(flag+'='):
            return value.split('=', 1)[1]
    return None


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--profile', choices=['auto', 'pde047', 'pde049'], default='auto')
    common.add_argument('--data-dir', default=os.environ.get('GEM_GSS_DATA_DIR', str(Path.cwd()/'data')))
    known, rest = common.parse_known_args(arguments)
    command = rest[0] if rest else None
    if command == 'check':
        check_release()
        return
    caseid = option(rest, '--case')
    definitions = {row['id']: row for row in manifest()['cases']}
    profile = definitions.get(caseid, {}).get('profile', 'pde049') if known.profile == 'auto' else known.profile
    core = configure(profile, known.data_dir)
    if command == 'doctor' and '--help' not in rest:
        doctor(core, caseid)
        return
    if command == 'run':
        # Use the benchmark's original flags, not the archived CLI's older
        # convenience defaults. The budget remains TOTAL 25 across outputs.
        if option(rest, '--coverage-schedule') is None:
            rest += ['--coverage-schedule', 'global_1pct']
        if '--compact-prompt' not in rest:
            rest += ['--compact-prompt']
        if option(rest, '--numerical-backend') is None:
            rest += ['--numerical-backend', definitions.get(caseid, {}).get('settings', {}).get('numerical_backend', 'reference')]
    importlib.import_module('gem_gss.cli').main(rest)


if __name__ == '__main__':
    main()
