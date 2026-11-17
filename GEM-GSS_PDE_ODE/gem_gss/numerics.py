"""Explicit numerical backend dispatch; original fitter and 1D behavior intact."""
from . import core as C, fitting
from .fused_numerics import FusedData, fit as fused_fit, final_equation as fused_final
from .chain_numerics import ChainData


def make_data(case,backend=None):
    backend=backend or C.SETTINGS.get('numerical_backend','reference')
    if backend=='reference':return C.Data(case)
    if backend=='compiled_chain_2d':return ChainData(case)
    if backend!='guarded_fused_2d':raise ValueError('Unknown numerical backend')
    if len(case['coordinates'])!=2:raise ValueError('Guarded fusion is restricted to 2D')
    return FusedData(case)


def fit(data,*args,**kwargs):
    return (fused_fit if isinstance(data,FusedData) else fitting.fit)(data,*args,**kwargs)


def final_equation(data,*args,**kwargs):
    return (fused_final if isinstance(data,FusedData) else fitting.final_equation)(data,*args,**kwargs)


def restore_models(data,archives,stats=None):
    if isinstance(data,ChainData):
        if stats:data.stats.update(stats)
        return
    if not isinstance(data,FusedData):return
    if stats:data.stats.update(stats)
    for field,archive in archives.items():
        for state in archive.states.values():
            data.register(state['model']);data.observe(field,state['model'])


def statistics(data):
    if isinstance(data,ChainData):return dict(backend='compiled_chain_2d',counts=dict(data.stats))
    if not isinstance(data,FusedData):return dict(backend='reference')
    return dict(backend='guarded_fused_2d',counts=dict(data.stats))
