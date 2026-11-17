"""Compiled continuous spatial chain rules on shared observed-field spline jets.

Fixed nodes keep their original discrete feature vectors. Internal parameters
remain independent per mechanism in the unchanged variable-projection fitter.
Only non-smooth/nonfinite parameter branches use the old discrete evaluator.
"""
from collections import Counter, OrderedDict
from functools import lru_cache
import time
import numpy as np
from scipy.interpolate import make_interp_spline
from . import core as C
from .derivative_program import DAG


class ChainData(C.Data):
    def __init__(self,case,prepare=True):
        if len(case['coordinates'])!=2 or case['fields']!=['u']:
            raise ValueError('Compiled chain backend currently supports scalar 2D only')
        super().__init__(case)
        self.reference=C.Data(case);self.dag=DAG();self.roots={};self.kernels={}
        self.pinned_jets={(0,0):self.full['u']};self.constants=OrderedDict();self.constant_bytes=0
        self.fixed_vectors={};self.stats=Counter();self.guards={}
        if prepare:self.prepare()

    def prepare(self):
        begin=time.perf_counter();graph=C.initial_graph(self.case)
        candidates=[n for n in graph['nodes'] if n['candidate']]
        for i,node in enumerate(candidates):
            self.program(node['expr'])
            if (i+1)%4000==0:print('PREPARE DERIVATIVE PROGRAMS',i+1,'/',len(candidates),flush=True)
        for op,args,value in self.dag.nodes:
            if op=='Jet':self.jet(value)
        self.stats['prepare_program_seconds']=time.perf_counter()-begin
        self.stats['prepared_initial_roots']=len(candidates)
        begin=time.perf_counter()
        from .entry import cache_directory
        cache=cache_directory()/'T2D_fixed_operator_vectors.npz'
        cache.parent.mkdir(parents=True,exist_ok=True)
        identity=self.case['data_sha256']+C.sha(C.ROOT/'core.py')+C.sha(C.ASSETS/'initial_graphs.json.gz')
        fixed={n['expr'] for n in candidates if not n['params']}
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                if str(saved['identity'])==identity:
                    self.fixed_vectors={str(e):v.copy() for e,v in zip(saved['expressions'],saved['vectors'])}
                    self.fixed_vectors.update({str(e):None for e in saved['invalid']})
            if set(self.fixed_vectors)!=fixed:self.fixed_vectors={}
        loaded=bool(self.fixed_vectors)
        for node in candidates:
            if not node['params']:
                if node['expr'] in self.fixed_vectors:continue
                try:self.fixed_vectors[node['expr']]=self.reference.feature(node,[])
                except (ValueError,ArithmeticError,np.linalg.LinAlgError):self.fixed_vectors[node['expr']]=None
        if not loaded:
            exprs=[e for e,v in self.fixed_vectors.items() if v is not None]
            np.savez_compressed(cache,identity=identity,expressions=np.asarray(exprs),
                vectors=np.vstack([self.fixed_vectors[e] for e in exprs]),
                invalid=np.asarray([e for e,v in self.fixed_vectors.items() if v is None]))
        self.stats['prepare_fixed_vectors_seconds']=time.perf_counter()-begin
        self.stats['prepared_fixed_vectors']=sum(v is not None for v in self.fixed_vectors.values())
        self.stats['fixed_vectors_loaded_from_disk']=int(loaded)

    def jet(self,alpha):
        if alpha not in self.pinned_jets:
            begin=time.perf_counter();value=self.full['u']
            for axis,n in enumerate(alpha):
                if n:value=make_interp_spline(self.axes[axis],value,k=5,axis=axis)(self.axes[axis],nu=n)
            self.pinned_jets[alpha]=value;self.stats['base_jet_computations']+=1
            self.stats['base_jet_seconds']+=time.perf_counter()-begin
        return self.pinned_jets[alpha]

    @lru_cache(None)
    def deps(self,node):
        op,args,value=self.dag.nodes[node]
        if op=='Parameter':return frozenset([value])
        return frozenset().union(*(self.deps(i) for i in args))

    def constant(self,node):
        op,args,value=self.dag.nodes[node]
        if op=='Jet':return self.jet(value)
        if op=='Constant':return value
        if op=='Coordinate':
            axis=2 if value=='t' else self.coordinates.index(value)
            a=self.t if axis==2 else self.axes[axis];shape=[1,1,1];shape[axis]=len(a)
            return a.reshape(shape)
        if node in self.constants:
            self.constants.move_to_end(node);self.stats['constant_cache_hits']+=1;return self.constants[node]
        result=self.kernel(node)({})
        if isinstance(result,np.ndarray):
            self.constants[node]=result;self.constant_bytes+=result.nbytes
            while self.constant_bytes>256*1024**2:
                _,old=self.constants.popitem(last=False);self.constant_bytes-=old.nbytes
        return result

    def kernel(self,node):
        if node in self.kernels:return self.kernels[node]
        order=[];seen=set();leaves=set()
        def visit(i):
            if i in seen:return
            seen.add(i);op,args,value=self.dag.nodes[i]
            if op=='Constant':return
            if op=='Parameter':return
            if not self.deps(i) and i!=node:leaves.add(i);return
            if op in {'Jet','Coordinate'}:leaves.add(i);return
            for child in args:visit(child)
            order.append(i)
        visit(node)
        def var(i):
            op,args,value=self.dag.nodes[i]
            if op=='Constant':return repr(value)
            if op=='Parameter':return 'p['+repr(value)+']'
            return 'v'+str(i)
        lines=['def run(p):']
        for i in sorted(leaves):lines.append('    '+var(i)+' = constant('+str(i)+')')
        uses=Counter(c for i in order for c in self.dag.nodes[i][1])
        fnmap={'Exp':'exp','Log':'log','Sin':'sin','Cos':'cos','Tan':'tan','ArcSin':'arcsin',
            'ArcCos':'arccos','ArcTan':'arctan','Sinh':'sinh','Cosh':'cosh','Tanh':'tanh',
            'ArcSinh':'arcsinh','ArcCosh':'arccosh','ArcTanh':'arctanh','Sqrt':'sqrt','CubeRoot':'cbrt'}
        for i in order:
            op,args,value=self.dag.nodes[i];a=[var(c) for c in args]
            if op in {'Add','Subtract','Multiply','Divide'}:
                symbol={'Add':'+','Subtract':'-','Multiply':'*','Divide':'/'}[op];expression=f'({a[0]} {symbol} {a[1]})'
            elif op=='Negate':expression='(-'+a[0]+')'
            elif op=='Power':
                exponent=self.dag.nodes[args[1]]
                if exponent[0]=='Constant' and exponent[2]==2:expression='np.square('+a[0]+')'
                elif exponent[0]=='Constant' and exponent[2]==-1:expression='np.reciprocal('+a[0]+')'
                elif exponent[0]=='Constant' and exponent[2]==.5:expression='np.sqrt('+a[0]+')'
                elif exponent[0]=='Constant' and exponent[2]==-.5:expression='np.reciprocal(np.sqrt('+a[0]+'))'
                else:expression='np.power('+','.join(a)+')'
            else:expression='np.'+fnmap[op]+'('+a[0]+')'
            lines.append('    '+var(i)+' = '+expression)
            for child in set(args):
                uses[child]-=args.count(child)
                if uses[child]==0 and child!=node and self.dag.nodes[child][0] not in {'Constant','Parameter'}:
                    lines.append('    del '+var(child))
        lines.append('    return '+var(node))
        if not order and node not in leaves and self.dag.nodes[node][0] not in {'Constant','Parameter'}:
            raise ValueError('Empty compiled expression')
        scope={'np':np,'constant':self.constant}
        exec(compile('\n'.join(lines),'<GEM-precomputed-spatial-derivative>','exec'),scope)
        self.kernels[node]=scope['run'];self.stats['compiled_array_kernels']+=1
        return scope['run']

    def program(self,expr):
        if expr in self.roots:return self.roots[expr]
        e=self.language.parse(expr);root=self.dag.convert(e);self.roots[expr]=root
        self.kernel(root)
        obligations=[]
        def visit(z):
            if z.op in {'Log','Sqrt','ArcSin','ArcCos','ArcCosh','ArcTanh','CubeRoot','Tan'}:
                obligations.append((z.op,self.dag.convert(z.args[0])))
            if z.op=='Divide':obligations.append(('Divide',self.dag.convert(z.args[1])))
            for child in z.args:visit(child)
        visit(e);self.guards[expr]=tuple(dict.fromkeys(obligations))
        return root

    def singular(self,expr,p):
        for op,arg in self.guards[expr]:
            v=self.constant(arg) if not self.deps(arg) else self.kernel(arg)(p)
            lo=float(np.min(v));hi=float(np.max(v))
            if not np.isfinite(lo+hi):return True
            if op in {'CubeRoot','Divide'} and lo<=0<=hi:return True
            if op in {'Log','Sqrt'} and lo<=0:return True
            if op in {'ArcSin','ArcCos','ArcTanh'} and (lo<=-1 or hi>=1):return True
            if op=='ArcCosh' and lo<=1:return True
            if op=='Tan' and np.floor((lo+np.pi/2)/np.pi)!=np.floor((hi+np.pi/2)/np.pi):return True
        return False

    def feature(self,node,params,full=False):
        self.feature_calls+=1;expr=node['expr'];key=(expr,tuple(map(float,params)),full)
        if full:raise ValueError('Compiled backend uses the full observation-window objective only')
        if not node['params']:
            if expr not in self.fixed_vectors:
                try:self.fixed_vectors[expr]=self.reference.feature(node,[])
                except (ValueError,ArithmeticError,np.linalg.LinAlgError):self.fixed_vectors[expr]=None
            value=self.fixed_vectors[expr]
            if value is None:raise ValueError('Fixed operator outside finite domain')
            self.stats['fixed_vector_hits']+=1;return value
        if key in self.feature_cache:
            self.feature_cache.move_to_end(key);self.stats['parameter_vector_cache_hits']+=1;return self.feature_cache[key]
        root=self.program(expr);p=dict(zip(node['params'],params));begin=time.perf_counter()
        with np.errstate(all='ignore'):
            try:
                if self.singular(expr,p):raise ValueError('Non-smooth original-expression branch')
                raw=np.broadcast_to(self.kernels[root](p),self.shape)
                if not np.isfinite(raw).all():raise ValueError('Nonfinite chain result')
                value=self.window(raw).ravel()
                if not np.isfinite(value).all():raise ValueError('Nonfinite window')
                self.stats['chain_evaluations']+=1
            except (ValueError,ArithmeticError,np.linalg.LinAlgError):
                self.stats['singular_or_nonfinite_original_fallbacks']+=1
                value=self.reference.feature(node,params)
        self.stats['parameter_feature_seconds']+=time.perf_counter()-begin
        self.feature_cache[key]=value;self.featurebytes+=value.nbytes
        while self.featurebytes>128*1024**2:
            _,old=self.feature_cache.popitem(last=False);self.featurebytes-=old.nbytes
        return value

    def reference_feature_for_proposals(self,node,params,full=False):
        self.feature_calls+=1;self.stats['original_history_probe_calls']+=1
        return self.reference.feature(node,params,full)
