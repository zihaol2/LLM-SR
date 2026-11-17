"""Shared parameterized spatial-derivative program; no observation or API access."""
from functools import lru_cache
import math

class DAG:
    def __init__(self):
        self.nodes=[];self.lookup={};self.diff_requests=0;self.diff_misses=0
        self.zero=self.make('Constant',value=0.);self.one=self.make('Constant',value=1.)

    def make(self,op,args=(),value=None):
        if op=='Constant': value=float(value)
        if op in {'Add','Subtract','Multiply','Divide','Power'}:
            a,b=args;aa=self.nodes[a];bb=self.nodes[b]
            if aa[0]==bb[0]=='Constant':
                x,y=aa[2],bb[2]
                try:
                    z={'Add':lambda:x+y,'Subtract':lambda:x-y,'Multiply':lambda:x*y,
                       'Divide':lambda:x/y,'Power':lambda:x**y}[op]()
                    if isinstance(z,(int,float)) and math.isfinite(z):return self.make('Constant',value=z)
                except (ValueError,ZeroDivisionError,OverflowError):pass
            az=aa[0]=='Constant' and aa[2]==0; bz=bb[0]=='Constant' and bb[2]==0
            ao=aa[0]=='Constant' and aa[2]==1; bo=bb[0]=='Constant' and bb[2]==1
            if op=='Add':
                if az:return b
                if bz:return a
            if op=='Subtract':
                if bz:return a
                if a==b:return self.zero
            if op=='Multiply':
                if az or bz:return self.zero
                if ao:return b
                if bo:return a
            if op=='Divide':
                if az:return self.zero
                if bo:return a
            if op=='Power':
                if bz:return self.one
                if bo:return a
        if op=='Negate':
            a=args[0]
            if self.nodes[a][0]=='Constant':return self.make('Constant',value=-self.nodes[a][2])
            if self.nodes[a][0]=='Negate':return self.nodes[a][1][0]
        key=(op,tuple(args),value)
        if key not in self.lookup:
            self.lookup[key]=len(self.nodes);self.nodes.append(key)
        return self.lookup[key]

    def add(self,a,b):return self.make('Add',(a,b))
    def sub(self,a,b):return self.make('Subtract',(a,b))
    def mul(self,a,b):return self.make('Multiply',(a,b))
    def div(self,a,b):return self.make('Divide',(a,b))
    def power(self,a,p):return self.make('Power',(a,self.make('Constant',value=p)))
    def fn(self,op,a):return self.make(op,(a,))

    @lru_cache(None)
    def convert(self,e):
        if e.op=='PD':
            node=self.convert(e.args[0])
            for axis,n in enumerate(e.value):
                for _ in range(n):node=self.derivative(node,axis)
            return node
        if e.op=='Field':return self.make('Jet',value=(0,0))
        return self.make(e.op,tuple(self.convert(a) for a in e.args),e.value)

    def derivative(self,node,axis):
        self.diff_requests+=1
        return self._derivative(node,axis)

    @lru_cache(None)
    def _derivative(self,node,axis):
        self.diff_misses+=1;op,args,value=self.nodes[node]
        if op in {'Constant','Parameter'}:return self.zero
        if op=='Coordinate':return self.one if value==['x','y'][axis] else self.zero
        if op=='Jet':
            alpha=list(value);alpha[axis]+=1
            return self.make('Jet',value=tuple(alpha))
        a=args[0];da=self.derivative(a,axis)
        if op=='Negate':return self.fn('Negate',da)
        if op in {'Add','Subtract','Multiply','Divide','Power'}:
            b=args[1];db=self.derivative(b,axis)
            if op=='Add':return self.add(da,db)
            if op=='Subtract':return self.sub(da,db)
            if op=='Multiply':return self.add(self.mul(da,b),self.mul(a,db))
            if op=='Divide':return self.div(self.sub(self.mul(da,b),self.mul(a,db)),self.power(b,2))
            if db==self.zero:
                return self.mul(self.mul(b,self.make('Power',(a,self.sub(b,self.one)))),da)
            return self.mul(node,self.add(self.mul(db,self.fn('Log',a)),self.div(self.mul(b,da),a)))
        if da==self.zero:return self.zero
        one=self.one
        if op=='Exp':factor=node
        elif op=='Log':factor=self.div(one,a)
        elif op=='Sin':factor=self.fn('Cos',a)
        elif op=='Cos':factor=self.fn('Negate',self.fn('Sin',a))
        elif op=='Tan':factor=self.add(one,self.power(node,2))
        elif op in {'ArcSin','ArcCos'}:
            factor=self.power(self.sub(one,self.power(a,2)),-.5)
            if op=='ArcCos':factor=self.fn('Negate',factor)
        elif op=='ArcTan':factor=self.div(one,self.add(one,self.power(a,2)))
        elif op=='Sinh':factor=self.fn('Cosh',a)
        elif op=='Cosh':factor=self.fn('Sinh',a)
        elif op=='Tanh':factor=self.sub(one,self.power(node,2))
        elif op=='ArcSinh':factor=self.power(self.add(one,self.power(a,2)),-.5)
        elif op=='ArcCosh':factor=self.div(one,self.mul(self.fn('Sqrt',self.sub(a,one)),self.fn('Sqrt',self.add(a,one))))
        elif op=='ArcTanh':factor=self.div(one,self.sub(one,self.power(a,2)))
        elif op=='Sqrt':factor=self.div(one,self.mul(self.make('Constant',value=2),node))
        elif op=='CubeRoot':factor=self.div(one,self.mul(self.make('Constant',value=3),self.power(node,2)))
        else:raise ValueError('Unsupported derivative '+op)
        return self.mul(factor,da)

