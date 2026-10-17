"""Reproduce all reported numerical data and publication figures."""
import sys,json,csv,platform,time
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as L
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from coarsen import *
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
for name in ('results','figures'):(ROOT/name).mkdir(exist_ok=True)


def write_csv(name,rows):
    with (ROOT/'results'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def algebra():
    rng=np.random.default_rng(6602027);rows=[]
    for q in (8,16,24,32,48):
        for p in sorted(set((1,2,4,q//2,q-1))):
            if p>=q:continue
            for length in (.03,.3,3.):
                for sample in range(40):
                    a=rng.normal(size=q+1)/(1+np.arange(q+1))**2
                    b=derivative(q)@a
                    u,v=compatible_transfer(a,p,length)
                    ug,vg=graph_transfer(a,b,p,length)
                    direct=b[:p+1]-derivative(p)@a[:p+1]
                    m=mass(p);n=p+1;t=tail_traces(a,p)
                    formula=n*n/2*np.dot(t,t)-(-1.)**p*n*t[0]*t[1]
                    reset=pair_error2(a,b,a[:p+1],derivative(p)@a[:p+1],length)
                    optimum=pair_error2(a,b,u,v,length)
                    norm=pair_norm2(a,b,length)
                    rows.append(dict(q=q,p=p,length=length,sample=sample,
                      injection_relative_residual=float(np.linalg.norm(direct-injection(a,p))/(1+np.linalg.norm(direct))),
                      norm_formula_relative_residual=float(abs(formula-np.dot(m,direct**2))/(1+abs(formula))),
                      response_relative_residual=float(np.linalg.norm(u-ug)/(1+np.linalg.norm(ug))),
                      pythagoras_relative_residual=float(abs(norm-pair_norm2(u,v,length)-optimum)/(1+norm)),
                      graph_energy_ratio=pair_norm2(u,v,length)/norm,
                      reset_energy_ratio=pair_norm2(a[:p+1],derivative(p)@a[:p+1],length)/norm,
                      graph_over_reset_error=float(optimum/reset)))
    write_csv('algebra.csv',rows)
    return dict(cases=len(rows),max_injection_residual=max(r['injection_relative_residual'] for r in rows),
      max_norm_formula_residual=max(r['norm_formula_relative_residual'] for r in rows),
      max_response_residual=max(r['response_relative_residual'] for r in rows),
      max_pythagoras_residual=max(r['pythagoras_relative_residual'] for r in rows),
      max_graph_energy_ratio=max(r['graph_energy_ratio'] for r in rows),
      max_reset_energy_ratio=max(r['reset_energy_ratio'] for r in rows),
      max_graph_over_reset_error=max(r['graph_over_reset_error'] for r in rows))


def analytic_tests():
    rows=[]
    q=48;x,w=L.leggauss(96);V=L.legvander(x,q)
    funcs={'sine':lambda x:np.sin(3*x+.3),'gaussian':lambda x:np.exp(-20*(x-.17)**2),'boundary_layer':lambda x:np.exp(8*(x-1))}
    for name,f in funcs.items():
        a=(V.T@(w*f(x)))/mass(q); b=derivative(q)@a
        for p in range(2,25):
            raw=(a[:p+1],b[:p+1]);reset=(a[:p+1],derivative(p)@a[:p+1]);graph=compatible_transfer(a,p)
            for method,(u,v) in [('independent',raw),('reset',reset),('graph',graph)]:
                rows.append(dict(profile=name,p=p,method=method,error=np.sqrt(pair_error2(a,b,u,v)),constraint=np.sqrt(np.dot(mass(p),(derivative(p)@u-v)**2))))
    write_csv('analytic.csv',rows)
    fig,axs=plt.subplots(1,3,figsize=(11,3.2))
    styles={'independent':('o','#6e7781'),'reset':('s','#b95336'),'graph':('^','#176c8c')}
    for ax,name in zip(axs,funcs):
        for method,(marker,color) in styles.items():
            rr=[r for r in rows if r['profile']==name and r['method']==method]
            ax.semilogy([r['p'] for r in rr],[r['error'] for r in rr],label=method,color=color,marker=marker,markevery=4,ms=4)
        ax.set(title=name.replace('_',' '),xlabel='Target degree p');ax.grid(alpha=.2)
    axs[0].set_ylabel('Pair error (exact-mass norm)');axs[-1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'figures/analytic.pdf');plt.close(fig)


def cancellation_figure():
    a=np.zeros(17);a[9]=1;a[11]=-1
    rows=[]
    for p in range(1,17):
        inj=np.sqrt(np.dot(mass(p),injection(a,p)**2)) if p<16 else 0.
        rows.append(dict(p=p,injection=inj,tail=np.sqrt(np.dot(mass(16)[p+1:],a[p+1:]**2))))
    write_csv('cancellation.csv',rows)
    fig,axs=plt.subplots(1,2,figsize=(9,3.1))
    p,q=10,24;C=derivative(q)[:p+1,p+1:];normalized=np.sqrt(mass(p))[:,None]*C/np.sqrt(mass(q)[p+1:])[None,:]
    axs[0].semilogy(np.arange(1,len(np.linalg.svd(normalized,compute_uv=False))+1),np.maximum(np.linalg.svd(normalized,compute_uv=False),1e-15),'o-',color='#176c8c')
    axs[0].set(xlabel='Singular-value index',ylabel='Singular value',title='Derivative-transfer commutator (24 to 10)');axs[0].grid(alpha=.2)
    axs[1].plot([r['p'] for r in rows],[r['injection'] for r in rows],'o-',color='#b95336');axs[1].set(xlabel='Target degree p',ylabel='Injected constraint norm',title='Cancellation: f = P9 - P11');axs[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'figures/boundary.pdf');plt.close(fig)


def dg_rhs(y,h):
    # y: (element, field psi/pi/phi, mode); upwind flux for A=[[0,1],[1,0]].
    p=y.shape[-1]-1;d=derivative(p);m=mass(p);r=np.ones(p+1);l=(-1.)**np.arange(p+1)
    u=y[:,1:,:];flux=u[:,::-1,:]
    left=u@l;right=u@r;neighbor=np.roll(left,-1,axis=0)
    star=.5*(right[:,::-1]+neighbor[:,::-1])-.5*(neighbor-right)
    starleft=np.roll(star,1,axis=0)
    dudt=-2/h*(flux@d.T+((star-right[:,::-1])[:,:,None]*r-(starleft-left[:,::-1])[:,:,None]*l)/m)
    out=np.zeros_like(y);out[:,0,:]=-y[:,1,:];out[:,1:,:]=dudt
    return out


def advance(y,h,duration,dt):
    n=int(np.ceil(duration/dt));dt=duration/n
    for _ in range(n):
        k1=dg_rhs(y,h);k2=dg_rhs(y+.5*dt*k1,h);k3=dg_rhs(y+.5*dt*k2,h);k4=dg_rhs(y+dt*k3,h)
        y=y+dt/6*(k1+2*k2+2*k3+k4)
    return y


def dg_experiment():
    rows=[];controls=[]
    # smooth periodic wave; this is a scalar-wave DG test, not Einstein evolution.
    for elements in (4,8,16):
        h=2*np.pi/elements;q=10;x,w=L.leggauss(32);V=L.legvander(x,q)
        coords=h/2*x[None,:]+h*(np.arange(elements)[:,None]+.5)
        y=np.zeros((elements,3,q+1))
        y[:,0,:]=(np.sin(coords)@(w[:,None]*V))/mass(q)
        y[:,1,:]=(np.cos(coords)@(w[:,None]*V))/mass(q)
        y[:,2,:]=2/h*y[:,0,:]@derivative(q).T
        y=advance(y,h,.2,.02*h/(q+1)**2)
        control=advance(y.copy(),h,.3,.02*h/(q+1)**2)
        vc=L.legvander(x,q)
        truth=np.stack([np.sin(coords-.5),np.cos(coords-.5),np.cos(coords-.5)],axis=1)
        ce=np.sqrt(h/2*np.sum((control@vc.T-truth)**2*w))
        cc=2/h*control[:,0,:]@derivative(q).T-control[:,2,:]
        controls.append(dict(elements=elements,p=q,final_error=float(ce),final_constraint=float(np.sqrt(h/2*np.sum(cc**2*mass(q))))))
        for p in (3,5,7):
            for method in ('independent','reset','graph'):
                z=y[:,:,:p+1].copy()
                for e in range(elements):
                    if method=='reset':z[e,2]=2/h*derivative(p)@z[e,0]
                    if method=='graph':
                        # physical phi becomes logical b=(h/2)phi; length=2/h
                        u,v=graph_transfer(y[e,0],h/2*y[e,2],p,2/h)
                        z[e,0]=u;z[e,2]=2/h*v
                initial_c=2/h*z[:,0,:]@derivative(p).T-z[:,2,:]
                for factor in (1.,.5):
                    end=advance(z.copy(),h,.3,factor*.02*h/(p+1)**2)
                    xe,we=L.leggauss(40);ve=L.legvander(xe,p)
                    xx=h/2*xe[None,:]+h*(np.arange(elements)[:,None]+.5)
                    truth=np.stack([np.sin(xx-.5),np.cos(xx-.5),np.cos(xx-.5)],axis=1)
                    pred=end@ve.T
                    error=np.sqrt(h/2*np.sum((pred-truth)**2*we))
                    constraint=2/h*end[:,0,:]@derivative(p).T-end[:,2,:]
                    cn=np.sqrt(h/2*np.sum(constraint**2*mass(p)))
                    rows.append(dict(elements=elements,p=p,method=method,dt_factor=factor,final_error=float(error),final_constraint=float(cn),post_transfer_constraint=float(np.sqrt(h/2*np.sum(initial_c**2*mass(p))))))
    write_csv('dg_wave.csv',rows)
    write_csv('dg_uncoarsened_controls.csv',controls)
    fig,axs=plt.subplots(1,2,figsize=(9,3.1))
    for method,color in [('independent','#6e7781'),('reset','#b95336'),('graph','#176c8c')]:
        rr=[r for r in rows if r['p']==3 and r['method']==method and r['dt_factor']==.5]
        axs[0].loglog([r['elements'] for r in rr],[r['final_error'] for r in rr],'o-',label=method,color=color)
        axs[1].loglog([r['elements'] for r in rr],[r['final_constraint'] for r in rr],'o-',label=method,color=color)
    axs[0].set(xlabel='Number of elements',ylabel='Final three-field L2 error',title='Periodic scalar wave: degree 10 to 3')
    axs[1].set(xlabel='Number of elements',ylabel='Final reduction-constraint norm',title='Constraint after subsequent DG evolution')
    for ax in axs:ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/dg.pdf');plt.close(fig)
    return rows


if __name__=='__main__':
    start=time.perf_counter();summary=algebra();analytic_tests();cancellation_figure();dg=dg_experiment()
    summary.update(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),elapsed_seconds=time.perf_counter()-start)
    summary['dg_cases']=len(dg)
    summary['dg_max_timestep_relative_change']=max(abs(r['final_error']-next(s['final_error'] for s in dg if s['elements']==r['elements'] and s['p']==r['p'] and s['method']==r['method'] and s['dt_factor']==.5))/r['final_error'] for r in dg if r['dt_factor']==1.)
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
