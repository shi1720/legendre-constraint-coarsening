import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
from numpy.polynomial import legendre as L
from coarsen import *


class TransferTests(unittest.TestCase):
    def test_differentiation_against_numpy(self):
        for q in (2,7,16,32):
            a=np.random.default_rng(q).normal(size=q+1)
            np.testing.assert_allclose(derivative(q)@a,embed(L.legder(a),q),atol=1e-11)

    def test_boundary_factorization(self):
        rng=np.random.default_rng(4181)
        for q in range(2,33):
            a=rng.normal(size=q+1)
            for p in range(1,q):
                direct=(derivative(q)@a)[:p+1]-derivative(p)@a[:p+1]
                np.testing.assert_allclose(direct,injection(a,p),atol=1e-11)

    def test_quadrature_independent_least_squares(self):
        rng=np.random.default_rng(814)
        for q,p in [(8,3),(20,7),(32,16)]:
            x,w=L.leggauss(q+3)
            v=L.legvander(x,p)
            dv=np.column_stack([L.legval(x,L.legder(np.eye(p+1)[j])) for j in range(p+1)])
            for length in (.03,1.,30.):
                a,b=rng.normal(size=(2,q+1))
                mat=np.vstack([np.sqrt(w)[:,None]*v,length*np.sqrt(w)[:,None]*dv])
                rhs=np.r_[np.sqrt(w)*L.legval(x,a),length*np.sqrt(w)*L.legval(x,b)]
                ref=np.linalg.lstsq(mat,rhs,rcond=None)[0]
                u,_=graph_transfer(a,b,p,length)
                np.testing.assert_allclose(u,ref,atol=3e-9,rtol=3e-9)

    def test_contraction_optimality_and_fast_response(self):
        rng=np.random.default_rng(137)
        for q in (5,12,24,48):
            for p in (1,q//2,q-1):
                a=rng.normal(size=q+1)/(1+np.arange(q+1))**2
                b=derivative(q)@a
                for length in (.01,1.,10.):
                    u,v=compatible_transfer(a,p,length)
                    u2,v2=graph_transfer(a,b,p,length)
                    np.testing.assert_allclose(u,u2,atol=1e-10)
                    np.testing.assert_allclose(v,v2,atol=1e-9)
                    self.assertLessEqual(pair_norm2(u,v,length),pair_norm2(a,b,length)*(1+1e-10))
                    self.assertAlmostEqual(u[0],a[0],places=10)
                    e=pair_error2(a,b,u,v,length)
                    z=rng.normal(size=p+1)
                    zerr=pair_error2(a,b,z,derivative(p)@z,length)
                    self.assertLessEqual(e,zerr+1e-8)
                    np.testing.assert_allclose(pair_norm2(a,b,length),pair_norm2(u,v,length)+e,rtol=1e-9)

    def test_reset_energy_counterexample(self):
        a=np.array([0.,0.,1.,0.,-.3]); b=derivative(4)@a
        u=a[:3];v=derivative(2)@u
        self.assertAlmostEqual(pair_norm2(u,v)/pair_norm2(a,b),6.4/4.62)

    def test_tail_cancellation(self):
        a=np.zeros(11); a[7]=1.; a[9]=-1.
        np.testing.assert_allclose(injection(a,5),0.,atol=1e-14)
        self.assertGreater(np.linalg.norm(a[6:]),1.)

    def test_off_constraint_decomposition(self):
        a,b=np.random.default_rng(3).normal(size=(2,15));p=5
        old=derivative(14)@a-b
        new=derivative(p)@a[:p+1]-b[:p+1]
        np.testing.assert_allclose(new,old[:p+1]-injection(a,p),atol=1e-12)

    def test_sharp_commutator_singular_values(self):
        for p,q in [(1,6),(4,5),(8,16),(15,32)]:
            c=derivative(q)[:p+1,p+1:]
            cn=np.sqrt(mass(p))[:,None]*c/np.sqrt(mass(q)[p+1:])[None,:]
            values=[]
            for parity in (0,1):
                retained=np.arange(p+1);tail=np.arange(p+1,q+1)
                values.append(np.sqrt(np.sum((2*retained+1)[retained%2==parity])*np.sum((2*tail+1)[tail%2!=parity])))
            sv=np.linalg.svd(cn,compute_uv=False)
            np.testing.assert_allclose(np.sort(sv[:2]),np.sort(values) if len(sv)>1 else np.sort(values)[-1:],atol=1e-10)
            if len(sv)>2: self.assertLess(sv[2],1e-10)

    def test_tensor_face_rank(self):
        p,q=3,7
        c=derivative(q)[:p+1,p+1:]
        self.assertEqual(np.linalg.matrix_rank(np.kron(c,np.eye(p+1))),2*(p+1))


if __name__=='__main__':unittest.main(verbosity=2)
