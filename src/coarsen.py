"""Exact-mass Legendre p-transfer for first-order derivative constraints.

All coefficients use P_n(1)=1 on [-1,1]. No SpECTRE dependency.
"""
from functools import lru_cache
import numpy as np


def mass(p):
    return 2.0 / (2 * np.arange(p + 1) + 1)


def derivative(p):
    i, j = np.indices((p + 1, p + 1))
    return np.where((j > i) & ((j - i) % 2 == 1), 2 * i + 1, 0.).astype(float)


def tail_traces(a, p):
    n = np.arange(p + 1, len(a))
    return np.array([np.sum(a[p + 1:]), np.sum(a[p + 1:] * (-1.) ** n)])


def lift(p):
    k = np.arange(p + 1)
    return np.column_stack(((2 * k + 1) / 2, -(2 * k + 1) / 2 * (-1.) ** k))


def injection(a, p):
    """P_p d(a)-d(P_p a), computed with two endpoint tail sums."""
    return lift(p) @ tail_traces(a, p)


@lru_cache(None)
def response(p, length=1.):
    d, m = derivative(p), mass(p)
    h = np.diag(m) + length**2 * d.T @ (m[:, None] * d)
    return np.linalg.solve(h, length**2 * d.T @ (m[:, None] * lift(p)))


def graph_transfer(a, b, p, length=1.):
    """Nearest pair (u,du) in ||u-a||²+length²||du-b||².

    Valid for constrained and unconstrained source pairs. The fast
    two-trace specialization is separate, to expose its precondition.
    """
    d, m = derivative(p), mass(p)
    h = np.diag(m) + length**2 * d.T @ (m[:, None] * d)
    u = np.linalg.solve(h, m * a[:p+1] + length**2 * d.T @ (m * b[:p+1]))
    return u, d @ u


def compatible_transfer(a, p, length=1.):
    """Fast graph transfer when b=d(a) before coarsening."""
    u = a[:p+1] + response(p, length) @ tail_traces(a, p)
    return u, derivative(p) @ u


def embed(a, q):
    return np.pad(a, (0, q + 1 - len(a)))


def pair_norm2(a, b, length=1.):
    return float(np.dot(mass(len(a)-1), a*a + length**2*b*b))


def pair_error2(a, b, u, v, length=1.):
    q = len(a)-1
    return pair_norm2(a-embed(u,q), b-embed(v,q), length)


def tolerance_degree(a, epsilon, minimum=1):
    """Smallest p passing an absolute transfer-injection tolerance."""
    q = len(a)-1
    for p in range(minimum,q):
        if np.linalg.norm(np.sqrt(mass(p))*injection(a,p)) <= epsilon:
            return p
    return q
