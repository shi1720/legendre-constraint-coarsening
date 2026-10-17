# Literature and overlap review

Search date: 17 October 2026. This is a targeted literature review, not an
exhaustive proof of novelty or a patent search.

## Closest prior work

1. Holst, Lindblom, Owen, Pfeiffer, Scheel and Kidder (2004), *Optimal constraint
   projection for hyperbolic evolution systems*, PRD 70, 084017.
   https://doi.org/10.1103/PhysRevD.70.084017
   Nearest-state constraint projection and scalar-wave experiments on black-hole
   backgrounds are already established. Our graph projection is a finite-dimensional
   specialization, not a new general constraint-projection method.
2. Rapisarda, van Waarde and Camlibel (2024), *Orthogonal polynomial bases for
   data-driven analysis and control of continuous-time systems*, TAC 69, 4307-4319.
   https://doi.org/10.1109/TAC.2023.3321214
   Explicit Legendre differentiation and noncommutation of differentiation and
   polynomial truncation are already discussed. We derive the endpoint
   factorization, its exact finite-band singular values, and consequences for
   first-order field transfer. The underlying identities are classical.
3. Shen (1994), *Efficient spectral-Galerkin method I*, SISC 15, 1489-1505.
   https://doi.org/10.1137/0915089
   Efficient Legendre variational solvers predate this work. No new elliptic-solver
   complexity is claimed.
4. Huynh (2014), *On formulations of discontinuous Galerkin and related methods for conservation laws*, NASA/TM-2014-218135.
   https://ntrs.nasa.gov/citations/20140009969
   Boundary correction/lifting functions are established. Our endpoint kernels
   should be understood in that tradition.
5. Canuto, Nochetto and Verani (2014), *Contraction and optimality properties of
   adaptive Legendre-Galerkin methods: the one-dimensional case*, CAMWA 67, 752-770.
   https://doi.org/10.1016/j.camwa.2013.05.025
   Adaptive spectral coarsening and optimality theory are established; this paper
   addresses a local first-order derivative consistency diagnostic instead.

## Numerical-relativity context

* Lindblom et al. (2006), CQG 23, S447-S462:
  https://doi.org/10.1088/0264-9381/23/16/S09
* Szilagyi (2014), IJMPD 23, 1430014:
  https://doi.org/10.1142/S0218271814300146
* Kidder et al. (2017), JCP 335, 84-114:
  https://doi.org/10.1016/j.jcp.2016.12.059
* SpECTRE spectral API (accessed 17 October 2026):
  https://spectre-code.org/namespaceSpectral.html

These sources motivate derivative constraints, spectral adaptivity, and relevant
software interfaces. They do not establish that current SpECTRE has the particular
failure case studied here. No bug claim is made against SpECTRE.

## Queries used

Searches included: generalized harmonic constraint-preserving p-coarsening;
spectral constraint projection; Legendre derivative projection commutator;
Legendre rank-two commutator; p-adaptive constraint-preserving wave methods;
SpECTRE projection and AMR projectors; DG/subcell reconstruction. The initial
DG/subcell idea was narrowed because substantial conservative reconstruction and
constraint-preserving reconstruction literature already exists.

## Author's earlier drafts

A separate author-supplied draft corpus was text-extracted and screened for
related numerical-method themes. No apparent duplicate numerical-relativity
p-coarsening manuscript was identified. Shared use of projection and norm
identities does not alone establish a duplicate contribution. The private draft
corpus is not distributed with this repository.

## Assessment

The defensible contribution is a focused synthesis: an exact two-trace diagnostic,
sharp finite-band amplification, nonmonotonic degree selection, a cached response
specialization, and negative/positive transfer experiments. Absolute novelty cannot
be certified by these searches. The work supports a technical workshop discussion;
the evidence does not support an industry-defining breakthrough, patentability,
or a high probability of acceptance at a major refereed conference.
