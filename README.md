# Boundary traces and derivative constraints in Legendre p-coarsening

Reproducible research accompanying Shivam Gupta's technical manuscript.
The project studies an exact transfer diagnostic, not an Einstein solver.

## Reproduce

Python 3.9+ and the packages in `requirements.txt` are required.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python run_experiments.py
cd paper
tectonic main.tex
```

The committed CSV files are the raw experiment outputs. `results/summary.json`
records versions, seed-derived experiments, residuals, and elapsed time.
Figures are regenerated as vector PDFs. The manuscript includes complete proofs.

## Main findings

* For exact unweighted Legendre projection on an affine interval, the
  differentiation/projection commutator is determined by two discarded-tail
  endpoint sums. Its rank is at most two, with explicit sharp singular values.
* Injection can be nonmonotonic as the target degree changes. A constraint
  diagnostic must be combined with an accuracy criterion.
* A cached two-response implementation realizes the standard graph-norm
  projection for constraint-consistent source data. The general solver also
  handles inconsistent data.
* Graph projection is contractive in the specified pair norm. This is not a
  physical wave-energy or full-evolution stability theorem.
* A periodic upwind DG scalar-wave experiment shows that zero constraint at
  transfer does **not** guarantee improved subsequent accuracy or constraint
  control. Ordinary fieldwise projection performs better in final field error
  in the reported degree-3 benchmark. All outcomes are retained.

## Relationship to prior work

Optimal constraint projection is established by Holst et al. (2004).
Polynomial differentiation/projection noncommutation is also established,
including in Rapisarda et al. (2024). Boundary lifts and spectral Galerkin
methods are classical. This project specializes these ideas into an exact
coarsening diagnostic and auditable numerical comparison; it does not claim
the invention of any of those broader methods. See the manuscript and
`LITERATURE_REVIEW.md` for scope and novelty limits.

## Scope

No full SpECTRE/SpEC installation, binary-black-hole simulation, curved-map
validation, gravitational-wave mismatch study, patentability determination,
or production speedup is claimed. Tensor face rank is checked algebraically
in two dimensions; evolution tests are one dimensional. The ICERM event is
a summer school and developers' meeting, not a refereed paper proceedings
venue. This manuscript is an unpublished technical study.

## Files

* `src/coarsen.py`: transfer operators and diagnostic
* `tests/test_coarsen.py`: independent verification and counterexamples
* `run_experiments.py`: all experiments and figure generation
* `results/`: raw CSV and run summary
* `figures/`: vector figures
* `paper/main.tex`: manuscript source

License: MIT for code. Manuscript copyright 2026 Shivam Gupta.
