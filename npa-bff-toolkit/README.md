# NPA–BFF Toolkit

A research-oriented Python implementation of the **Navascués–Pironio–Acín (NPA) hierarchy** for finite-party Bell scenarios and **Brown–Fawzi–Fawzi (BFF) iterated-mean conditional-entropy relaxations**.

> **Status:** v0.1.0 research prototype. Solver outputs are numerical SDP estimates, not automatically rigorous quantum/cryptographic certificates. Please verify feasibility, dual bounds and numerical accuracy before citing quantitative results.

## Features

| Module | Capability |
| --- | --- |
| `qresource.NPA` | Bell-function optimization, arbitrary finite parties/settings/outcomes, full and selected intermediate hierarchy levels, probability and correlator syntax, local deterministic bound |
| `qresource.BFFNPA` | Conditional-entropy bounds for classical outcome registers against Eve; finite BFF orders, Gram and polynomial localizing matrices, Bell-statistics constraints, numerical diagnostic fields |

BFF full hierarchy levels (`level=1,2,...`) are distinct from the selected NPA intermediate levels (`level='1+AB'`).

## Installation

Use Python 3.10+:

```bash
python -m pip install -e '.[sdp,test]'
python -m pytest -q
```

You need an SDP solver supported by CVXPY. SCS is usable for small tests; MOSEK may be helpful for larger relaxations if available and licensed.

## Bell NPA: CHSH

```python
from qresource import NPA

chsh = '<A0B0> + <A0B1> + <A1B0> - <A1B1>'
model = NPA(parties=2, inputs=2, outcomes=2, level='1+AB')
print('Local bound:', model.local_bound(chsh))
print('NPA estimate:', model.maximize(chsh, solver='SCS'))
```

## BFF: conditional entropy H(A|E)

```python
from qresource import BFFNPA

model = BFFNPA(outputs=[[2,2],[2,2]], key_settings={0:0},
               bff_order=1, level=2)
result = model.solve(statistics=[(chsh, '>=', 2.6)], solver='SCS')
print(result.entropy_bits, result.status, result.reliable)
```

In the BFF class, `key_settings={0:0}` selects Alice's outcome at input zero, while `key_settings={0:0, 1:0}` selects the joint Alice–Bob outcome register. The entropy order is `alpha_k = 1 + 1/(2**k-1)`.

## Examples and tests

- [`examples/chsh_npa.py`](examples/chsh_npa.py)
- [`examples/i3322_npa.py`](examples/i3322_npa.py)
- [`examples/chsh_bff.py`](examples/chsh_bff.py)
- [`notebooks/Quickstart.ipynb`](notebooks/Quickstart.ipynb)

Source files are under `src/qresource/`. Tests include word reduction, CHSH algebra, BFF physical-polynomial checks and randomized local-bound comparisons. CI runs the tests on pushes and pull requests.

## Scope and limitations

- Finite-input, finite-output Bell scenarios. No source-independent network NPA or CV-specific NPA.
- Real-symmetrized moment representation. No general complex-Hermitian extension.
- BFF: supports conditional entropy of selected **classical** measurement outcomes against Eve; not arbitrary quantum conditioning systems.
- An `optimal` status or `reliable=True` is a numerical diagnostic, **not a rigorous dual certificate**.
- Numerical benchmark coverage is limited. More tests, solver comparisons and dual-validation are planned.

## Main references

1. M. Navascués, S. Pironio and A. Acín, *Bounding the Set of Quantum Correlations*, Physical Review Letters **98**, 010401 (2007). DOI: [10.1103/PhysRevLett.98.010401](https://doi.org/10.1103/PhysRevLett.98.010401).
2. P. Brown, H. Fawzi and O. Fawzi, *Computing Conditional Entropies for Quantum Correlations*, Nature Communications **12**, 575 (2021). DOI: [10.1038/s41467-020-20018-1](https://doi.org/10.1038/s41467-020-20018-1).

## License

No license is supplied yet. Choose a license before publishing the repository for unrestricted reuse, or keep it private while validating the code.
