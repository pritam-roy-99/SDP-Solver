import numpy as np
from qresource import NPA

CHSH = '<A0B0> + <A0B1> + <A1B0> - <A1B1>'


def test_classical_chsh_and_words():
    npa = NPA(parties=2, inputs=2, outcomes=2, level='1+AB')
    assert npa.local_bound(CHSH) == 2
    assert len(npa.word_basis()) == 9
    npa.build()
    assert npa.F.shape[0] == 81
    assert npa.n_moments > 0


def test_probability_and_correlator_parser():
    npa = NPA(parties=2, inputs=3, outcomes=2, level=2)
    assert npa.parse_atom('p(00|01)')[0] == 'probability'
    assert npa.parse_atom('<A0B1>')[0] == 'correlator'
    terms, relation = npa.parse_constraint('<A0> == 0')
    assert relation == '=='
    assert len(terms) == 2


def test_projector_idempotence_orthogonality():
    npa = NPA(parties=1, inputs=2, outcomes=3, level=2)
    I = npa.identity
    a = npa.projector[0, 0, 0]
    b = npa.projector[0, 0, 1]
    w = npa.append(I, a)
    assert npa.append(w, a) == w
    assert npa.append(w, b) is None
