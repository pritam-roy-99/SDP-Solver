"""Independent deterministic checks with seeded random Bell correlators."""
import itertools
import numpy as np
from qresource import NPA

def brute_force(C):
    n, m = C.shape
    return max(
        sum(C[i, j]*a[i]*b[j] for i in range(n) for j in range(m))
        for a in itertools.product((-1, 1), repeat=n)
        for b in itertools.product((-1, 1), repeat=m)
    )

def test_random_binary_correlation_local_bound():
    rng = np.random.default_rng(20261103)
    for n, m in [(2,2), (3,3), (2,3)]:
        C = rng.integers(-3, 4, size=(n, m))
        terms = [f'{int(C[i,j]):+d}*<A{i}B{j}>'
                 for i in range(n) for j in range(m) if C[i,j]]
        expr = ''.join(terms) or '0'
        model = NPA(parties=2, inputs=[n,m], outcomes=2, level=1)
        assert np.isclose(model.local_bound(expr), brute_force(C))
