from qresource import BFFNPA, parse_bell

CHSH = '<A0B0>+<A0B1>+<A1B0>-<A1B1>'


def test_bff_chsh_k1_basis_and_moments():
    bff = BFFNPA(outputs=[[2,2],[2,2]], key_settings={0:0}, bff_order=1, level=2)
    assert len(bff.basis(2)) == 49
    assert bff.alpha == 2
    ids = {bff.ID: 0}
    W = bff.basis(2)
    WL = bff.basis(1)
    Q, locs = bff.bff_polynomials()
    mats = [bff._matrix({bff.ID:1.0}, W, ids)] + [bff._matrix(q, WL, ids) for q in locs]
    assert len(locs) == 3
    assert len(mats) == 4
    assert all(n == 9 for n,*_ in mats[1:])
    assert len(ids) - 1 == 498
    assert all(bff.key(w) in ids for w in Q)


def test_bff_k2_and_parser():
    bff = BFFNPA(outputs=[[2,2],[2,2]], key_settings={0:0}, bff_order=2, level=2)
    assert len(bff.basis(2)) == 117
    _, locs = bff.bff_polynomials()
    assert len(locs) == 5  # global normalization + 2 positivity + 2 chain
    bell = parse_bell(CHSH, bff.outputs)
    assert len(bell) == 16


def test_bff_adjoint_order():
    bff = BFFNPA(outputs=[[2,2],[2,2]], key_settings={0:0})
    v = ((), (bff.eve_token(1,0), bff.eve_token(1,1,1)))
    w = bff.adjoint(v)
    assert w == ((), (bff.eve_token(1,1), bff.eve_token(1,0,1)))
    assert bff.adjoint(w) == v


def test_bff_physical_bell_state():
    """Check Q1 and CHSH polynomials against an explicit valid quantum strategy."""
    import numpy as np
    bff = BFFNPA(outputs=[[2,2],[2,2]], key_settings={0:0}, bff_order=1, level=2)
    I = np.eye(2)
    X = np.array([[0,1],[1,0]])
    Z = np.diag([1.,-1.])
    obs = {(0,0):Z, (0,1):X,
           (1,0):(Z+X)/np.sqrt(2), (1,1):(Z-X)/np.sqrt(2)}
    projs = {}
    for (p,x,a), token in bff.proj.items():
        projs[token] = (np.eye(4) + np.kron(obs[p,x] if p == 0 else I,
                                                 obs[p,x] if p == 1 else I))/2
    phi = np.array([1.,0.,0.,1.])/np.sqrt(2)
    def evaluate(poly):
        total = 0.0
        for (phys,eve), coeff in poly.items():
            op = np.eye(4)
            for t in phys:
                op = op @ projs[t]
            total += coeff * float(np.real(np.vdot(phi,op@phi))) * (1/np.sqrt(2))**len(eve)
        return total
    assert np.isclose(evaluate(bff.bell_poly(CHSH)), 2*np.sqrt(2))
    Q, positivity = bff.bff_polynomials()
    assert np.isclose(evaluate(Q), 1/np.sqrt(2))
    assert np.isclose(evaluate(positivity[0]), 0.)
