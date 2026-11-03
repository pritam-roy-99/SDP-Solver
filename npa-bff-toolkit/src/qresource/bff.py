"""General finite-level Brown--Fawzi--Fawzi iterated-mean entropy SDP (fast version).

Same real-symmetrized noncommutative moment relaxation as the reference
implementation (same word basis, same localizing polynomials, same
moment identification w ~ w^dagger); only the implementation is optimized:

  * integer tokens; a word is (phys, eve) with phys = party-sorted tuple of
    projector ids and eve = tuple of Eve-operator ids. Because Eve commutes
    with every physical projector, (P_u E_u)(P_v E_v) = (P_u P_v)(E_u E_v), so
    only the physical part ever needs reduction.
  * memoized physical product / adjoint / moment key (physical products are
    shared by the Gram matrix and *all* localizing matrices).
  * adj(u)*q hoisted out of the inner j-loop of every localizing matrix.
  * W_{L-1} is a prefix of W_L (words sorted by length): one BFS, no repeat.
  * words generated level by level (no set differences against all words).
  * sparse F assembled from numpy arrays (vectorized symmetrization).

This is NOT a finite-dimensional / sequential-instrument hierarchy.
Numerical solver diagnostics are not rigorously validated dual certificates.
A small primal/dual gap alone is NOT enough for cryptographic certification.

Dependencies: numpy, scipy, cvxpy.
Reference: Brown, H. Fawzi, O. Fawzi, Nature Communications 12, 575 (2021).
"""
from collections import defaultdict
from itertools import product
from dataclasses import dataclass
import math
import re

import numpy as np
from scipy import sparse

_MISSING = object()


@dataclass
class BFFResult:
    """Numerical estimate, not a cryptographically rigorous certificate."""
    entropy_bits: float | None
    raw_entropy_bits: float | None
    q_upper_estimate: float | None     # approximate; NOT certified upper bound
    alpha: float
    bff_order: int
    npa_level: int
    matrix_size: int
    localization_size: int
    n_moments: int
    status: str
    primal_q: float | None = None
    dual_q: float | None = None       # SCS diagnostic, not validated dual bound
    rel_gap: float | None = None
    min_eig: float | None = None
    reliable: bool = False           # numerical consistency check only
    max_stat_violation: float | None = None
    scaled_stat_violation: float | None = None
    solver_primal_residual: float | None = None
    solver_dual_residual: float | None = None
    solver_gap: float | None = None
    solver_name: str | None = None

    def __repr__(self):
        def fmt(x):
            return "None" if x is None else f"{x:.8g}"
        return (f"BFFResult(H={fmt(self.entropy_bits)}, raw_H={fmt(self.raw_entropy_bits)}, "
                f"Q={fmt(self.q_upper_estimate)}, status={self.status!r}, "
                f"reliable={self.reliable}, Gram={self.matrix_size}, "
                f"localizing={self.localization_size}, moments={self.n_moments}, "
                f"max_stat_violation={fmt(self.max_stat_violation)}, "
                f"min_eig={fmt(self.min_eig)}, gap={fmt(self.rel_gap)})")


# Bell objective: [(coefficient, ((party, setting, outcome), ...)), ...]
# An empty event () denotes constant 1.
def binary_correlator(settings, coefficient=1.0):
    """Expand a binary correlator as a signed sum of probabilities."""
    return [(float(coefficient) * (-1)**sum(a),
             tuple((p, int(x), int(out)) for p, (x, out) in enumerate(zip(settings, a))))
            for a in product((0, 1), repeat=len(settings))]


def parse_bell(expr, outputs):
    """Optional compact parser: <A0B1>, p(01|10), pAC(01|10), constants.

    Examples: '<A0B0> + <A0B1> + <A1B0> - <A1B1>'.
    Comma-separated indices required if any index >=10.
    """
    if not isinstance(expr, str):
        return list(expr)
    term = re.compile(r'([+-]?)(?:(\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)?\*?(<[a-z0-9*]+>|p[a-z]*\([0-9,]+\|[0-9,]+\))?', re.I)
    s = re.sub(r'\s+', '', expr.lower())
    s = re.sub(r'[+-]{2,}', lambda m: '-' if m.group().count('-') % 2 else '+', s)
    out = []
    pos = 0
    def split_indices(t):
        return [int(i) for i in (t.split(',') if ',' in t else list(t))]
    while pos < len(s):
        m = term.match(s, pos)
        if not m or m.end() == pos or (m.group(2) is None and m.group(3) is None):
            raise ValueError(f'Cannot parse Bell expression at {s[pos:]}')
        if pos > 0 and not m.group(1):
            raise ValueError(f'Missing + or - near {s[pos:]}')
        raw = s[pos:m.end()]
        a = m.group(3)
        prefix = raw[:len(raw)-len(a)] if a else raw
        prefix = prefix.strip('+-').rstrip('*')
        c = float(prefix) if prefix else 1.0
        if m.group(1) == '-': c = -c
        if a is None:
            out.append((c, ()))
        elif a.startswith('<'):
            body = a[1:-1].replace('*','')
            pairs = re.findall(r'([a-z])(\d+)', body)
            if ''.join(p+x for p,x in pairs) != body:
                raise ValueError(f'Bad correlator {a}')
            specs = [(ord(p)-97, int(x)) for p,x in pairs]
            for p,x in specs:
                if not (0<=p<len(outputs) and 0<=x<len(outputs[p]) and outputs[p][x]==2):
                    raise ValueError(f'Correlators require valid binary setting, got {a}')
            if len({p for p,x in specs}) != len(specs):
                raise ValueError('Repeated party')
            for outcomes in product((0,1),repeat=len(specs)):
                event = tuple((p,x,v) for (p,x),v in zip(specs,outcomes))
                out.append((c*(-1)**sum(outcomes),event))
        else:
            match = re.fullmatch(r'p([a-z]*)\(([0-9,]+)\|([0-9,]+)\)',a)
            if match is None: raise ValueError(f'Bad probability {a}')
            parties = [ord(p)-97 for p in match.group(1)] if match.group(1) else list(range(len(outputs)))
            outs = split_indices(match.group(2))
            settings = split_indices(match.group(3))
            if len(parties)!=len(outs) or len(parties)!=len(settings):
                raise ValueError(f'Unmatched party/setting/outcome indices in {a}')
            out.append((c,tuple(zip(parties,settings,outs))))
        pos=m.end()
    return out


class BFFNPA:
    """Compute finite-NPA BFF conditional entropy bound for Z|E.

    outputs: [[d_A0, d_A1, ...], [d_B0, ...], ...]
    key_settings: {party: setting}, e.g. {0:0} for H(A|E),
                  {0:0,1:0} for H(AB|E) at fixed settings.
    bff_order: k (alpha_k=1+1/(2**k-1))
    level: full operator-word NPA level L, must be >=1.
    """
    ID = ((), ())

    def __init__(self, outputs, key_settings, bff_order=1, level=2, max_words=1600):
        self.outputs = [list(map(int, row)) for row in outputs]
        if not self.outputs or any(not row or any(d < 2 for d in row) for row in self.outputs):
            raise ValueError('Each party must have >=1 measurement and >=2 outcomes each')
        self.P = len(self.outputs)
        self.key_settings = dict(sorted(key_settings.items()))
        if not self.key_settings:
            raise ValueError('key_settings must contain at least one key party')
        for p, x in self.key_settings.items():
            if not (isinstance(p, int) and 0 <= p < self.P and isinstance(x, int) and 0 <= x < len(self.outputs[p])):
                raise ValueError(f'Invalid key party/setting {(p, x)}')
        if not isinstance(bff_order, int) or bff_order < 1 or not isinstance(level, int) or level < 1:
            raise ValueError('bff_order and level must be positive integers')
        self.k, self.L, self.max_words = bff_order, level, max_words
        self.alpha = 1 + 1/(2**self.k - 1)
        self.key_events = [tuple((p, x, a) for (p, x), a in zip(self.key_settings.items(), outcomes))
                           for outcomes in product(*(range(self.outputs[p][x])
                                                     for p, x in self.key_settings.items()))]
        self.nz = len(self.key_events)

        # Integer tokens. Physical projectors first (last outcome omitted),
        # then Eve operators V_{i,z} / V_{i,z}^dagger in adjacent pairs.
        self.proj, self.owner, self.meas = {}, [], []
        sid = 0
        for p, row in enumerate(self.outputs):
            for x, d in enumerate(row):
                for a in range(d - 1):
                    self.proj[p, x, a] = len(self.owner)
                    self.owner.append(p)
                    self.meas.append(sid)
                sid += 1
        self.nphys = len(self.owner)
        self.ntok = self.nphys + 2*self.k*self.nz
        self.adj_tok = list(range(self.ntok))
        for t in range(self.nphys, self.ntok, 2):
            self.adj_tok[t], self.adj_tok[t+1] = t+1, t

        self._pmul_cache, self._padj_cache, self._key_cache = {}, {}, {}
        self._levels = [[self.ID]]  # _levels[d] = sorted reduced words of length exactly d

    # ---- word algebra ---------------------------------------------------
    # word = (phys, eve). Eve commutes with all physical projectors; Eve
    # operators obey no relations among themselves.
    def eve_token(self, i, z, star=0):
        return self.nphys + 2*((i-1)*self.nz + z) + star

    def _reduce_phys(self, seq):
        owner, meas = self.owner, self.meas
        blocks = {}
        for g in seq:
            b = blocks.get(owner[g])
            if b is None:
                blocks[owner[g]] = [g]
            elif meas[b[-1]] == meas[g]:
                if b[-1] != g:
                    return None          # orthogonal projectors
            else:
                b.append(g)
        return tuple(g for p in sorted(blocks) for g in blocks[p])

    def _pmul(self, a, b):
        if not a: return b
        if not b: return a
        k = (a, b)
        r = self._pmul_cache.get(k, _MISSING)
        if r is _MISSING:
            r = self._pmul_cache[k] = self._reduce_phys(a + b)
        return r

    def _padj(self, a):
        if len(a) < 2: return a
        r = self._padj_cache.get(a)
        if r is None:
            r = self._padj_cache[a] = self._reduce_phys(a[::-1])
        return r

    def mul(self, u, v):
        if u is None or v is None: return None
        p = self._pmul(u[0], v[0])
        return None if p is None else (p, u[1] + v[1])

    def adjoint(self, w):
        if w is None: return None
        at = self.adj_tok
        return (self._padj(w[0]), tuple(at[t] for t in reversed(w[1])))

    def key(self, w):
        r = self._key_cache.get(w)
        if r is None:
            r = self._key_cache[w] = min(w, self.adjoint(w))
        return r

    def basis(self, degree):
        """All reduced words of length <= degree, sorted by length.
        W_d is a prefix of W_{d+1}, so one BFS serves every degree."""
        lv, nphys = self._levels, self.nphys
        while len(lv) <= degree:
            new = set()
            for w in lv[-1]:
                for g in range(nphys):
                    r = self.mul(w, ((g,), ()))
                    if r is not None:
                        new.add(r)
                for t in range(nphys, self.ntok):
                    new.add((w[0], w[1] + (t,)))
            d = len(lv)
            lv.append(sorted(w for w in new if len(w[0]) + len(w[1]) == d))
            total = sum(map(len, lv))
            if total > self.max_words:
                raise MemoryError(f'{total} words already at degree <= {d}; '
                                  'choose a smaller level or raise max_words cautiously')
        return [w for lvl in lv[:degree + 1] for w in lvl]

    # ---- polynomials = {word: real coefficient} --------------------------
    def poly_add(self, *polys):
        ret = defaultdict(float)
        for poly in polys:
            for w, c in poly.items():
                ret[w] += c
        return {w: c for w, c in ret.items() if abs(c) > 1e-13}

    def poly_mul(self, A, B):
        ret = defaultdict(float)
        for u, c in A.items():
            for v, d in B.items():
                w = self.mul(u, v)
                if w is not None: ret[w] += c*d
        return {w: c for w, c in ret.items() if abs(c) > 1e-13}

    def dagger_poly(self, A):
        return {self.adjoint(w): c for w, c in A.items()}

    def scale(self, A, c):
        return {w: c*v for w, v in A.items() if abs(c*v) > 1e-13}

    def prob_poly(self, event):
        if len({p for p, x, a in event}) != len(event):
            raise ValueError('Only one measurement for each party in an event')
        result = {self.ID: 1.0}
        for p, x, a in event:
            if not (0 <= p < self.P and 0 <= x < len(self.outputs[p]) and 0 <= a < self.outputs[p][x]):
                raise ValueError(f'Invalid event {(p, x, a)}')
            if a < self.outputs[p][x] - 1:
                factor = {((self.proj[p, x, a],), ()): 1.0}
            else:
                factor = {self.ID: 1.0}
                for b in range(self.outputs[p][x] - 1):
                    factor[((self.proj[p, x, b],), ())] = -1.0
            result = self.poly_mul(result, factor)
        return result

    def bell_poly(self, expr):
        terms = parse_bell(expr, self.outputs)
        return self.poly_add(*(self.scale(self.prob_poly(ev), c) for c, ev in terms))

    def vpoly(self, i, z, star=0):
        return {((), (self.eve_token(i, z, star),)): 1.0}

    def bff_polynomials(self):
        I = {self.ID: 1.0}
        Q = {}
        G = I
        positivity, chains = [], []
        for z, event in enumerate(self.key_events):
            M = self.prob_poly(event)
            V, Vdag = self.vpoly(1, z), self.vpoly(1, z, 1)
            Q = self.poly_add(Q, self.scale(self.poly_mul(M, self.poly_add(V, Vdag)), 0.5))
            positivity.append(self.poly_add(V, Vdag))
            last = self.vpoly(self.k, z)
            G = self.poly_add(G, self.scale(self.poly_mul(self.dagger_poly(last), last), -1))
            for i in range(1, self.k):
                Vi, Vi1 = self.vpoly(i, z), self.vpoly(i + 1, z)
                chains.append(self.poly_add(Vi1, self.dagger_poly(Vi1),
                                            self.scale(self.poly_mul(self.dagger_poly(Vi), Vi), -2)))
        return Q, [G] + positivity + chains

    # ---- moment matrices ---------------------------------------------------
    def _matrix(self, poly, basis, ids):
        """Entries of M_ij = <b_i^dagger q b_j> (real part), i <= j, as COO data."""
        n = len(basis)
        mul, key, ID = self.mul, self.key, self.ID
        adjs = [self.adjoint(u) for u in basis]
        terms = list(poly.items())
        constant = np.zeros(n*n)
        I, J, M, C = [], [], [], []
        for i in range(n):
            lefts = [(c, l) for q, c in terms if (l := mul(adjs[i], q)) is not None]  # hoisted
            for j in range(i, n):
                bj = basis[j]
                entry = {}
                for c, l in lefts:
                    w = mul(l, bj)
                    if w is None: continue
                    k = key(w)
                    entry[k] = entry.get(k, 0.) + c
                for w, c in entry.items():
                    if abs(c) < 1e-12: continue
                    if w == ID:
                        constant[i*n + j] += c
                        if i != j: constant[j*n + i] += c
                    else:
                        idx = ids.get(w)
                        if idx is None:
                            idx = ids[w] = len(ids)
                        I.append(i); J.append(j); M.append(idx - 1); C.append(c)
        return n, constant, (np.array(I, dtype=np.int64), np.array(J, dtype=np.int64),
                             np.array(M, dtype=np.int64), np.array(C))

    @staticmethod
    def _assemble(n, m, coo):
        I, J, M, C = coo
        off = I != J                       # symmetrize: (i,j) and (j,i)
        rows = np.concatenate([I*n + J, (J*n + I)[off]])
        cols = np.concatenate([M, M[off]])
        data = np.concatenate([C, C[off]])
        return sparse.csr_matrix((data, (rows, cols)), shape=(n*n, m))

    @staticmethod
    def _min_eigenvalue(matrix, dense_threshold=250):
        """Smallest eigenvalue of a numerically symmetrized PSD block.

        Use sparse Lanczos for larger matrices; fall back to dense eigvalsh
        if ARPACK does not converge. For security proofs, use validated bounds.
        """
        matrix = (matrix + matrix.T) / 2.0
        if matrix.shape[0] <= dense_threshold:
            return float(np.linalg.eigvalsh(matrix)[0])
        from scipy.sparse.linalg import eigsh
        try:
            return float(eigsh(sparse.csr_matrix(matrix), k=1, which="SA",
                               return_eigenvectors=False, tol=1e-7)[0])
        except Exception:
            return float(np.linalg.eigvalsh(matrix)[0])

    @staticmethod
    def _scs_diagnostics(problem, solver_name):
        """Read SCS solver residuals when available; other solvers return None.

        Note: `dobj` is an *approximate* dual objective, not a validated bound.
        CVXPY uses a minimization of -Q internally for this maximization.
        """
        if str(solver_name).upper() != "SCS":
            return None, None, None, None
        info = getattr(problem.solver_stats, "extra_stats", None)
        if not isinstance(info, dict):
            return None, None, None, None
        info = info.get("info", info)
        if not isinstance(info, dict):
            return None, None, None, None

        def finite(field):
            value = info.get(field)
            try:
                number = float(value)
                return number if np.isfinite(number) else None
            except (TypeError, ValueError):
                return None

        obj = finite("dobj")
        # SCS's `dobj` is for CVXPY's minimization of -Q.
        dual_q = -obj if obj is not None else None
        return dual_q, finite("res_pri"), finite("res_dual"), finite("gap")

    def solve(self, statistics=(), solver=None, eps=1e-6, verbose=False,
              allow_inaccurate=False, gap_tol=1e-4, psd_tol=1e-5,
              stat_tol=1e-5, solver_residual_tol=1e-4):
        """Numerical BFF conditional-entropy SDP.

        statistics: [(bell_expression, relation, value), ...]
        relation: '>=', '<=', or '=='; e.g. CHSH >= 2.6.

        `entropy_bits` is shown only when numerical checks pass, unless
        allow_inaccurate=True. `raw_entropy_bits` is always a *numerical*
        conversion when Q>0, even if unreliable. Neither is a rigorous
        certificate without validated dual feasibility/error bounds.

        `reliable` means: solver reported optimal, primal Bell constraints and
        PSD matrices pass the requested tolerances, and available solver
        residual/gap checks pass. A missing dual gap from MOSEK does not cause
        an automatic failure. This is NOT rigorous dual certification.
        """
        import cvxpy as cp

        if eps <= 0 or min(gap_tol, psd_tol, stat_tol, solver_residual_tol) < 0:
            raise ValueError("Invalid tolerance values")

        W = self.basis(self.L)
        Wloc = self.basis(self.L - 1)
        objective_poly, loc_polys = self.bff_polynomials()
        stat_polys = []
        for expression, relation, target in statistics:
            if relation not in ("==", ">=", "<="):
                raise ValueError("Relation must be one of ==, >=, <=")
            stat_polys.append((self.bell_poly(expression), relation, float(target)))

        ids = {self.ID: 0}  # identity moment is fixed at one
        matrices = [self._matrix({self.ID: 1.0}, W, ids)]
        matrices += [self._matrix(poly, Wloc, ids) for poly in loc_polys]
        for poly in [objective_poly] + [poly for poly, _, _ in stat_polys]:
            for word, coefficient in poly.items():
                if abs(coefficient) > 1e-12 and self.key(word) not in ids:
                    raise ValueError(
                        f"Moment {word} missing at NPA level {self.L}; increase level."
                    )

        m = len(ids) - 1
        z = cp.Variable(m)
        constraints = []
        psd_blocks = []
        for n, const, coo in matrices:
            F = self._assemble(n, m, coo)
            psd_blocks.append((n, const, F))
            G = cp.reshape(F @ z + const, (n, n), order="C")
            constraints.append(G >> 0)

        def affine(poly):
            coeff = np.zeros(m)
            offset = 0.0
            for word, factor in poly.items():
                k = self.key(word)
                if k == self.ID:
                    offset += factor
                else:
                    coeff[ids[k] - 1] += factor
            return coeff @ z + offset

        stat_exprs = []
        for poly, relation, target in stat_polys:
            expression = affine(poly)
            stat_exprs.append((expression, relation, target))
            if relation == ">=":
                constraints.append(expression >= target)
            elif relation == "<=":
                constraints.append(expression <= target)
            else:
                constraints.append(expression == target)

        objective = affine(objective_poly)
        problem = cp.Problem(cp.Maximize(objective), constraints)
        if solver is None:
            solver = "MOSEK" if "MOSEK" in cp.installed_solvers() else "SCS"
        solver_name = str(solver).upper()
        options = {"verbose": verbose}
        if solver_name == "SCS":
            options.update({"eps": eps, "max_iters": 200000})
        value = problem.solve(solver=solver, **options)

        result_kwargs = dict(alpha=self.alpha, bff_order=self.k,
                             npa_level=self.L, matrix_size=len(W),
                             localization_size=len(Wloc), n_moments=m,
                             status=problem.status, solver_name=solver_name)
        if value is None or z.value is None or not np.isfinite(value):
            return BFFResult(None, None, None, **result_kwargs)

        point = np.asarray(z.value).reshape(-1)
        min_eig = min(
            self._min_eigenvalue((F @ point + const).reshape(n, n, order="C"))
            for n, const, F in psd_blocks
        )
        stat_violations = []
        for expression, relation, target in stat_exprs:
            observed = float(expression.value)
            if relation == ">=":
                violation = max(0.0, target - observed)
            elif relation == "<=":
                violation = max(0.0, observed - target)
            else:
                violation = abs(observed - target)
            stat_violations.append((violation, violation / (1 + abs(target))))
        max_stat = max((v for v, _ in stat_violations), default=0.0)
        scaled_stat = max((v for _, v in stat_violations), default=0.0)

        dual_q, res_pri, res_dual, solver_gap = self._scs_diagnostics(
            problem, solver_name
        )
        primal_q = float(value)
        # A heuristic more cautious than reporting primal alone, NOT a proven
        # upper bound: even the numerical dual may be slightly infeasible.
        q_est = max(primal_q, dual_q) if dual_q is not None else primal_q
        rel_gap = (abs(primal_q - dual_q) / (1 + abs(primal_q) + abs(dual_q))
                   if dual_q is not None else None)
        gap_ok = rel_gap is None or rel_gap <= gap_tol
        residuals_ok = all(r is None or abs(r) <= solver_residual_tol
                           for r in (res_pri, res_dual))
        # `reliable` is a *numerical* flag, including for solvers without
        # an exported dual objective. Never mistake it for proof of security.
        reliable = (problem.status == "optimal"
                    and min_eig >= -psd_tol
                    and scaled_stat <= stat_tol
                    and gap_ok and residuals_ok)

        raw_entropy = (-2**self.k * math.log2(q_est) if q_est > 0 else None)
        display_entropy = (max(0.0, raw_entropy)
                           if raw_entropy is not None
                           and (reliable or allow_inaccurate) else None)

        return BFFResult(
            display_entropy, raw_entropy, q_est, **result_kwargs,
            primal_q=primal_q, dual_q=dual_q, rel_gap=rel_gap,
            min_eig=min_eig, reliable=reliable,
            max_stat_violation=max_stat, scaled_stat_violation=scaled_stat,
            solver_primal_residual=res_pri, solver_dual_residual=res_dual,
            solver_gap=solver_gap,
        )
