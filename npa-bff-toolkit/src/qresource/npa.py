"""Readable, sparse NPA hierarchy for finite-party projective Bell experiments.

Example (also works if this file is pasted into a notebook cell):
    npa = NPA(parties=2, inputs=2, outcomes=2, level="1+AB")
    npa.maximize("<A0B0> + <A0B1> + <A1B0> - <A1B1>")

Basic notation:
  <A0B1>        binary correlator, outputs mapped 0->+1, 1->-1
  p(00|01)      full joint probability, parties in alphabetical order
  pAC(01|10)    marginal for Alice and Charlie
  0.5*pA(0|1)   coefficients and constants are supported
  constraints=["<A0> == 0", "pA(0|0) <= 0.8"]

The real-symmetrized NPA matrix is a necessary quantum condition.
No finite-dimensional Hilbert-space restriction is imposed.
"""

import re
import time
from itertools import product
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse

# A word is a tuple of per-party blocks of projector IDs.
# Example with three parties: ((0, 1), (5,), ())
# means two Alice projectors, one Bob projector, and identity on Charlie.

# A simple term scanner. It accepts, e.g., '-.5*<A0B1>', '+2pA(0|0)', '+1'.
TERM = re.compile(
    r"([+-]?)"                         # sign
    r"((?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)?"  # number
    r"\*?"                            # optional multiplication sign
    r"(<[a-z0-9*]+>|p[a-z]*\([0-9,]+\|[0-9,]+\))?"  # observable/probability
)
RELATION = re.compile(r"<=|>=|==|=")


@dataclass
class Result:
    value: float | None
    status: str
    matrix_size: int
    n_moments: int
    seconds: float

    def __repr__(self):
        value = "None" if self.value is None else f"{float(self.value):.8f}"
        return (f"Result(value={value}, status={self.status!r}, "
                f"matrix={self.matrix_size}x{self.matrix_size}, "
                f"moments={self.n_moments}, time={self.seconds:.2f}s)")


class NPA:
    """Projector-based, real-symmetrized NPA with cached sparse moments."""

    def __init__(self, parties=2, inputs=2, outcomes=2, level=2):
        # Fully specify the outcomes per input and per party.
        if isinstance(outcomes, int):
            sizes = [inputs] * parties if isinstance(inputs, int) else list(inputs)
            outputs = [[outcomes] * count for count in sizes]
        else:
            outputs = []
            for p, spec in enumerate(outcomes):
                if isinstance(spec, int):
                    count = inputs if isinstance(inputs, int) else inputs[p]
                    outputs.append([spec] * count)
                else:
                    outputs.append(list(spec))

        if (not outputs or len(outputs) > 26 or
            any(not row or any(not isinstance(d, int) or d < 2 for d in row)
                for row in outputs)):
            raise ValueError("Specify 1..26 parties and >=2 outcomes per setting")

        self.outputs = outputs
        self.parties = len(outputs)
        self.level = level
        self.identity = ((),) * self.parties

        # Omit the last outcome projector of each measurement.
        # projector[(p,x,a)] = integer label g
        self.projector = {}
        self.owner = []           # owner[g] = party index
        self.measurement = []     # measurement[g] = unique setting ID
        self.party_generators = [[] for _ in outputs]
        setting_id = 0

        for p, row in enumerate(outputs):
            for x, number_of_outcomes in enumerate(row):
                for a in range(number_of_outcomes - 1):
                    g = len(self.owner)
                    self.projector[p, x, a] = g
                    self.owner.append(p)
                    self.measurement.append(setting_id)
                    self.party_generators[p].append(g)
                setting_id += 1

        self._ready = False  # Build F, c and moment IDs only once.

    # ----------------------- projector-word algebra -----------------------
    def append(self, word, g):
        """Multiply a reduced word by projector g; None denotes zero."""
        p = self.owner[g]
        block = word[p]
        if block and self.measurement[block[-1]] == self.measurement[g]:
            return word if block[-1] == g else None  # P_a P_b = delta_ab P_a
        return word[:p] + (block + (g,),) + word[p+1:]

    @staticmethod
    def dagger(word):
        """Reverse each party's block: (UV)^dagger = V^dagger U^dagger."""
        return tuple(block[::-1] for block in word)

    def multiply(self, left, right):
        """Reduce left * right, using only the boundary in each party block."""
        answer = []
        for a, b in zip(left, right):
            if a and b and self.measurement[a[-1]] == self.measurement[b[0]]:
                if a[-1] != b[0]:
                    return None              # orthogonal projectors
                answer.append(a + b[1:])      # identical: P^2 = P
            else:
                answer.append(a + b)
        return tuple(answer)

    def moment_key(self, word):
        """Identify w with w^dagger for the real part of <w>."""
        return min(word, self.dagger(word))

    # --------------------- generate NPA word basis ------------------------
    def word_basis(self):
        """Full level n, or intermediate levels like '1+AB+AC+BC'."""
        match = re.fullmatch(r"(\d+)((?:\+[a-z]+)*)", str(self.level).lower().replace(" ", ""))
        if not match or int(match.group(1)) < 1:
            raise ValueError("Use level=2 or level='1+AB', etc.")

        n = int(match.group(1))
        words, frontier = {self.identity}, {self.identity}

        for _ in range(n):
            new = {
                self.append(w, g)
                for w in frontier
                for g in range(len(self.owner))
            }
            frontier = (new - {None}) - words
            words |= frontier

        # Additional party-cross terms, e.g. all A_x B_y for '1+AB'.
        for party_string in match.group(2).split("+")[1:]:
            party_ids = [ord(ch) - ord("a") for ch in party_string]
            if (len(set(party_ids)) != len(party_ids) or
                any(p >= self.parties for p in party_ids)):
                raise ValueError(f"Invalid intermediate level: +{party_string}")
            for gs in product(*(self.party_generators[p] for p in party_ids)):
                w = self.identity
                for g in gs:
                    w = self.append(w, g)
                words.add(w)

        return sorted(words, key=lambda w: (sum(map(len, w)), w))

    # --------------------- sparse moment matrix ---------------------------
    def build(self):
        """Cache vec(Gamma) = F @ z + c, without creating CVXPY expressions."""
        if self._ready:
            return

        self.words = self.word_basis()
        self.matrix_size = len(self.words)
        n = self.matrix_size
        self.moment_ids = {self.identity: 0}  # identity -> constant 1
        rows, columns = [], []               # entries of sparse F
        constant = np.zeros(n * n)

        for i, u in enumerate(self.words):
            u_dagger = self.dagger(u)
            for j in range(i, n):
                w = self.multiply(u_dagger, self.words[j])
                if w is None:
                    continue                  # matrix entry is zero
                key = self.moment_key(w)
                if key not in self.moment_ids:
                    self.moment_ids[key] = len(self.moment_ids)

                index = self.moment_ids[key]
                positions = [i*n+j] if i == j else [i*n+j, j*n+i]
                if index == 0:
                    constant[positions] = 1   # matrix entry is identity
                else:
                    rows.extend(positions)
                    columns.extend([index-1] * len(positions))

        self.n_moments = len(self.moment_ids) - 1
        self.F = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(n*n, self.n_moments)
        )
        self.c = constant
        self._ready = True

    # ----------------------- parse mathematical expressions ---------------
    @staticmethod
    def _indices(s):
        """'012' -> [0,1,2]; '0,10,2' -> [0,10,2]."""
        return [int(t) for t in (s.split(",") if "," in s else s)]

    def parse_atom(self, text):
        """Parse <A0B1>, p(01|10) or pAC(01|10)."""
        text = text.lower()
        if text.startswith("<"):
            body = text[1:-1].replace("*", "")
            pairs = re.findall(r"([a-z])(\d+)", body)
            if "".join(p+x for p, x in pairs) != body:
                raise ValueError(f"Invalid correlator: {text}")
            items = [(ord(p)-97, int(x), None) for p, x in pairs]
            kind = "correlator"
        else:
            match = re.fullmatch(r"p([a-z]*)\(([0-9,]+)\|([0-9,]+)\)", text)
            if match is None:
                raise ValueError(f"Invalid probability: {text}")
            parties = ([ord(p)-97 for p in match.group(1)]
                       if match.group(1) else list(range(self.parties)))
            outcomes = self._indices(match.group(2))
            settings = self._indices(match.group(3))
            if len(parties) != len(outcomes) or len(parties) != len(settings):
                raise ValueError(f"Mismatched indices in {text}")
            items = list(zip(parties, settings, outcomes))
            kind = "probability"

        if len({p for p, x, a in items}) != len(items):
            raise ValueError(f"Repeated party in {text}")
        for p, x, a in items:
            if not (0 <= p < self.parties and 0 <= x < len(self.outputs[p])):
                raise ValueError(f"Invalid party or setting in {text}")
            if kind == "correlator" and self.outputs[p][x] != 2:
                raise ValueError("Correlator notation is for binary outcomes only")
            if kind == "probability" and not 0 <= a < self.outputs[p][x]:
                raise ValueError(f"Invalid outcome in {text}")
        return kind, items

    def parse_sum(self, text):
        """Return [(coefficient, parsed_atom_or_None), ...] for a linear sum."""
        s = re.sub(r"\s+", "", text.lower())
        s = re.sub(r"[+-]{2,}", lambda m: "-" if m.group().count("-") % 2 else "+", s)
        terms, position = [], 0
        while position < len(s):
            m = TERM.match(s, position)
            if (m is None or m.end() == position or
                (m.group(2) is None and m.group(3) is None) or
                (position > 0 and not m.group(1))):
                raise ValueError(f"Invalid expression near {s[position:]} in {text!r}")
            sign = -1 if m.group(1) == "-" else 1
            coef = float(m.group(2)) if m.group(2) is not None else 1.0
            atom = self.parse_atom(m.group(3)) if m.group(3) else None
            terms.append((sign * coef, atom))
            position = m.end()
        if not terms:
            raise ValueError("Expression is empty")
        return terms

    def parse_constraint(self, text):
        """Split a relation without mistaking the closing '>' of <A0> for >=."""
        clean = re.sub(r"\s+", "", text.lower())
        # Keep character positions, masking correlator atoms only.
        masked = re.sub(r"<[a-z0-9*]+>",
                        lambda match: "#" * len(match.group()), clean)
        matches = list(RELATION.finditer(masked))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one <=, >=, or == in {text!r}")
        relation = matches[0]
        left = clean[:relation.start()]
        right = clean[relation.end():]
        terms = self.parse_sum(left) + [(-c, atom) for c, atom in self.parse_sum(right)]
        return terms, relation.group()

    # --------------------------- expression -> moments --------------------
    def local_factor(self, p, x, a):
        """One local event as {projector_id_or_None: coefficient}."""
        d = self.outputs[p][x]
        if a is None:                            # binary observable: 2P_0-I
            return {self.projector[p, x, 0]: 2.0, None: -1.0}
        if a < d-1:
            return {self.projector[p, x, a]: 1.0}
        return {None: 1.0, **{self.projector[p, x, b]: -1.0 for b in range(d-1)}}

    def polynomial(self, terms):
        """Expand a Bell expression into sums of reduced projector words."""
        answer = defaultdict(float)
        for coefficient, atom in terms:
            if atom is None:
                answer[self.identity] += coefficient
                continue
            kind, items = atom
            current = {self.identity: coefficient}
            for p, x, a in items:
                factor = self.local_factor(p, x, a)
                following = defaultdict(float)
                for w, c in current.items():
                    for g, factor_c in factor.items():
                        if g is None:
                            new_word = w
                        else:
                            new_word = self.append(w, g)
                        if new_word is not None:
                            following[new_word] += c * factor_c
                current = following
            for w, c in current.items():
                answer[w] += c
        return answer

    def affine(self, terms, z):
        """Convert parsed terms to a CVXPY affine expression."""
        coef = np.zeros(self.n_moments)
        offset = 0.0
        for w, c in self.polynomial(terms).items():
            if w == self.identity:
                offset += c
                continue
            moment_id = self.moment_ids.get(self.moment_key(w))
            if moment_id is None:
                raise ValueError("Moment missing at this level; increase level or add +AB etc.")
            coef[moment_id-1] += c
        return coef @ z + offset

    # ------------------------------ numerical optimization ----------------
    def _sdp_variables(self):
        """Cache the CVXPY matrix and PSD constraint, not just its sparse data."""
        import cvxpy as cp
        self.build()
        if not hasattr(self, "_z"):
            self._z = cp.Variable(self.n_moments)
            gamma = cp.reshape(self.F @ self._z + self.c,
                               (self.matrix_size, self.matrix_size), order="C")
            self._psd = gamma >> 0
        return self._z, self._psd

    def _optimize(self, expression, maximize, constraints, solver, opts):
        import cvxpy as cp
        start = time.perf_counter()
        z, psd = self._sdp_variables()
        objective = self.affine(self.parse_sum(expression), z)
        cons = [psd]
        constraints = [constraints] if isinstance(constraints, str) else constraints
        for text in constraints:
            terms, relation = self.parse_constraint(text)
            val = self.affine(terms, z)
            cons.append(val <= 0 if relation == "<=" else
                        val >= 0 if relation == ">=" else val == 0)
        task = cp.Problem(cp.Maximize(objective) if maximize else cp.Minimize(objective), cons)
        if solver is None:
            solver = "MOSEK" if "MOSEK" in cp.installed_solvers() else "SCS"
        if str(solver).upper() == "SCS":
            opts = {"eps": 1e-8, "max_iters": 200000, **opts}
        value = task.solve(solver=solver, **opts)
        return Result(None if value is None else float(value), task.status,
                      self.matrix_size, self.n_moments, time.perf_counter()-start)

    def maximize(self, expression, constraints=(), solver=None, **opts):
        return self._optimize(expression, True, constraints, solver, opts)

    def minimize(self, expression, constraints=(), solver=None, **opts):
        return self._optimize(expression, False, constraints, solver, opts)

    # ------------------------ deterministic classical maximum -------------
    def local_bound(self, expression):
        """Enumerate all but the last party; optimize the last party's response."""
        last = self.parties - 1
        terms = self.parse_sum(expression)
        prepared = []
        constant = 0.0
        for coef, atom in terms:
            if atom is None:
                constant += coef
                continue
            kind, items = atom
            others = [(p,x,a) for p,x,a in items if p != last]
            last_item = next(((x,a) for p,x,a in items if p == last), None)
            prepared.append((coef, others, last_item))

        measurements = [(p,x) for p in range(last) for x in range(len(self.outputs[p]))]
        ranges = [range(self.outputs[p][x]) for p,x in measurements]
        best = -float("inf")
        for decisions in product(*ranges):
            assignment = dict(zip(measurements, decisions))
            # Each setting of the last party can independently choose its best outcome.
            scores = [[0.0] * d for d in self.outputs[last]]
            total = constant
            for coefficient, others, last_item in prepared:
                weight = coefficient
                for p,x,a in others:
                    actual = assignment[p,x]
                    weight *= (actual == a) if a is not None else (1 - 2*actual)
                    if weight == 0:
                        break
                if weight == 0:
                    continue
                if last_item is None:
                    total += weight
                else:
                    x,a = last_item
                    if a is None:
                        scores[x][0] += weight
                        scores[x][1] -= weight
                    else:
                        scores[x][a] += weight
            best = max(best, total + sum(max(row) for row in scores))
        return float(best)


if __name__ == "__main__":
    chsh = "<A0B0> + <A0B1> + <A1B0> - <A1B1>"
    npa = NPA(parties=2, inputs=2, outcomes=2, level="1+AB")
    print("CHSH classical:", npa.local_bound(chsh))
    print("CHSH matrix size:", len(npa.word_basis()))
    print("CHSH NPA:", npa.maximize(chsh))
