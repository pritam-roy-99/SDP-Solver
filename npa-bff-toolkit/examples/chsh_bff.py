"""BFF conditional entropy H(A|E) under a CHSH constraint."""
from qresource import BFFNPA

CHSH = '<A0B0> + <A0B1> + <A1B0> - <A1B1>'
model = BFFNPA(outputs=[[2, 2], [2, 2]], key_settings={0: 0},
               bff_order=1, level=2)
result = model.solve(statistics=[(CHSH, '>=', 2.6)], solver='SCS', eps=1e-7)
print(result)
print('Entropy numerical estimate (bits):', result.entropy_bits)
print('Numerical diagnostics passed:', result.reliable)
print('Constraint residual:', result.max_stat_violation)
