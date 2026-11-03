"""Bell-NPA CHSH example. Requires cvxpy and an SDP solver."""
from qresource import NPA

CHSH = '<A0B0> + <A0B1> + <A1B0> - <A1B1>'
model = NPA(parties=2, inputs=2, outcomes=2, level='1+AB')
print('Classical local maximum:', model.local_bound(CHSH))
print('NPA upper estimate:', model.maximize(CHSH, solver='SCS'))
