"""I3322-style probability inequality: Collins--Gisin convention."""
from qresource import NPA

expr = (
    '-pA(0|0)-2*pB(0|0)-pB(0|1)'
    '+p(00|00)+p(00|01)+p(00|02)'
    '+p(00|10)+p(00|11)-p(00|12)'
    '+p(00|20)-p(00|21)'
)
model = NPA(parties=2, inputs=3, outcomes=2, level=2)
print('Local maximum:', model.local_bound(expr))
print('NPA upper estimate:', model.maximize(expr, solver='SCS'))
