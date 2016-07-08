from sympy import *
from sympy.printing import print_ccode
from sympy.physics.vector import ReferenceFrame, gradient, divergence
from sympy.vector import CoordSysCartesian

R = ReferenceFrame('R');
x = R[0]; y = R[1];

eps = 1e-4;

u=sin(x)*cos(y);
grad_u = gradient(u, R);#.to_matrix(R);
print(u)
print(grad_u)

f = -divergence(pow(eps+grad_u.magnitude()**2.0,0.5) * grad_u, R);
print("\n * RHS:")
print(ccode(f, assign_to = "values").replace("R_","").replace("2)","2.0)"));

print("\n * ExactSolution:")
print(ccode(u, assign_to = "values"));

print("")
#print("pressure mean:", N(integrate(p,(x,a,b))))
