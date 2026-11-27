"""Two-qubit H2 effective Hamiltonian. Energy units are Hartree."""
from math import atan2, cos, hypot, sin
from random import Random

from shotweave import optimal_allocation
from shotweave.applications import grouped_energy

# O'Malley et al., Phys. Rev. X 6, 031007 (2016), R=0.75 angstrom.
# Rounded coefficients define the numerical model and its exact reference.
OFFSET, ZI, IZ, ZZ, YY, XX = -.4804, .3435, -.4347, .5716, .0910, .0910


def benchmark(budget=1000, seed=47):
    # State cos(t/2)|01> + sin(t/2)|10>. In this sector XX=YY in mean.
    delta, coupling = ZI-IZ,XX+YY
    theta = atan2(-coupling,-delta)
    exact = min(OFFSET-ZZ-hypot(delta,coupling), OFFSET+ZZ-hypot(ZI+IZ,XX-YY))
    rng = Random(seed)
    z,x,y = optimal_allocation(budget,[abs(delta*sin(theta)),abs(XX*cos(theta)),abs(YY*cos(theta))])
    def outcomes(n,mu):
        return [1 if rng.random()<(1+mu)/2 else -1 for _ in range(n)]
    observations = [[[a,-a,-1] for a in outcomes(z,cos(theta))],
                    [[a] for a in outcomes(x,sin(theta))],[[a] for a in outcomes(y,sin(theta))]]
    value,variance = grouped_energy(OFFSET,[[ZI,IZ,ZZ],[XX],[YY]],observations)
    return exact,value,variance,(z,x,y)


if __name__ == "__main__":
    print(benchmark())
