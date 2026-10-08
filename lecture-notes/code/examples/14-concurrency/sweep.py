"""A parameter sweep over processes: the Lorenz system of Chapter 1.

Each parameter value is independent work, which is the case where processes
help. Run it as a script, not in a notebook cell:

    poetry run python examples/14-concurrency/sweep.py
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp


def largest_excursion(rho, sigma=10.0, beta=8.0 / 3.0):                #<1>
    """Integrate the Lorenz system and return max |u_3| after the transient."""
    def field(t, u):
        return [sigma * (u[1] - u[0]),
                u[0] * (rho - u[2]) - u[1],
                u[0] * u[1] - beta * u[2]]

    sol = solve_ivp(field, (0, 200), [1.0, 1.0, 1.0], method="DOP853",
                    rtol=1e-10, atol=1e-12, dense_output=True)
    t = np.linspace(100, 200, 20_000)                                  # drop the transient
    return rho, float(np.abs(sol.sol(t)[2]).max())


if __name__ == "__main__":                                             #<2>
    rhos = np.linspace(20.0, 200.0, 180)

    t0 = time.perf_counter()
    serial = [largest_excursion(r) for r in rhos]
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:                                #<3>
        parallel = list(pool.map(largest_excursion, rhos, chunksize=4))  #<4>
    t_parallel = time.perf_counter() - t0

    assert np.allclose([v for _, v in serial], [v for _, v in parallel])
    print(f"{len(rhos)} values of rho on {os.cpu_count()} cores")
    print(f"  serial   {t_serial:6.1f} s")
    print(f"  parallel {t_parallel:6.1f} s   speed-up {t_serial / t_parallel:.1f}x")
