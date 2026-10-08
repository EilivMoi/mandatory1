import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")

# Below we create a solver that reuses some of the implementation from
# the 1D solver in poisson.py.


class Poisson2D:
    r"""Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), x, y in [0, L] x [0, L]

    with Dirichlet boundary conditions.
    """

    def __init__(self, L: float):
        self.p = Poisson(L)  # we can reuse some of the code from the 1D case

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a 2D Cartesian mesh

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh
        """
        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    @staticmethod
    def D2(N: int) -> sparse.lil.matrix:
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')

        D[0, :] = 0
        D[0,0]=1
        D[-1, :] = 0
        D[-1,-1] = 1
        return D

    def laplace(self, N: int) -> sparse.lil_matrix:
        """Return a vectorized Laplace operator

        Parameters
        ----------
        N : int
            The number of uniform intervals in both x and y directions

        Returns
        -------
        A : scipy sparse LIL matrix
            The vectorized Laplace operator
        
        from lecture 6
        def laplace(dx, dy, Nx, Ny):
            D2x = (1./dx**2)*D2(Nx)
            D2y = (1./dy**2)*D2(Ny)
            return (sparse.kron(D2x, sparse.eye(Ny+1)) + sparse.kron(sparse.eye(Nx+1), D2y))
        """

        h = self.p.L / N
        D2x = (1./h**2) * self.D2(N) # 1‑D operator in x
        D2y = (1./h**2) * self.D2(N) # identical in y (square domain)
        A = sparse.kron(D2x, sparse.eye(N+1)) + sparse.kron(sparse.eye(N+1), D2y)      
        
        return A.tocsr()


    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Return assembled coefficient matrix A and right hand side vector b

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        f : Sympy expression
            The right hand side as a Sympy expression in x and y
        ue : Sympy expression
            The exact solution as a Sympy expression in x and y

        Returns
        -------
        A : scipy sparse CSR matrix
            Coefficient matrix
        b : 1D array
            Right hand side vector

        Note
        ----
        Compute the Kronecker product of the 1D Laplace operator with itself
        to create the 2D Laplace operator. Then, assemble the right-hand side
        vector b by evaluating the function f at the mesh points and applying
        Dirichlet boundary conditions using the exact solution ue.

        """

        # mesh

        xij, yij = self.create_mesh(N)
        h = self.p.L/N

        # matrix A
        A = self.laplace(N)

        # source and interior values
        f_mesh = sp.lambdify((x,y), f, "numpy")(xij, yij)
        b = f_mesh.ravel()

        # dirichlet boundary values
        ue_mesh = sp.lambdify((x,y), ue, "numpy")(xij, yij)
        boundary = self.get_boundary_indices(N)

        A[boundary, :] = 0
        A[boundary, boundary] = 1

        b[boundary] = ue_mesh.ravel()[boundary]

        return A, b
    

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        """Return Sympy function as mesh function

        Parameters
        ----------
        u : Sympy function

        Returns
        -------
        array - The input function as a mesh function
        """

        f = sp.lambdify((x,y), u, "numpy")
        return f(xij,yij)


    def get_boundary_indices(self, N: int) -> np.ndarray:
        """Return indices of vectorized matrix that belongs to the boundary"""

        idx = np.arange((N+1)*(N+1))
        i = idx // (N+1)
        j = idx % (N+1)
        mask = (i == 0) | (i == N) | (j == 0 ) | (j == N)
        return idx[mask]

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        """Return l2-error

        Parameters
        ----------
        u : array
            The numerical solution (mesh function)
        ue : Sympy expression
            The exact solution

        Returns
        -------
        float - The l2-error

        """

        N = u.shape[0]- 1
        dx = self.p.L/N
        dy = dx 

        xij, yij = self.create_mesh(N)
        ue_mesh = sp.lambdify((x,y), ue, "numpy")(xij, yij)

        err = u - ue_mesh

        err_c = (err[:-1, :-1] + err[1:, :-1] + err[-1:, 1:] + err[1:, 1:])/4

        return np.sqrt(np.sum(err_c**2)* dx * dy)
        

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        """Solve Poisson's equation with a given manufactured solution

        Parameters
        ----------
        Nx : int
            The number of uniform intervals in both x and y directions
        ue : Sympy expression
            The exact solution

        Returns
        -------
        The solution as a Numpy array

        """
        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, U: np.ndarray, x: float, y: float) -> float:
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """

        N = U.shape[0] - 1
        h = self.p.L / N
        i  = int(np.floor( x/h))
        j = int(np.floor(y/h))

        xi = (x - i*h) / h
        eta = (y - j*h) / h

        u00 = U[i, j]
        u10 = U[i + 1, j]
        u01 = U[i, j + 1]
        u11 = U[i + 1, j + 1]

        return (1-xi)*(1-eta)*u00 + xi*(1-eta)*u10 + (1-xi)*eta*u01 + xi*eta*u11
    


def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2


def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3


if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
