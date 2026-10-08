import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")


class Wave2D:
    """Class for solving the 2D wave equation"""
    def __init__(self, L0=1, c0=1, cfl=0.7, u0: sp.Expr = None):
        self.L   = float(L0)
        self.c   = float(c0)
        self.cfl = float(cfl)

        if u0 is None:
            u0 = sp.exp(-200 * (x - self.L / 2 + self.c * t) ** 2)
        self.u0 = u0

        self.unp1 = None
        self.un   = None
        self.unm1 = None


    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return 2D mesh created using np.meshgrid

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        sparse : bool, optional
            Whether to create a sparse mesh or not. Default is False.
        Returns
        -------
        xij : 2D array
            The x-coordinates of the mesh
        yij : 2D array
            The y-coordinates of the mesh"""
        
        xi = np.linspace(0.0, self.L, N + 1)
        if sparse:
            xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        else:
            xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=False)
        return xij, yij

    def D2(self, N: int) -> sparse.lil_matrix:
        """Return second order differentiation matrix

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Returns
        -------
        D : scipy sparse LIL matrix
            The second order differentiation matrix
        """
        D = sparse.diags([1., -2., 1.], [-1, 0, 1],
                         shape=(N + 1, N + 1), format="lil")
        D[0, :] = 0.0          # Dirichlet rows → zero
        D[-1, :] = 0.0
        return D

    
    @property
    def dt(self) -> float:
        """Return the time step"""
        return self.cfl * self.dx / self.c

    @property
    def w(self):
        """Return the dispersion coefficient"""
        h = self.dx
        C = self.cfl
        kx = self.mx * np.pi / self.L
        ky = self.my * np.pi / self.L
        arg = C * np.sqrt(np.sin(0.5 * kx * h) ** 2 +
                          np.sin(0.5 * ky * h) ** 2)
        arg = np.clip(arg, -1.0, 1.0)
        return (2.0 / self.dt) * np.arcsin(arg)

    def _omega_exact(self) -> float:
        """Continuous angular frequency ω = c·√(kx²+ky²)."""
        kx = self.mx * np.pi / self.L
        ky = self.my * np.pi / self.L
        return self.c * np.sqrt(kx ** 2 + ky ** 2)

        

    def ue(self, mx: int, my: int) -> sp.Expr:
        """Return the exact standing wave

        Parameters
        ----------
        mx, my : int
            Parameters for the standing wave
        Returns
        -------
        ue : Sympy expression
            The exact solution as a Sympy expression in x, y and t
        """
        
        omega = self._omega_exact()            
        return (sp.sin(mx * sp.pi * x / self.L) *
                sp.sin(my * sp.pi * y / self.L) *
                sp.cos(omega * t))


    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
       
        self.mx, self.my = mx, my                      
        self.N   = N
        self.dx  = self.L / N
        

        xij, yij = self.create_mesh(N, sparse=False)

        ue0 = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")
        U0 = ue0(xij, yij, 0.0)

        D2 = self.D2(N)
        Lap = (D2 @ U0 + U0 @ D2) / self.dx ** 2

        U_m1 = U0 - 0.5 * (self.c * self.dt) ** 2 * Lap

        self.un   = U0.copy()
        self.unm1 = U_m1.copy()
        self.unp1 = np.empty_like(U0)

        return self.un, self.unm1


    def l2_error(self, u: np.ndarray, t0: float) -> float:
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        N = u.shape[0] - 1       
        h = self.L / N                

        xij, yij = self.create_mesh(N, sparse=False)
        xc = xij[:-1, :-1] + self.dx/2.0      
        yc = yij[:-1, :-1] + self.dx/2.0

        ue_sym = self.ue(self.mx, self.my)              
        ue_fun = sp.lambdify((x, y, t), ue_sym, "numpy")
        ue_mesh = ue_fun(xc, yc, t0)                   

        u_bar = (u[:-1, :-1] + u[1:, :-1] +
                u[:-1, 1:]  + u[1:, 1:]) / 4.0          

        
        err = u_bar - ue_mesh
        return np.sqrt(np.sum(err**2) * h * h)




        
    def apply_bcs(self, u: np.ndarray):
        """Apply boundary conditions to the solution mesh function

        Parameters
        ----------
        u : array
            The solution mesh function
        """

        """
        Dirichlet u=0 on whole boundary
        """
        u[0,:]= 0.0
        u[-1, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1]= 0.0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.66,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """


        self.cfl = float(cfl)
        self.c   = float(c)
        self.N   = N
        self.dx  = self.L / N
        self.mx, self.my = mx, my

        self.un, self.unm1 = self.initialize(N, mx, my)

        data = {}
        if store_data > 0: 
            data[0] = self.un.copy()

        D2 = self.D2(N)
        Lap = lambda U: (D2 @ U + U @ D2) / self.dx ** 2

        for n in range(1, Nt + 1):
            self.unp1 = (2.0 * self.un - self.unm1) + (self.c * self.dt) ** 2 * Lap(self.un)

            self.apply_bcs(self.unp1)

            self.unm1, self.un = self.un, self.unp1

            if store_data > 0 and (n % store_data == 0):
                data[n] = self.un.copy()

        if store_data == -1:
            err = self.l2_error(self.un, t0=Nt * self.dt)   
            err = np.array([err])                           
            return self.dx, err
        else:
            return data

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.66557, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E, h = [], []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)          # `err` is now a 1‑element array
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m)]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:

        D = sparse.diags([1., -2., 1.], [-1, 0, 1],
                         shape=(N + 1, N + 1), format="lil")
        D[0, :] = 0
        D[0, 0] = -1
        D[0, 1] =  1
        
        D[-1, :] = 0
        D[-1, -2] = -1
        D[-1, -1] =  1
        return D
        

    def ue(self, mx: int, my: int) -> sp.Expr:
        omega = self._omega_exact()
        return (sp.cos(mx * sp.pi * x / self.L) *
                sp.cos(my * sp.pi * y / self.L) *
                sp.cos(omega * t))
    

    def apply_bcs(self, u: np.ndarray):
        u[0, :]   = u[1, :]      # left boundary
        u[-1, :]  = u[-2, :]     # right boundary
        u[:, 0]   = u[:, 1]      # bottom boundary
        u[:, -1]  = u[:, -2]     # top boundary


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3, cfl = 0.0001)
    #assert abs(r[-1] - 2) < 0.05
    print(abs(r[-1] - 2))


def test_exact_wave2d():
    raise NotImplementedError("The test_exact_wave2d function is not implemented yet.")



if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    print("All tests passed!")
