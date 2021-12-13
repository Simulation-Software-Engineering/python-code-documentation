"""
Solving the two-dimensional diffusion equation

Example acquired from https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
"""

import numpy as np
import matplotlib.pyplot as plt

def create_figure( fig, plt, u, n, fignum, dt, T_cold, T_hot ):
    """
    A method to create a subfigure that has the correct
    color scaling and title set
    """
    fignum += 1
    ax = fig.add_subplot(220 + fignum)
    im = ax.imshow(u.copy(), cmap=plt.get_cmap('hot'), vmin=T_cold, vmax=T_hot)
    ax.set_axis_off()
    ax.set_title('{:.1f} ms'.format(n * dt * 1000))

    return fignum, im

def output_figure( fig, plt, im ):
    """
    A method to output the created figures on screen
    and adds a corresponding colorbar
    """
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar_ax.set_xlabel('$T$ / K', labelpad=20)
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

class DiffusionSolver():
    """
    A class describing a finite-difference method for solving
    the heat equation in 2 space dimensions.
    """

    def __init__(self, h = 10. , dx = 0.1, D = 4., T_cold = 300, T_hot = 700):
        """Constructor"""
        self.h = h
        self.w = self.h
        self.dx = dx
        self.dy = self.dx
        self.D = D
        self.T_cold = T_cold
        self.T_hot = T_hot

    def solve(self):
        """
        Solves the 2D heat equation for the parameters specified
        in the constructor
        """
        nx, ny = int(self.w / self.dx), int(self.h / self.dy)

        dx2, dy2 = self.dx * self.dx, self.dy * self.dy
        dt = dx2 * dy2 / (2 * self.D * (dx2 + dy2))

        u0 = self.T_cold * np.ones((nx, ny))
        u = u0.copy()

        # Initial conditions - circle of radius r centred at (cx,cy) (mm)
        r, cx, cy = 2, 5, 5
        r2 = r ** 2
        for i in range(nx):
            for j in range(ny):
                p2 = (i * self.dx - cx) ** 2 + (j * self.dy - cy) ** 2
                if p2 < r2:
                    u0[i, j] = self.T_hot

        def do_timestep(u_nm1, u):
            """Internal function that advances the solution by on time step dt"""
            # Propagate with forward-difference in time, central-difference in space
            u[1:-1, 1:-1] = u_nm1[1:-1, 1:-1] + self.D * dt * (
                    (u_nm1[2:, 1:-1] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[:-2, 1:-1]) / dx2
                    + (u_nm1[1:-1, 2:] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[1:-1, :-2]) / dy2)

            u_nm1 = u.copy()
            return u_nm1, u

        # Number of timesteps
        nsteps = 101
        # Output 4 figures at these timesteps
        n_output = [0, 10, 50, 100]
        fignum = 0
        fig = plt.figure()

        # Time loop
        for n in range(nsteps):
            u0, u = do_timestep(u0, u)

            # Create figure
            if n in n_output:
                fignum, im = create_figure( fig, plt, u, n, fignum, dt, self.T_cold, self.T_hot )
                print(n, fignum)

        # Plot output figures
        output_figure( fig, plt, im )

if __name__ == "__main__":
    diffusion_solver =  DiffusionSolver(h = 10.,
                                        dx = 0.1,
                                        D = 4.,
                                        T_cold = 300,
                                        T_hot = 700)
    diffusion_solver.solve()
