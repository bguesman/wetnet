import numpy as np
import scipy.interpolate

class Smoke():

    def __init__(self, w, h, dt=1):
        # Grid width and height.
        self.w = w
        self.h = h
        # Density grid. Stored in row-major order.
        # 1st dimension = y coordinate, second dimension = x coordinate.
        self.d = np.full((h, w), fill_value=0.0)
        self.d[0:int(h/2),:] = 0.8
        # Velocity grid. Stored in row-major order.
        self.v = np.full((h, w, 2), fill_value=0.01)
        # Force grid. Stored in row-major order.
        self.F = np.full((h, w, 2), fill_value=0.0)
        # Time counter.
        self.t = 0
        # Time step.
        self.dt = dt

    def step(self):
        # Run through all our velocity updates.
        self.add_force()
        self.v = self.advect(self.v, 2, 0.0, 'nearest')
        self.diffuse()
        self.project()
        self.v = self.impose_boundary(self.v, 2, 'zero')

        # Run through all our density updates.
        self.d = self.advect(self.d, 1, 0.0, 'linear')

        # Update timestep.
        self.t += self.dt
        return self.d

    def add_force(self):
        # Just take one first order step.
        self.v += self.F * self.dt

    def advect(self, data, dim, fill, interp_method):
        # Get a grid of cell indices (cell center point locations).
        x_range = np.arange(0, self.w)
        y_range = np.arange(0, self.h)
        xx, yy = np.meshgrid(x_range, y_range)
        grid = np.stack([yy, xx], axis=-1)
        # Trace those points backward in time using the velocity field.
        backtraced_locations = grid - self.dt * self.v
        # Sample the velocity at those points, set it to the new velocity.
        interpolated = scipy.interpolate.griddata(grid.reshape(-1,2),
            data.reshape(-1, dim), backtraced_locations.reshape(-1,2),
            method=interp_method, fill_value=fill)
        # Make sure to reshape back to a grid!
        return interpolated.reshape(data.shape)

    def diffuse(self):
        pass

    def project(self):
        pass

    def impose_boundary(self, data, dim, type):
        if (type == 'zero'):
            data[:,[0,-1]] = data[[0,-1]] = np.zeros(dim)
        if (type == 'no_slip'):
            assert data.shape[2] == 2, "no slip boundary requires vector field"
            # Top and bottom rows.
            for i in range(self.w):
                data[0, i] = np.array([data[0, i][0], 0])
                data[data.shape[0]-1, i] = np.array([data[0, i][0], 0])
            # Left and right columns.
            for i in range(self.h):
                data[i, 0] = np.array([0, data[i, 0][1]])
                data[i, data.shape[1]-1] = np.array([0, data[i, 0][1]])
        return data
