import numpy as np
import scipy.interpolate

class Smoke():

    def __init__(self, w, h, dt=3):
        # Grid width and height. Pad with dummy cells for boundary conditions.
        self.w = w+2
        self.h = h+2

        # Density grid. Stored in column-major order.
        # 1st dimension = x coordinate, second dimension = y coordinate.
        self.d = np.zeros((self.w, self.h))

        # Density sources.
        self.sources = np.zeros((self.w, self.h))
        self.sources[int(self.w/2)-1:int(self.w/2)+1,int(self.h/2)-1:int(self.h/2)+1] = 0.3

        # Velocity grid. Stored in column-major order.
        self.v = np.zeros((self.w, self.h, 2))

        # Force grid. Stored in column-major order.
        self.F = np.zeros((self.w, self.h, 2))
        self.F[int(self.w/2)-2:int(self.w/2)+2,:,1] = -0.01

        # Time counter.
        self.t = 0
        # Time step.
        self.dt = dt

        # Viscosity.
        self.viscosity = 0.001

    def step(self):
        # Run through all our velocity updates.
        self.v = self.add_force(self.v, self.F)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        self.v = self.advect(self.v, 2, 0.0, 'linear')
        self.v = self.impose_boundary(self.v, 2, 'collision')

        self.v = self.diffuse(self.v, self.viscosity, 2, 'collision')
        self.v = self.impose_boundary(self.v, 2, 'collision')

        self.v = self.project(self.v)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # Run through all our density updates.
        self.d = self.add_force(self.d, self.sources)

        self.d = self.advect(self.d, 1, 0.0, 'linear')
        self.d = self.impose_boundary(self.d, 1, 'same')

        # Update timestep.
        self.t += self.dt
        return np.transpose(self.d[1:-1,1:-1])

    def add_force(self, data, force):
        # Just take one first order step.
        return data + force * self.dt

    def advect(self, data, dim, fill, interp_method, collision=True):
        # Get a grid of cell indices (cell center point locations).
        x_range = np.arange(0, self.w)
        y_range = np.arange(0, self.h)
        xx, yy = np.meshgrid(x_range, y_range)

        # Use x, y to fit with velocity grid's order.
        grid = np.stack([np.transpose(xx), np.transpose(yy)], axis=-1)

        # Trace those points backward in time using the velocity field.
        backtraced_locations = grid - self.dt * self.v
        if (collision):
            backtraced_locations = np.abs(backtraced_locations)

        # Sample the velocity at those points, set it to the new velocity.
        interpolated = scipy.interpolate.griddata(grid.reshape(-1,2),
            data.reshape(-1, dim), (backtraced_locations[:,:,0], backtraced_locations[:,:,1]),# backtraced_locations.reshape(-1,2),
            method=interp_method, fill_value=fill)

        # Make sure to reshape back to a grid!
        interpolated = interpolated.reshape(data.shape)

        return interpolated

    def diffuse(self, v, rate, dim, boundary_type):
        a = self.dt * rate
        v_new = np.zeros(v.shape)
        for i in range(20):
            v_new[1:-1,1:-1] = (1.0 / (4.0 * a + 1.0)) * \
                (a*(v_new[2:,1:-1] + v_new[0:-2,1:-1]
                + v_new[1:-1,2:] + v_new[1:-1,0:-2]) + v[1:-1,1:-1])
            v_new = self.impose_boundary(v_new, dim, boundary_type)
        return v_new

    # TODO: seems to be biased toward positive grid edges in some way?
    def project(self, v):
        div = np.zeros((self.w, self.h))
        p = np.zeros((self.w, self.h))
        div[1:-1,1:-1] = 0.5 * (v[1:-1,2:,1] - v[1:-1,0:-2,1] \
            + v[2:,1:-1,0] - v[0:-2,1:-1,0])

        # Above vectorizes this:
        # for x in range(1, self.w-1):
        #     for y in range(1, self.h-1):
        #         # Finite differences approximation of divergence.
        #         div[x,y] = 0.5 * (v[x+1, y][0] - v[x-1, y][0]
        #             + v[x, y+1][1]-v[x,y-1][1])

        div = self.impose_boundary(div, 1, 'same')

        # Projection iteration.
        for i in range(20):
            p[1:-1,1:-1] = 0.25 * (p[1:-1,2:] + p[1:-1,0:-2] + p[2:,1:-1] \
                + p[0:-2, 1:-1] - div[1:-1,1:-1])

            # Above vectorizes this:
            # for x in range(1, self.w-1):
            #     for y in range(1, self.h-1):
            #         p[x,y] = 0.25 * (p[x+1, y] + p[x-1, y]
            #             + p[x, y+1]+p[x,y-1]-div[x,y])

            p = self.impose_boundary(p, 1, 'same')

        # Velocity minus grad of pressure.
        v[1:-1,1:-1,1] -= 0.5 * (p[1:-1,2:] - p[1:-1,0:-2])
        v[1:-1,1:-1,0] -= 0.5 * (p[2:,1:-1] - p[0:-2,1:-1])
        # Above vectorizes this:
        # for x in range(1, self.w-1):
            # for y in range(1, self.h-1):
            #     v[x,y,0] -= 0.5 * (p[x+1, y] - p[x-1, y])
            #     v[x,y,1] -= 0.5 * (p[x, y+1] - p[x, y-1])

        return v


    def in_bounds(self, point):
        return point[0] >= 0.0 and point[1] >= 0.0 \
            and point[0] <= self.w - 1 and point[1] <= self.h - 1

    # TODO: vectorize.
    def impose_boundary(self, data, dim, type):
        if (type == 'zero'):
            data[:,[0,-1]] = data[[0,-1]] = np.zeros(dim)
        if (type == 'same'):
            # Left and right columns.
            data[0, :] = data[1, :]
            data[-1, :] = data[-2, :]
            # Top and bottom rows.
            data[:, 0] = data[:, 1]
            data[:, -1] = data[:, -2]
            # Corners.
            data[0,0] = 0.5 * (data[1,0] + data[0,1])
            data[-1,-1] = 0.5 * (data[-2,-1] + data[-1,-2])
            data[0,-1] = 0.5 * (data[0,-2] + data[1,-1])
            data[-1,0] = 0.5 * (data[-2,0] + data[1,-1])
        if (type == 'no_slip'):
            assert (dim == 2)
            # Left and right columns.
            for y in range(self.h):
                data[0, y] = np.array([0, data[1, y][1]])
                data[-1, y] = np.array([0, data[-2, y][1]])
            # Top and bottom rows.
            for x in range(self.w):
                data[x, 0] = np.array([data[x, 1][0], 0])
                data[x, -1] = np.array([data[x, -2][0], 0])
        if (type == 'collision'):
            assert (dim == 2)
            # Left and right columns.

            # for y in range(self.h):
            data[0,:] = np.stack([-data[1,:,0], data[1,:,1]], axis=-1)
            data[-1,:] = np.stack([-data[-2,:,0], data[-2,:,1]], axis=-1)
            # Top and bottom rows.
            data[:,0] = np.stack([data[:,1,0], -data[:,1,1]], axis=-1)
            data[:,-1] = np.stack([data[:,-2,0], -data[:,-2,1]], axis=-1)
        return data
