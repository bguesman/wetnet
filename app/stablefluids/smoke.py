import numpy as np
import scipy.interpolate
from scipy import ndimage
import datetime

class Smoke():

    def __init__(self, w, h, dt=1):
        # Grid width and height. Pad with dummy cells for boundary conditions.
        self.w = w+2
        self.h = h+2

        # Density grid. Stored in column-major order.
        # 1st dimension = x coordinate, second dimension = y coordinate.
        self.d = np.zeros((self.w, self.h))

        # Density sources.
        self.sources = np.zeros((self.w, self.h))
        self.sources[int(self.w/2)-15:int(self.w/2)+15,int(self.h/2)-15:int(self.h/2)+15] = 0.3 * np.random.rand(30, 30)

        # Velocity grid. Stored in column-major order.
        self.v = np.zeros((self.w, self.h, 2))

        # Force grid. Stored in column-major order.
        self.F = np.zeros((self.w, self.h, 2))

        # self.F[int(self.w/2)-15:int(self.w/2)+15,:,1] = 0.1 * (np.random.rand(30, 202) - 1)
        # self.F[int(self.w/2)-15:int(self.w/2)+15,:,0] = 0.1 * (np.random.rand(30, 202) - 0.5)
        self.F[int(self.w/2)-15:int(self.w/2)+15,int(self.w/2)-15:int(self.w/2)+15,1] = 0.4 * (np.random.rand(30, 30) - 1)
        self.F[:,:,0] = 0.2 * (np.random.rand(202, 202) - 0.5)

        # Time counter.
        self.t = 0
        # Time step.
        self.dt = dt

        # Viscosity.
        self.viscosity = 0.001

        # Number of iterations to use when performing diffusion and
        # projection steps.
        self.num_steps = 10

    def step(self):
        self.sources[int(self.w/2)-15:int(self.w/2)+15,int(self.h/2)-15:int(self.h/2)+15] = 0.1 * np.random.rand(30, 30)
        # self.F[int(self.w/2)-15:int(self.w/2)+15,:,1] = 0.1 * (np.random.rand(30, 202) - 1)
        # self.F[int(self.w/2)-15:int(self.w/2)+15,:,0] = 1 * (np.random.rand(30, 202) - 0.5)
        self.F[int(self.w/2)-15:int(self.w/2)+15,int(self.w/2)-15:int(self.w/2)+15,1] = 0.4 * (np.random.rand(30, 30) - 1)
        self.F[:,:,0] = 0.2 * (np.random.rand(202, 202) - 0.5)
        # Run through all our velocity updates.
        # start = datetime.datetime.now()
        self.v = self.add_force(self.v, self.F)
        # end = datetime.datetime.now()
        # print("addforce time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # start = datetime.datetime.now()
        self.v = self.advect(self.v, 2, 0.0, 'linear')
        # end = datetime.datetime.now()
        # print("advect time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # start = datetime.datetime.now()
        self.v = self.diffuse(self.v, self.viscosity, 2, 'collision')
        # end = datetime.datetime.now()
        # print("diffuse time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # start = datetime.datetime.now()
        self.v = self.project(self.v)
        # end = datetime.datetime.now()
        # print("project time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # Run through all our density updates.
        self.d = self.add_force(self.d, self.sources)

        self.d = self.advect(self.d, 1, 0.0, 'linear')
        self.d = self.impose_boundary(self.d, 1, 'same')

        if(self.t  == 150):
            save_d = []
            d_transpose =np.transpose(self.d[1:-1,1:-1])
            for i in range(202):
                save_d.append(d_transpose)
            save_d = np.array(save_d)
            save_v = self.v[..., np.newaxis]
            np.savez("smoke_style_transfer/data/waterfall/d/001", x=save_d)
            np.savez("smoke_style_transfer/data/waterfall/v/001", x=save_v)
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
        # interpolated = scipy.interpolate.griddata(grid.reshape(-1,2),
        #     data.reshape(-1, dim), (backtraced_locations[:,:,0], backtraced_locations[:,:,1]),# backtraced_locations.reshape(-1,2),
        #     method=interp_method, fill_value=fill)
        backtraced_locations_reshaped = backtraced_locations.reshape(-1,2).transpose()
        if (dim == 2):
            interpolated_x = ndimage.map_coordinates(data[:,:,0],
                backtraced_locations_reshaped, order=1, mode='constant', cval=fill)
            interpolated_y = ndimage.map_coordinates(data[:,:,1],
                backtraced_locations_reshaped, order=1, mode='constant', cval=fill)
            interpolated = np.stack([interpolated_x, interpolated_y], axis=-1)
        else:
            interpolated = ndimage.map_coordinates(data,
                backtraced_locations_reshaped, order=1, mode='constant', cval=fill)

        # Make sure to reshape back to a grid!
        interpolated = interpolated.reshape(data.shape)

        return interpolated

    def diffuse(self, v, rate, dim, boundary_type):
        a = self.dt * rate
        v_new = np.zeros(v.shape)
        for i in range(self.num_steps):
            v_new[1:-1,1:-1] = (1.0 / (4.0 * a + 1.0)) * \
                (a*(v_new[2:,1:-1] + v_new[0:-2,1:-1]
                + v_new[1:-1,2:] + v_new[1:-1,0:-2]) + v[1:-1,1:-1])
            v_new = self.impose_boundary(v_new, dim, boundary_type)
        return v_new

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
        for i in range(self.num_steps):
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

    def impose_boundary(self, data, dim, type):
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
