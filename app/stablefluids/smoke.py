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
        # self.sources[0:int(self.w),0:int(self.h/2)] = 0.3 * np.random.rand(int(self.w), int(self.h/2))
        # self.sources[int(self.w/2)-15:int(self.w/2)+15,int(self.h/2)-15:int(self.h/2)+15] = 1 * np.random.rand(30, 30)
        self.sources[int(self.w/2)-15:int(self.w/2)+15,-30:] = 10 * np.random.rand(30, 30)

        # Velocity grid. Stored in column-major order.
        self.v = np.zeros((self.w, self.h, 2))

        # Force grid. Stored in column-major order.
        self.F = np.zeros((self.w, self.h, 2))

        self.F[int(self.w/2)-15:int(self.w/2)+15,-30:,1] = -0.1

        # Time counter.
        self.t = 0
        # Time step.
        self.dt = dt

        # Viscosity.
        self.viscosity = 0.01

        # Vorticity confinement weight.
        self.epsilon = 0.01

        # Number of iterations to use when performing diffusion and
        # projection steps.
        self.num_steps = 20

    def step(self):

        # Run through all our velocity updates.
        # start = datetime.datetime.now()
        self.v = self.add_force(self.v, self.F)
        # end = datetime.datetime.now()
        # print("addforce time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # Add vorticity confinement force.
        self.v = self.vorticity_confinement(self.v)
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
        self.sources = np.zeros((self.w, self.h))

        self.d = self.advect(self.d, 1, 0.0, 'linear')
        self.d = self.impose_boundary(self.d, 1, 'zero')

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
        div = self.impose_boundary(div, 1, 'same')

        # Projection iteration.
        for i in range(self.num_steps):
            p[1:-1,1:-1] = 0.25 * (p[1:-1,2:] + p[1:-1,0:-2] + p[2:,1:-1] \
                + p[0:-2, 1:-1] - div[1:-1,1:-1])
            p = self.impose_boundary(p, 1, 'same')

        # Velocity minus grad of pressure.
        v[1:-1,1:-1,1] -= 0.5 * (p[1:-1,2:] - p[1:-1,0:-2])
        v[1:-1,1:-1,0] -= 0.5 * (p[2:,1:-1] - p[0:-2,1:-1])

        return v

    def vorticity_confinement(self, v):
        # Code snippet:
        # https://softologyblog.wordpress.com/2019/03/13/vorticity-confinement-for-eulerian-fluid-simulations/

        # Compute curls of each neighboring cell.
        curl_x0 = np.zeros((v.shape[0], v.shape[1]))
        curl_x1 = np.zeros((v.shape[0], v.shape[1]))
        curl_y0 = np.zeros((v.shape[0], v.shape[1]))
        curl_y1 = np.zeros((v.shape[0], v.shape[1]))

        curl_x0[2:-2,2:-2] = 0.5*(v[2:-2,3:-1,0] - v[2:-2,1:-3,0] - (v[2:-2,2:-2,1] - v[0:-4,2:-2,1]))
        curl_x1[2:-2,2:-2] = 0.5*(v[2:-2,3:-1,0] - v[2:-2,1:-3,0] - (v[4:,2:-2,1] - v[2:-2,2:-2,1]))
        curl_y0[2:-2,2:-2] = 0.5*(-v[2:-2,2:-2,1] + v[2:-2,0:-4,1] + (v[3:-1,2:-2,0] - v[1:-3,2:-2,0]))
        curl_y1[2:-2,2:-2] = 0.5*(-v[2:-2,4:,1] + v[2:-2,2:-2,1] + (v[3:-1,2:-2,0] - v[1:-3,2:-2,0]))

        # Impose boundaries.
        for curl in [curl_x0, curl_x1, curl_y0, curl_y1]:
            self.impose_boundary(curl[1:-1,1:-1], 1, 'same')
            self.impose_boundary(curl, 1, 'same')

        dx = np.abs(curl_y0) - np.abs(curl_y1)
        dy = np.abs(curl_x1) - np.abs(curl_x0)

        length = np.sqrt(dx**2 + dy**2) + 1e-5

        dx = self.epsilon / length * dx
        dy = self.epsilon / length * dy

        centered_curl = np.zeros((v.shape[0], v.shape[1]))
        centered_curl[2:-2,2:-2] = 0.5*(v[2:-2,3:-1,0] - v[2:-2,1:-3,0] - (v[3:-1,2:-2,1] - v[1:-3,2:-2,1]))
        self.impose_boundary(centered_curl[1:-1,1:-1], 1, 'same')
        self.impose_boundary(centered_curl, 1, 'same')

        v[:,:,0] += self.dt * dx * centered_curl
        v[:,:,1] += self.dt * dy * centered_curl

        return v

    def in_bounds(self, point):
        return point[0] >= 0.0 and point[1] >= 0.0 \
            and point[0] <= self.w - 1 and point[1] <= self.h - 1

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
