import numpy as np
import scipy.interpolate
from scipy import ndimage
import datetime

class ParticleSmoke():

    def __init__(self, w, h, n_particles=5000, dt=1, isocontour_scale=6):
        # Grid width and height. Pad with dummy cells for boundary conditions.
        self.w = w+2
        self.h = h+2

        # Scale of isocontour grid
        self.isocontour_scale = isocontour_scale

        # Density grid. Stored in column-major order.
        # 1st dimension = x coordinate, second dimension = y coordinate.
        self.d = np.zeros((self.w, self.h))

        # Isocontour grid.
        self.i = np.zeros((self.w * self.isocontour_scale, \
            self.h * self.isocontour_scale))

        # Particle list. Each particle has only a position, since our
        # velocity is a grid.
        self.n_particles = n_particles
        self.p = np.zeros((n_particles, 2))
        self.p[:,0] = int(self.w/2) + 5 * (np.random.rand(n_particles) - 0.5)
        # self.p[:,1] = self.h - 3 - 50 * (np.random.rand(n_particles))
        self.p[:,1] = self.h - 3 - 5 * (np.random.rand(n_particles))

        # Density sources.
        self.sources = np.zeros((self.w, self.h))
        # self.sources[0:int(self.w),0:int(self.h/2)] = 0.3 * np.random.rand(int(self.w), int(self.h/2))
        # self.sources[int(self.w/2)-15:int(self.w/2)+15,int(self.h/2)-15:int(self.h/2)+15] = 1 * np.random.rand(30, 30)
        # self.sources[int(self.w/2)-15:int(self.w/2)+15,0:30] = 10 * np.random.rand(30, 30)

        # Velocity grid. Stored in column-major order.
        self.v = np.zeros((self.w, self.h, 2))

        # Force grid. Stored in column-major order.
        self.F = np.zeros((self.w, self.h, 2))

        # self.F[int(self.w/2)-15:int(self.w/2)+15,-30:,1] = -0.1
        self.F[:,:,1] = -0.1
        # self.F[:,:,0] = 10 * (np.random.rand(self.w, self.h) - 0.5)

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

        # self.F[:,:,1] = -0.05
        # self.F[:,:,1] = 0

        # if (self.t > 150 and self.t % 100 == 0):
            # self.F[40:60,:,1] = 10 * (np.random.rand(20, self.h))

        # self.F[:,:,0] = 1 * (np.random.rand(self.w, self.h) - 0.5)

        # Convert particles to densities to use in projetion step.
        self.d = self.particles_to_density(self.p)

        # start = datetime.datetime.now()
        self.v = self.advect(self.v, 2, 0.0, 'linear')
        # end = datetime.datetime.now()
        # print("advect time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

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
        self.v = self.diffuse(self.v, self.viscosity, 2, 'collision')
        # end = datetime.datetime.now()
        # print("diffuse time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # start = datetime.datetime.now()
        self.v = self.project(self.v)
        # end = datetime.datetime.now()
        # print("project time:", end.microsecond - start.microsecond)
        self.v = self.impose_boundary(self.v, 2, 'collision')

        # Advect the particles.
        self.p = self.advect_particles(self.p, self.v)

        # Update timestep.
        self.t += self.dt

        return np.transpose(self.particles_to_isocontour(self.p)[1:-1,1:-1])

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
                backtraced_locations_reshaped, order=3, mode='constant', cval=fill)
            interpolated_y = ndimage.map_coordinates(data[:,:,1],
                backtraced_locations_reshaped, order=3, mode='constant', cval=fill)
            interpolated = np.stack([interpolated_x, interpolated_y], axis=-1)
        else:
            interpolated = ndimage.map_coordinates(data,
                backtraced_locations_reshaped, order=3, mode='constant', cval=fill)

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

        # Volume conservation. THIS IS A CRUDE HACK.
        # Neighborhood density estimate?
        max_density = 3
        k = 0.2
        volume_conserve = self.particles_to_density(self.p) - max_density
        volume_conserve[volume_conserve < 0] = 0
        div -= np.sqrt(k*volume_conserve)
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

    def advect_particles(self, p, v):
        # Sample velocity grid at particle positions.
        interpolated_x = ndimage.map_coordinates(v[:,:,0],
            p.transpose(), order=1, mode='constant', cval=0.0)
        interpolated_y = ndimage.map_coordinates(v[:,:,1],
            p.transpose(), order=1, mode='constant', cval=0.0)
        p[:,0] += interpolated_x * self.dt
        p[:,1] += interpolated_y * self.dt

        # Collisions are INELASTIC, so this is correct:
        left_boundary = p[:,0] < 1
        bottom_boundary = p[:,1] < 1
        p[left_boundary,0] = 1
        p[bottom_boundary,1] = 1

        right_boundary = p[:,0] > self.w-2
        top_boundary = p[:,1] > self.h-2
        p[right_boundary,0] = self.w-2
        p[top_boundary,1] = self.h-2

        return p

    def particles_to_density(self, p):
        d = np.zeros((self.w, self.h))
        unique, counts = np.unique(p.astype(int), return_counts=True, axis=0)
        d[unique[:,0], unique[:,1]] += counts
        return d

    def particles_to_isocontour(self, p):
        # d = np.zeros((self.w, self.h))
        # unique, counts = np.unique(p.astype(int), return_counts=True, axis=0)
        # # Each particle defines a neighborhood of points.
        # neighborhood = stack
        # d[unique[:,0], unique[:,1]] = 1
        upres = 6
        sdf = np.zeros((self.w*upres, self.h*upres))
        p_up = upres * p
        radius = 10
        x_range = np.arange(0, self.w*upres)
        y_range = np.arange(0, self.h*upres)
        xx, yy = np.meshgrid(x_range, y_range)
        # Use x, y to fit with velocity grid's order.
        grid = np.stack([np.transpose(xx), np.transpose(yy)], axis=-1)
        for k in range(p_up.shape[0]):
            pos = p_up[k]
            # diff_sq = np.sum((pos - grid) ** 2, axis=2)
            # sdf[diff_sq < radius ** radius] = 1
            low_x = max(int(pos[0])-radius, 0)
            hi_x = min(int(pos[0])+radius+1, self.w * upres - 1)
            low_y = max(int(pos[1])-radius, 0)
            hi_y = min(int(pos[1])+radius+1, self.h * upres - 1)
            sdf[low_x:hi_x, low_y:hi_y] = np.ones((hi_x-low_x,hi_y-low_y))
        # Smooth

        return sdf
