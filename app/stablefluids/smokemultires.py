import numpy as np
import scipy.interpolate
from scipy import ndimage
import datetime
import cv2
from fluidgan.fluid_autoencoder import FluidAutoencoder

class SmokeMultiRes():

    def __init__(self, w, h, dt=2):
        ####################### METADATA #######################

        # Frame counter.
        self.frame = 0

        # Number of iterations to use when performing diffusion and
        # projection steps.
        self.num_steps = 20

        ##################### END METADATA #####################


        ######################## GRIDS #########################

        # NOTES:
        #   - All grids are in column major order. I.e.,
        #   1st dimension = x coordinate,
        #   2nd dimension = y coordinate.
        #   - All 2 dimensional quantities are stored [x, y].

        # Width and height of grids. Pad with one layer of dummy
        # cells for imposing boundary conditions.
        self.w = w+2
        self.h = h+2

        # Time counter.
        self.t = 0

        # Fluid density.
        self.d = np.zeros((self.w, self.h))

        # Density sources.
        self.sources = np.zeros((self.w, self.h))

        # Fluid velocity.
        self.v = np.zeros((self.w, self.h, 2))

        # Non-mouse forces. Aka, "velocity sources".
        self.F = np.zeros((self.w, self.h, 2))

        # Force applied by mouse. Separated out so we can
        # apply attenuation.
        self.F_mouse = np.zeros((self.w, self.h, 2))

        ###################### END GRIDS #######################


        ################## PHYSICAL CONSTANTS ##################

        # Time step.
        self.dt = dt

        # Gravitational force. To make this quantity scale
        # properly with the fluid sim's size, we need to
        # multiply it by the grid's scale, which we'll take
        # to be the unit length.
        self.g = -1e-4 * self.w

        # Viscosity. In theory, should be scaled, but effect is
        # negligible.
        self.viscosity = 0.01

        # Vorticity confinement weight. Also has to be scaled by
        # width---technically a scaling factor on the force, but
        # rolling it into epsilon is more convenient.
        self.epsilon = 2e-4 * self.w

        # Max flow rate of density sources.
        self.flow_rate = 0.005

        # Strength of the mouse force. Doesn't need to be scaled
        # by grid size because it is based dx's and dy's computed
        # in the grid's coordinate system.
        self.mouse_force = 0.05

        # Controls how quickly the force applied by the mouse
        # dissipates.
        # 1 -> never.
        # 0 -> immediately.
        self.mouse_attenuation_factor = 0.95

        # Radius of mouse's area of effect. In practice, this
        # is a square grid, but radius is an ok way of thinking
        # about it.
        self.mouse_aoe_radius = max(1, int(0.03 * self.w))

        ################ END PHYSICAL CONSTANTS ################


        ###################### GRID SETUP ######################

        self.randomize_density_source(self.flow_rate)

        # Gravity, but only by the sources to avoid leakage.
        self.F[int(self.w/2)-int(self.w/6):int(self.w/2)+int(self.w/6), \
            -int(self.h/4):,1] = self.g

        #################### END GRID SETUP ####################

        # Autoencoder.
        self.model = FluidAutoencoder([self.h, self.w, 2])
        # Call on data before loading weights.
        self.model(np.array([self.v]))
        self.model.load_weights("fluidgan/model_weights/model_weights")

    def step(self):

        # Re-randomize sources.
        self.randomize_density_source(self.flow_rate)

        # Run through all our velocity updates.
        self.F_mouse *= 0.9
        start = datetime.datetime.now()
        self.add_force(self.v, self.F)
        self.add_force(self.v, self.F_mouse)
        end = datetime.datetime.now()
        # print("addforce time:", end.microsecond - start.microsecond)
        self.impose_boundary(self.v, 2, 'collision')

        # Add vorticity confinement force.
        start = datetime.datetime.now()
        self.vorticity_confinement(self.v)
        end = datetime.datetime.now()
        # print("vorticity confinement time:", end.microsecond - start.microsecond)
        self.impose_boundary(self.v, 2, 'collision')

        # Downsample our velocity.
        self.v = cv2.resize(self.v, dsize=(int(self.w/2), int(self.h/2)),
            interpolation=cv2.INTER_LINEAR)

        start = datetime.datetime.now()
        self.v = self.advect(self.v, self.v, 2, 0.0, 'linear')
        end = datetime.datetime.now()
        # print("advect time:", end.microsecond - start.microsecond)
        self.impose_boundary(self.v, 2, 'collision')

        self.impose_boundary(self.v, 2, 'collision')

        start = datetime.datetime.now()
        self.diffuse(self.v, self.viscosity, 2, 'collision')
        end = datetime.datetime.now()
        # print("diffuse time:", end.microsecond - start.microsecond)
        self.impose_boundary(self.v, 2, 'collision')

        start = datetime.datetime.now()
        self.project(self.v)
        end = datetime.datetime.now()
        # print("project time:", end.microsecond - start.microsecond)
        self.impose_boundary(self.v, 2, 'collision')

        self.v = cv2.resize(self.v, dsize=(self.w, self.h),
            interpolation=cv2.INTER_LINEAR)
        self.impose_boundary(self.v, 2, 'collision')

        # NEURAL NET:
        start = datetime.datetime.now()
        changes = ((self.model(np.array([self.v]))).numpy()).reshape(160,160,2) 
        temp_v = self.v + changes 
        # self.v = temp_v
        end = datetime.datetime.now()
        # print("neural net time:", end.microsecond - start.microsecond)
        self.impose_boundary(temp_v, 2, 'collision')

        # Run through all our density updates.
        self.add_force(self.d, self.sources)

        self.d = self.advect(self.d, temp_v, 1, 0.0, 'linear')
        self.impose_boundary(self.d, 1, 'zero')

        # Update timestep.
        self.t += self.dt
        self.frame += 1
        return np.transpose(self.d[1:-1,1:-1])

    def add_force(self, data, force):
        # Just take one first order step.
        data += force * self.dt

    def advect(self, data, v, dim, fill, interp_method, collision=True):
        # Get a grid of cell indices (cell center point locations).
        x_range = np.arange(0, data.shape[0])
        y_range = np.arange(0, data.shape[1])
        xx, yy = np.meshgrid(x_range, y_range)

        # Use x, y to fit with velocity grid's order.
        grid = np.stack([np.transpose(xx), np.transpose(yy)], axis=-1)

        # Trace those points backward in time using the velocity field.
        backtraced_locations = grid - self.dt * v 
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
            self.impose_boundary(v_new, dim, boundary_type)
        np.copyto(v, v_new)

    def project(self, v):
        div = np.zeros((v.shape[0], v.shape[1]))
        p = np.zeros((v.shape[0], v.shape[1]))
        div[1:-1,1:-1] = 0.5 * (v[1:-1,2:,1] - v[1:-1,0:-2,1] \
            + v[2:,1:-1,0] - v[0:-2,1:-1,0])
        self.impose_boundary(div, 1, 'same')

        # Projection iteration.
        for i in range(self.num_steps):
            p[1:-1,1:-1] = 0.25 * (p[1:-1,2:] + p[1:-1,0:-2] + p[2:,1:-1] \
                + p[0:-2, 1:-1] - div[1:-1,1:-1])
            self.impose_boundary(p, 1, 'same')

        # Velocity minus grad of pressure.
        v[1:-1,1:-1,1] -= 0.5 * (p[1:-1,2:] - p[1:-1,0:-2])
        v[1:-1,1:-1,0] -= 0.5 * (p[2:,1:-1] - p[0:-2,1:-1])

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
            data[:,0] = np.stack([-data[:,1,0], -data[:,1,1]], axis=-1)
            data[:,-1] = np.stack([-data[:,-2,0], -data[:,-2,1]], axis=-1)

    def randomize_density_source(self, flow_rate):
        self.sources[int(self.w/2)-int(self.w/6):int(self.w/2)+int(self.w/6), \
            -int(self.h/4):] = flow_rate * np.random.rand(2*int(self.w/6), \
            int(self.h/4))

    def update_mouse_force(self, px, py, dx, dy):
        x_low = max(px-self.mouse_aoe_radius,0)
        x_high = min(px+self.mouse_aoe_radius, self.w-1-self.mouse_aoe_radius)
        y_low = max(py-self.mouse_aoe_radius,0)
        y_high = min(py+self.mouse_aoe_radius, self.h-1-self.mouse_aoe_radius)
        self.F_mouse[x_low:x_high, y_low:y_high] += \
            self.mouse_force * np.array([dx, dy])
