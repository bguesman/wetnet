from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import datetime
from stablefluids.smoke import Smoke
from stablefluids.smokemultires import SmokeMultiRes
from water.water import Water
from water.particle_smoke import ParticleSmoke
from render.renderer import Renderer

class TexRenderer():

	def __init__(self, window_w, window_h, sim_w, sim_h,
		mode="RUN", path="", lo_res_scale=0.3):

		# Set member variables from initialization parameters.

		# Operating mode, one of:
		#	"RUN": runs simulation for demo purposes.
		#	"GEN_DATA": runs simulation at low and high res
		#	and writes density and velocity grids to files
		#	at specified file path.
		#	"VIEW": views pre-computed simulation at specified
		#	file path.
		if (mode != "RUN" and mode != "VIEW" and \
			mode !="GEN_DATA"):
			print("Mode must be one of [RUN, VIEW, GEN_DATA].")
			exit()
		self.mode = mode
		self.path = path
		# Width and height of the window.
		self.w = window_w
		self.h = window_h

		# Wrapper for OpenGL initialization calls.
		self.initOpenGL()

		# Mouse position.
		self.mouse_x = None
		self.mouse_y = None

		# Width and height of the sim.
		self.sim_w = sim_w
		self.sim_h = sim_h

		if (self.mode == "VIEW"):
			# We need no renderer for this sim! But we do
			# need a frame number to know what frame of the
			# precomputed sim to render.
			self.frame = 0
		elif (self.mode == "GEN_DATA"):
			# We need two sims: a high res "real one"
			# and a low res "fake" one.
			self.sim = Smoke(self.sim_w, self.sim_h, \
				save_data=True, path=self.path+"/hi_res/")
			# Low res sim parameters
			self.lr_w = int(self.sim_w * lo_res_scale)
			self.lr_h = int(self.sim_h * lo_res_scale)
			self.low_res_sim = Smoke(int(self.lr_w), int(self.lr_h), \
				save_data=True, path=self.path+"/lo_res/")
		elif (self.mode == "RUN"):
			# We just need one multi-res sim!
			# TODO: change to multi-res when ready.
			self.sim = SmokeMultiRes(self.sim_w, self.sim_h)

		# Renderer for the sim.
		self.renderer = Renderer(self.sim_w, self.sim_h)

		# Creates our main texture to render the fluid to,
		# with no data bound to it.
		self.texture = self.createTexture()

		# Render the first frame of the sim.
		self.updateTexture()

		# Hack for window resize bug.
		self.first_iteration = True

	def initOpenGL(self):
		# Open GL Initialization.
		glutInit()
		# We are rendering to RBGA.
		glutInitDisplayMode(GLUT_RGBA)
		# Set up the window. Resize in the draw loop on first time.
		glutInitWindowSize(self.w-1, self.h-1)
		glutInitWindowPosition(0, 0)
		wind = glutCreateWindow("Fluid Sim: " + self.mode + " Mode")

		# Set our draw loop to the display function.
		glutDisplayFunc(self.drawLoop)
		glutIdleFunc(self.drawLoop)

		# Set up mouse interaction if we are not in view mode.
		if (self.mode != "VIEW"):
			glutMotionFunc(self.mouseControl)
			glutMouseFunc(self.mouseReleased)

	def drawLoop(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glViewport(0, 0, self.w, self.h)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0.0, self.w, 0.0, self.h, -1.0, 1.0)
		glMatrixMode (GL_MODELVIEW)
		glLoadIdentity()

		# Step and render the next frame of our simulation.
		self.updateTexture()

		glBindTexture(GL_TEXTURE_2D, self.texture)
		glEnable(GL_TEXTURE_2D)
		glEnableClientState(GL_VERTEX_ARRAY)
		glEnableClientState(GL_TEXTURE_COORD_ARRAY)

		self.fullscreenQuad()

		# Hack to fix window resize bug.
		if (self.first_iteration):
			glutReshapeWindow(self.w, self.h)
			self.first_iteration = False

		glutSwapBuffers()

	def createTexture(self):
		texture = glGenTextures(1)
		glPixelStorei(GL_UNPACK_ALIGNMENT,1)
		glBindTexture(GL_TEXTURE_2D, texture)
		# Texture parameters are part of the texture object, so you need to
		# specify them only once for a given texture object.
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		return texture

	def updateTexture(self):
		# Fetch our new texture data.
		if (self.mode == "VIEW"):
			# If we are viewing a pre-computed sim, we get
			# our density data from the file corresponding to the
			# current frame.
			try:
				density = np.load(self.path + format(self.frame, "0>9") + ".npz")['d']
			except:
				self.frame = 0
				density = np.load(self.path + format(self.frame, "0>9") + ".npz")['d']
			self.tex_array = self.renderer.render(np.transpose(density[1:-1,1:-1]))
			self.frame += 1
		else:
			# If we are running a sim to generate data or for a demo,
			# we get our density data from the sim. In the case of
			# generate data, we use the hi-res sim.
			self.tex_array = self.renderer.render(self.sim.step())

		# If we are generating data, we also need to step our low-res sim.
		if (self.mode == "GEN_DATA"):
			self.low_res_sim.step()

		# Finally, bind the texture array, wherever we got it from, to our
		# texture.
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.tex_array.shape[0],
			self.tex_array.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE,
			self.tex_array)

	# Renders fullscreen quad
	def fullscreenQuad(self):
		glBegin(GL_QUADS)
		glTexCoord2f(0, 0)
		glVertex2f(0, 0)
		glTexCoord2f(0, 1)
		glVertex2f(0, self.h)
		glTexCoord2f(1, 1)
		glVertex2f(self.w, self.h)
		glTexCoord2f(1, 0)
		glVertex2f(self.w, 0)
		glEnd()

	# TODO: make this work for both sims, and make the force application area
	# SCALE INVARIANT!
	def mouseControl(self, mx, my):
		mx = mx
		my = self.h - my

		if (self.mouse_x == None or self.mouse_y == None):
			self.mouse_x = mx
			self.mouse_y = my
			return

		dx = mx - self.mouse_x
		dy = my - self.mouse_y

		xloc = int((mx/self.w)*self.sim_w)
		yloc = int((my/self.h)*self.sim_h)
		scaled_dx = self.sim_w*(dx/self.w)
		scaled_dy = self.sim_h*(dy/self.h)

		self.sim.update_mouse_force(xloc, yloc, scaled_dx, scaled_dy)

		if (self.mode == "GEN_DATA"):
			xloc = int((mx/self.w)*self.lr_w)
			yloc = int((my/self.h)*self.lr_h)
			scaled_dx = self.lr_w*(dx/self.w)
			scaled_dy = self.lr_h*(dy/self.h)
			self.low_res_sim.update_mouse_force(xloc, yloc, \
				scaled_dx, scaled_dy)

		self.mouse_x = mx
		self.mouse_y = my

	def mouseReleased(self, button, state, x, y):
		if (state == GLUT_UP):
			self.mouse_x = None
			self.mouse_y = None

	def run(self):
		glutMainLoop()

def main():
	if (len(sys.argv) < 2):
		print("USAGE: python app.py <run mode, one of [RUN, VIEW, GEN_DATA]> <path (if using VIEW or GEN_DATA)>")
		exit()
	mode = sys.argv[1]
	path = ""
	if (mode == "GEN_DATA" or mode=="VIEW"):
		path = sys.argv[2]
	tex_renderer = TexRenderer(600, 600, 150, 150, mode=mode, path=path)
	tex_renderer.run()

main()
