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

	def __init__(self, w, h, generate_data=False, view_data=False, path=""):

		self.generate_data = generate_data
		self.view_data = view_data
		self.path = path

		# Open GL Initialization.
		glutInit()
		# We are rendering to RBGA.
		glutInitDisplayMode(GLUT_RGBA)
		# Set up the window. Resize in the draw loop on first time.
		glutInitWindowSize(w-1, h-1)
		glutInitWindowPosition(0, 0)
		wind = glutCreateWindow("Fluid Sim")

		# Set our draw loop to the display function.
		glutDisplayFunc(self.drawLoop)
		glutIdleFunc(self.drawLoop)
		if (not self.view_data):
			glutMotionFunc(self.mouseControl)
			glutMouseFunc(self.mouseReleased)

		# Width and height of the window.
		self.w = w
		self.h = h

		# Width and height of the sim.

		# Mouse position.
		self.mouse_x = None
		self.mouse_y = None

		self.sim_w = 100
		self.sim_h = 100
		if (self.view_data):
			# We need no renderer for this sim! But we do
			# need a frame number
			self.frame = 0
		elif (self.generate_data):
			# Low res sim parameters
			self.lr_w = self.sim_w/5
			self.lr_h = self.sim_h/5
			# We need two sims: a high res "real one"
			# and a low res "fake" one.
			self.sim = Smoke(self.sim_w, self.sim_h, \
				save_data=True, path=self.path+"/hi_res/")
			self.low_res_sim = Smoke(int(self.lr_w), int(self.lr_h), \
				save_data=True, path=self.path+"/lo_res/")
		else:
			# We just need one multi-res sim!
			# TODO: change to multi-res when ready.
			self.sim = Smoke(self.sim_w, self.sim_h)

		# Renderer for the sim.
		self.renderer = Renderer(self.sim_w, self.sim_h)

		# Render the first frame of the sim, set it to our active texture.
		# Texture will be in row major order.
		if (self.view_data):
			density = np.load(self.path + format(self.frame, "0>9") + ".npz")['d']
			self.tex_array = self.renderer.render(np.transpose(density[1:-1,1:-1]))
			self.frame += 1
		elif (self.generate_data):
			self.tex_array = self.renderer.render(self.sim.step())
			self.low_res_sim.step()
		else:
			self.tex_array = self.renderer.render(self.sim.step())

		self.texture = self.textureFromNumpy(self.tex_array)

		# Hack for window resize bug.
		self.first_iteration = True

	def textureFromNumpy(self, tex_array):
		texture = glGenTextures(1)
		glPixelStorei(GL_UNPACK_ALIGNMENT,1)
		glBindTexture(GL_TEXTURE_2D, texture)
		# Texture parameters are part of the texture object, so you need to
		# specify them only once for a given texture object.
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex_array.shape[0], tex_array.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, tex_array)
		return texture

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

	def updateTexture(self):
		if (self.view_data):
			try:
				density = np.load(self.path + format(self.frame, "0>9") + ".npz")['d']
			except:
				self.frame = 0
				density = np.load(self.path + format(self.frame, "0>9") + ".npz")['d']
			self.tex_array = self.renderer.render(np.transpose(density[1:-1,1:-1]))
			self.frame += 1
		elif (self.generate_data):
			self.tex_array = self.renderer.render(self.sim.step())
			self.low_res_sim.step()
		else:
			self.tex_array = self.renderer.render(self.sim.step())
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

		self.sim.F_mouse[max(xloc-1,0):min(xloc+1, self.sim_w-1), \
			max(yloc-1,0):min(yloc+1, self.sim_h-1)] += self.sim.force_scale * np.array([dx, dy])

		if (self.generate_data):
			xloc = int((mx/self.w)*self.lr_w)
			yloc = int((my/self.h)*self.lr_h)
			self.low_res_sim.F_mouse[max(xloc-3,0):min(xloc+3, self.lr_w-1), \
				max(yloc-3,0):min(yloc+3, self.lr_h-1)] += self.sim.force_scale * np.array([dx, dy])

		self.mouse_x = mx
		self.mouse_y = my

	def mouseReleased(self, button, state, x, y):
		if (state == GLUT_UP):
			self.mouse_x = None
			self.mouse_y = None

	def run(self):
		glutMainLoop()


def main():
	# First arg can either be "GEN_DATA", "RUN", or "VIEW_DATA".
	run_type = sys.argv[1]
	path = ""
	if (run_type == "GEN_DATA" or run_type=="VIEW_DATA"):
		path = sys.argv[2]
	tex_renderer = TexRenderer(600, 600, \
		generate_data=(run_type == "GEN_DATA"), \
		view_data=(run_type == "VIEW_DATA"), path=path)
	tex_renderer.run()

main()
